"""Validation-only HPO and Top-1 delivery for the Listwise track."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline import FM  # noqa: E402
from data import FIELDS, encode, load  # noqa: E402
from evaluate import evaluate  # noqa: E402

from model import ExposureGroup, ListwiseFM, group_user_exposures  # noqa: E402


@dataclass(frozen=True)
class Config:
    k: int
    lr: float
    optimizer: str
    weight_decay: float
    score_temperature: float
    target_temperature: float
    anchor_mix: float
    ndcg_weight: float
    rank_temperature: float
    cutoff_temperature: float
    position_weight: float
    sort_temperature: float
    metric_weighting: bool
    warmup_steps: int
    update_embeddings: bool = True
    user_batch_size: int = 256
    validation_interval: int = 10
    max_epochs: int = 2
    patience_checks: int = 6
    hard_negative_cap: int | None = None
    hard_fraction: float = 1.0
    sampling_strategy: str = "hard_random"


DEFAULT_SEARCH = (
    *(
        Config(
            k=16, lr=1e-6, optimizer="adamw", weight_decay=0.0,
            score_temperature=1.0, target_temperature=0.5, anchor_mix=1.0,
            ndcg_weight=0.0, rank_temperature=0.5, cutoff_temperature=0.5,
            position_weight=weight, sort_temperature=0.5,
            metric_weighting=True, warmup_steps=50, update_embeddings=True,
            user_batch_size=256, validation_interval=20, max_epochs=2,
            patience_checks=6, hard_negative_cap=64,
            sampling_strategy="top5_boundary",
        )
        for weight in (0.25, 0.50, 0.75, 1.00)
    ),
    *(
        Config(
            k=16, lr=1e-6, optimizer="adamw", weight_decay=0.0,
            score_temperature=1.0, target_temperature=0.5, anchor_mix=1.0,
            ndcg_weight=0.0, rank_temperature=0.5, cutoff_temperature=0.5,
            position_weight=weight, sort_temperature=0.5,
            metric_weighting=True, warmup_steps=50, update_embeddings=True,
            user_batch_size=256, validation_interval=20, max_epochs=2,
            patience_checks=6, hard_negative_cap=32,
            sampling_strategy="multi_slate",
        )
        for weight in (0.25, 0.50, 0.75, 1.00)
    ),
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_hard_user_lists(
    groups: list[ExposureGroup],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    cap: int | None,
    hard_fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[ExposureGroup]:
    """Keep every positive plus a hard/random mix of same-user negatives.

    This constructs one listwise slate per user and never creates pairs or mixes
    users. Lists already at or below the cap remain unchanged.
    """

    if cap is None:
        return groups
    if cap < 2:
        raise ValueError("hard-negative list cap must be at least 2")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard fraction must be between 0 and 1")
    if hard_fraction < 1.0 and rng is None:
        raise ValueError("mixed hard/random lists require a random generator")
    selected_groups = []
    for group in groups:
        rows = group.row_indices
        if len(rows) <= cap:
            selected_groups.append(group)
            continue
        positive_rows = rows[labels[rows] == 1]
        negative_rows = rows[labels[rows] == 0]
        negative_count = max(1, cap - len(positive_rows))
        if len(negative_rows) > negative_count:
            hard_count = min(negative_count, int(round(negative_count * hard_fraction)))
            if hard_count:
                hard_positions = np.argpartition(
                    baseline_scores[negative_rows], -hard_count
                )[-hard_count:]
                hard_rows = negative_rows[hard_positions]
            else:
                hard_positions = np.empty(0, dtype=np.int64)
                hard_rows = np.empty(0, dtype=negative_rows.dtype)
            random_count = negative_count - hard_count
            if random_count:
                available_mask = np.ones(len(negative_rows), dtype=bool)
                available_mask[hard_positions] = False
                random_rows = rng.choice(
                    negative_rows[available_mask], size=random_count, replace=False
                )
                negative_rows = np.concatenate((hard_rows, random_rows))
            else:
                negative_rows = hard_rows
        selected_rows = np.sort(np.concatenate((positive_rows, negative_rows)))
        selected_groups.append(
            ExposureGroup(group.user_id, selected_rows, group.positives)
        )
    return selected_groups


def build_top5_boundary_lists(
    groups: list[ExposureGroup],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    cap: int,
    rng: np.random.Generator,
    boundary_count: int = 24,
) -> list[ExposureGroup]:
    """S-04: positives plus negatives around the baseline top-5 cutoff."""

    selected_groups = []
    for group in groups:
        rows = group.row_indices
        if len(rows) <= cap:
            selected_groups.append(group)
            continue
        positive_rows = rows[labels[rows] == 1]
        negative_rows = rows[labels[rows] == 0]
        negative_count = max(1, cap - len(positive_rows))
        ranked_scores = np.sort(baseline_scores[rows])[::-1]
        cutoff = ranked_scores[min(4, len(ranked_scores) - 1)]
        distances = np.abs(baseline_scores[negative_rows] - cutoff)
        take_boundary = min(boundary_count, negative_count)
        boundary_positions = np.argsort(distances, kind="stable")[:take_boundary]
        boundary_rows = negative_rows[boundary_positions]
        random_count = negative_count - take_boundary
        if random_count:
            available = np.ones(len(negative_rows), dtype=bool)
            available[boundary_positions] = False
            random_rows = rng.choice(
                negative_rows[available], size=random_count, replace=False
            )
            selected_negatives = np.concatenate((boundary_rows, random_rows))
        else:
            selected_negatives = boundary_rows
        selected_groups.append(
            ExposureGroup(
                group.user_id,
                np.sort(np.concatenate((positive_rows, selected_negatives))),
                group.positives,
            )
        )
    return selected_groups


def build_multi_slates(
    groups: list[ExposureGroup],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> list[ExposureGroup]:
    """S-06: two same-user slates, one boundary-hard and one random."""

    slates = []
    for group in groups:
        rows = group.row_indices
        positive_rows = rows[labels[rows] == 1]
        negative_rows = rows[labels[rows] == 0]
        negative_count = min(len(negative_rows), max(1, cap - len(positive_rows)))
        ranked_scores = np.sort(baseline_scores[rows])[::-1]
        cutoff = ranked_scores[min(4, len(ranked_scores) - 1)]
        distances = np.abs(baseline_scores[negative_rows] - cutoff)
        hard_positions = np.argsort(distances, kind="stable")[:negative_count]
        hard_rows = negative_rows[hard_positions]
        available = np.ones(len(negative_rows), dtype=bool)
        available[hard_positions] = False
        random_pool = negative_rows[available]
        if len(random_pool) >= negative_count:
            random_rows = rng.choice(random_pool, size=negative_count, replace=False)
        else:
            random_rows = rng.choice(negative_rows, size=negative_count, replace=False)
        for negative_selection in (hard_rows, random_rows):
            slates.append(
                ExposureGroup(
                    group.user_id,
                    np.sort(np.concatenate((positive_rows, negative_selection))),
                    group.positives,
                )
            )
    return slates


def build_training_slates(
    strategy: str,
    groups: list[ExposureGroup],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    cap: int | None,
    hard_fraction: float,
    rng: np.random.Generator,
) -> list[ExposureGroup]:
    if cap is None:
        return groups
    if strategy == "top5_boundary":
        return build_top5_boundary_lists(groups, labels, baseline_scores, cap, rng)
    if strategy == "multi_slate":
        return build_multi_slates(groups, labels, baseline_scores, cap, rng)
    if strategy == "hard_random":
        return build_hard_user_lists(
            groups, labels, baseline_scores, cap, hard_fraction, rng
        )
    raise ValueError(f"unknown sampling strategy: {strategy}")


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_users: list[str],
    dim: int,
    output_dir: Path,
    seed: int = 0,
    max_epochs: int = 40,
    patience: int = 4,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train the shared pointwise FM baseline without ever evaluating test."""

    model = FM(dim, k=16, lr=1e-3, l2=1e-6, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -np.inf
    best_state = None
    best_metrics = None
    best_epoch = 0
    bad_epochs = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(y_train))
        losses = [
            model.step(X_train[order[start : start + 8192]], y_train[order[start : start + 8192]])
            for start in range(0, len(order), 8192)
        ]
        metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
        print(
            f"[baseline] epoch {epoch:02d} loss={np.mean(losses):.5f} "
            f"valid_primary={metrics['primary']:.6f} GAUC={metrics['GAUC']:.6f} "
            f"nDCG@5={metrics['nDCG@5']:.6f}",
            flush=True,
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = {"V": model.V.copy(), "W": model.W.copy(), "b": np.asarray(model.b)}
            best_metrics = metrics
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    assert best_state is not None and best_metrics is not None
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "baseline_seed0.npz"
    np.savez_compressed(
        checkpoint_path,
        **best_state,
        seed=np.asarray(seed),
        best_epoch=np.asarray(best_epoch),
    )
    result = {
        "track": "shared_baseline",
        "method": "fm_pointwise_log_loss",
        "hyperparameters": {
            "k": 16, "lr": 1e-3, "l2": 1e-6, "batch_size": 8192,
            "max_epochs": max_epochs, "patience": patience,
        },
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "valid_GAUC": float(best_metrics["GAUC"]),
        "valid_nDCG@5": float(best_metrics["nDCG@5"]),
        "valid_primary": float(best_metrics["primary"]),
        "training_time_sec": time.time() - started,
        "git_commit": git_commit(),
        "test_metrics_used": False,
    }
    (metrics_dir / "baseline_seed0.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return best_state, result


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_users: list[str],
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_users: list[str],
    dim: int,
    config: Config,
    run_id: str,
    output_dir: Path,
    initial_state: dict[str, np.ndarray],
    baseline_valid_primary: float,
    seed: int = 0,
) -> dict:
    all_groups = group_user_exposures(train_users, y_train)
    base_groups = [g for g in all_groups if 0 < g.positives < len(g.row_indices)]
    model_keys = {
        "k",
        "lr",
        "optimizer",
        "weight_decay",
        "score_temperature",
        "target_temperature",
        "anchor_mix",
        "ndcg_weight",
        "rank_temperature",
        "cutoff_temperature",
        "position_weight",
        "sort_temperature",
        "warmup_steps",
        "update_embeddings",
    }
    model = ListwiseFM(
        dim,
        seed=seed,
        **{key: value for key, value in asdict(config).items() if key in model_keys},
    )
    model.load_state_dict(initial_state)
    baseline_train_scores = model.predict(X_train)
    mean_positives = float(np.mean([group.positives for group in base_groups]))
    if config.metric_weighting:
        weights = {
            group.user_id: 0.5 + 0.5 * min(group.positives / mean_positives, 5.0)
            for group in base_groups
        }
    else:
        weights = {group.user_id: 1.0 for group in base_groups}
    rng = np.random.default_rng(seed)
    best_primary = -np.inf
    best_state = None
    best_epoch = 0
    best_step = 0
    best_metrics = None
    bad_checks = 0
    stop_training = False
    started = time.time()
    log_rows = []
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        groups = build_training_slates(
            config.sampling_strategy,
            base_groups,
            y_train,
            baseline_train_scores,
            config.hard_negative_cap,
            config.hard_fraction,
            rng,
        )
        batches_per_epoch = (
            len(groups) + config.user_batch_size - 1
        ) // config.user_batch_size
        order = rng.permutation(len(groups))
        interval_losses = []
        for start in range(0, len(order), config.user_batch_size):
            batch_groups = [groups[i] for i in order[start : start + config.user_batch_size]]
            row_indices = np.concatenate([group.row_indices for group in batch_groups])
            group_sizes = [len(group.row_indices) for group in batch_groups]
            group_weights = [weights[group.user_id] for group in batch_groups]
            interval_losses.append(
                model.step(
                    X_train[row_indices],
                    y_train[row_indices],
                    group_sizes,
                    group_weights,
                    baseline_train_scores[row_indices],
                )
            )
            global_step += 1
            end_of_epoch = (start // config.user_batch_size) + 1 == batches_per_epoch
            if global_step % config.validation_interval and not end_of_epoch:
                continue

            metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
            fractional_epoch = (epoch - 1) + min(
                1.0, ((start // config.user_batch_size) + 1) / batches_per_epoch
            )
            record = {
                "step": global_step,
                "epoch": fractional_epoch,
                "loss": float(np.mean(interval_losses)),
                "valid_GAUC": float(metrics["GAUC"]),
                "valid_nDCG@5": float(metrics["nDCG@5"]),
                "valid_primary": float(metrics["primary"]),
                "elapsed_time_sec": time.time() - started,
            }
            interval_losses = []
            log_rows.append(record)
            print(
                f"[{run_id}] step={global_step:03d} epoch={fractional_epoch:.2f} "
                f"loss={record['loss']:.5f} valid_primary={metrics['primary']:.6f} "
                f"GAUC={metrics['GAUC']:.6f} nDCG@5={metrics['nDCG@5']:.6f}",
                flush=True,
            )
            if metrics["primary"] > best_primary + 1e-5:
                best_primary = metrics["primary"]
                best_state = model.state_dict()
                best_epoch = fractional_epoch
                best_step = global_step
                best_metrics = metrics
                bad_checks = 0
            else:
                bad_checks += 1
            if bad_checks >= config.patience_checks:
                stop_training = True
                break
        if stop_training:
            break

    assert best_state is not None and best_metrics is not None
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.npz"
    np.savez_compressed(
        checkpoint_path,
        **best_state,
        config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        seed=np.asarray(seed),
        best_epoch=np.asarray(best_epoch),
        best_step=np.asarray(best_step),
    )
    elapsed = time.time() - started
    result = {
        "track": "listwise",
        "method": (
            "fm_user_position_discounted_listnet_finetune"
            if config.position_weight
            else "fm_user_listnet_approx_ndcg_finetune"
        ),
        "hyperparameters": asdict(config),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
        "valid_GAUC": float(best_metrics["GAUC"]),
        "valid_nDCG@5": float(best_metrics["nDCG@5"]),
        "valid_primary": float(best_metrics["primary"]),
        "gain": float(best_metrics["primary"] - baseline_valid_primary),
        "training_time_sec": elapsed,
        "git_commit": git_commit(),
        "grouping": f"same-user listwise slate; sampling={config.sampling_strategy}",
        "train_users_total": len(all_groups),
        "train_users_discriminative": len(base_groups),
        "test_metrics_used": False,
        "initialized_from": "results_by_track/listwise/checkpoints/baseline_seed0.npz",
    }
    (metrics_dir / f"{run_id}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (log_dir / f"{run_id}.json").write_text(
        json.dumps(log_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "KuaiRand-Pure/data"))
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--max_runs", type=int, default=len(DEFAULT_SEARCH))
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience_checks", type=int, default=None)
    parser.add_argument(
        "--output_dir", default=str(REPO_ROOT / "results_by_track/listwise")
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading shared data from {args.data_dir}", flush=True)
    splits = load(args.data_dir)
    encoded, dim = encode(splits)
    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    # Shared encode() constructs every official split, but HPO never accesses or
    # scores encoded["test"] and therefore cannot use test metrics for selection.
    print(
        f"train_rows={len(y_train)} valid_rows={len(y_valid)} dim={dim} fields={FIELDS}",
        flush=True,
    )

    baseline_state, baseline_result = train_baseline(
        X_train,
        y_train,
        X_valid,
        y_valid,
        valid_users,
        dim,
        output_dir,
        seed=args.seed,
    )

    results = []
    for index, original_config in enumerate(DEFAULT_SEARCH[: args.max_runs], start=1):
        values = asdict(original_config)
        if args.max_epochs is not None:
            values["max_epochs"] = args.max_epochs
        if args.patience_checks is not None:
            values["patience_checks"] = args.patience_checks
        config = Config(**values)
        results.append(
            train_one(
                X_train,
                y_train,
                train_users,
                X_valid,
                y_valid,
                valid_users,
                dim,
                config,
                f"hpo_v6_{index:02d}",
                output_dir,
                baseline_state,
                baseline_result["valid_primary"],
                seed=args.seed,
            )
        )

    # Rank every retained HPO checkpoint, including earlier validation-only
    # search stages. Missing non-final checkpoints are intentionally ignored.
    retained = {row["checkpoint_path"]: row for row in results}
    for metrics_path in (output_dir / "metrics").glob("hpo*.json"):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = REPO_ROOT / row["checkpoint_path"]
        if row.get("track") == "listwise" and checkpoint.is_file():
            retained[row["checkpoint_path"]] = row
    # This is the only Top-1 ordering key. Test metrics do not exist in results.
    top1 = sorted(
        retained.values(),
        key=lambda row: (-row["valid_primary"], row["checkpoint_path"]),
    )[0]
    (output_dir / "top1.json").write_text(
        json.dumps(top1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Top 1 by validation primary only:", flush=True)
    print(
        f"{top1['checkpoint_path']} valid_primary={top1['valid_primary']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
