"""Leakage-safe, low-capacity context-aware composition layer.

The layer is intentionally a small linear gate over frozen family scores. It
does not retrain any family model.  Its inputs are all five family predictions
plus train-derived exposure/history context; its only trainable state is the
global gate logits and a regularized context-to-gate matrix.
"""
import json
import gc
import hashlib
import os

import numpy as np

from data import encode
from rich_data import encode_rich, load_rich
from validation_only import load_train_valid
from evaluate import evaluate
try:
    from dataset_config import data_dir, outputs_dir
except ImportError:
    from agent.dataset_config import data_dir, outputs_dir

from baseline import FM
from modules.composition import (FAMILY_ORDER, _checkpoint, _predict,
                                 _predict_history_external_train_valid,
                                 _user_zscore, _load_registry)


FEATURE_SCHEMA = "safe_context_v1"
FEATURE_NAMES = (
    "user_exposure_log1p", "video_exposure_log1p", "author_exposure_log1p",
    "tab_exposure_log1p", "user_history_log1p", "user_tab_history_log1p",
    "user_author_history_log1p", "duration_quantile", "family_prediction_std",
    "family_prediction_range", "family_prediction_mean_abs_disagreement",
)


def analyze_train_schema(train_rows):
    """Return an auditable schema summary without inspecting validation labels."""
    keys = sorted(set().union(*(row.keys() for row in train_rows))) if train_rows else []
    return {
        "schema": FEATURE_SCHEMA,
        "source": "train_rows_only",
        "base_fields": ["date", "user_id", "video_id", "author_id", "tab", "duration_ms"],
        "available_train_keys": keys,
        "selected_features": list(FEATURE_NAMES),
        "excluded_feedback": ["long_view", "is_click", "is_like", "is_follow",
                               "is_comment", "is_forward", "play_time_ms"],
        "leakage_policy": "counts and prior histories are computed from train only",
    }


def _feature_stats(train_rows):
    users = {}; videos = {}; authors = {}; tabs = {}
    for row in train_rows:
        _, user, video, author, tab, _ = row["base"]
        users[user] = users.get(user, 0) + 1
        videos[video] = videos.get(video, 0) + 1
        authors[author] = authors.get(author, 0) + 1
        tabs[tab] = tabs.get(tab, 0) + 1
    durations = np.asarray([row["base"][5] for row in train_rows], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1]) if len(durations) else np.zeros(9)
    return users, videos, authors, tabs, edges


def build_context_features(train_rows, rows, predictions=None, mean=None, std=None):
    """Build only inference-available, train-derived numeric context features."""
    users, videos, authors, tabs, edges = _feature_stats(train_rows)
    values = []
    for row in rows:
        _, user, video, author, tab, duration = row["base"]
        hist = row.get("hist", (0, 0, 0))
        values.append([
            np.log1p(users.get(user, 0)), np.log1p(videos.get(video, 0)),
            np.log1p(authors.get(author, 0)), np.log1p(tabs.get(tab, 0)),
            np.log1p(hist[0]), np.log1p(hist[1]), np.log1p(hist[2]),
            float(np.searchsorted(edges, duration)) / 9.0,
        ])
    matrix = np.asarray(values, dtype=np.float64)
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=np.float64)
        if len(predictions) != len(matrix) or predictions.shape[1] != len(FAMILY_ORDER):
            raise ValueError("context predictions must align with rows and all five families")
        center = predictions.mean(axis=1)
        disagreement = np.column_stack([
            predictions.std(axis=1), predictions.max(axis=1) - predictions.min(axis=1),
            np.mean(np.abs(predictions - center[:, None]), axis=1),
        ])
        matrix = np.column_stack([matrix, disagreement])
    elif len(matrix):
        # A context gate must always have the same schema at fit and predict.
        matrix = np.column_stack([matrix, np.zeros((len(matrix), 3), dtype=np.float64)])
    if mean is None:
        mean = matrix.mean(axis=0) if len(matrix) else np.zeros(len(FEATURE_NAMES))
    if std is None:
        std = matrix.std(axis=0) if len(matrix) else np.ones(len(FEATURE_NAMES))
    std = np.where(np.asarray(std) > 1e-12, std, 1.0)
    return (matrix - np.asarray(mean)) / std, np.asarray(mean), std


def load_all_family_predictions(root, rich_rows=None):
    """Load aligned train/valid predictions for all five frozen families."""
    data_path = data_dir()
    splits = load_train_valid(data_path)
    rich_train, rich_valid = rich_rows if rich_rows is not None else load_rich(data_path)
    if len(rich_train) != len(splits["train"]) or len(rich_valid) != len(splits["valid"]):
        raise ValueError("rich and official validation rows are not aligned")
    train_users = np.asarray([row["base"][1] for row in rich_train], dtype=object)
    valid_users = np.asarray([row["base"][1] for row in rich_valid], dtype=object)
    train_predictions = {}; valid_predictions = {}
    basic, dimension = encode({**splits, "test": []})
    train_predictions["bpr_fm"] = _predict_chunked(root, "bpr_fm", basic["train"][0], dimension)
    valid_predictions["bpr_fm"] = _predict_chunked(root, "bpr_fm", basic["valid"][0], dimension)
    del basic
    gc.collect()
    for family in FAMILY_ORDER[1:]:
        if family == "history_fm":
            train_base = [x["base"] + (x["y"],) for x in rich_train]
            valid_base = [x["base"] + (x["y"],) for x in rich_valid]
            train_scores, valid_scores = _predict_history_external_train_valid(
                root, train_base, valid_base)
            train_predictions[family] = train_scores
            valid_predictions[family] = valid_scores
            continue
        train_encoded, valid_encoded, dimension = encode_rich(
            rich_train, rich_valid, include_history=(family == "history_fm"))
        train_predictions[family] = _predict_chunked(root, family, train_encoded[0], dimension)
        valid_predictions[family] = _predict_chunked(root, family, valid_encoded[0], dimension)
        del train_encoded, valid_encoded
        gc.collect()
    train_matrix = np.column_stack([_user_zscore(train_predictions[f], train_users)
                                    for f in FAMILY_ORDER])
    valid_matrix = np.column_stack([_user_zscore(valid_predictions[f], valid_users)
                                    for f in FAMILY_ORDER])
    train_labels = np.asarray([row["y"] for row in rich_train], dtype=np.float64)
    valid_labels = np.asarray([row["y"] for row in rich_valid], dtype=np.float64)
    return {"train_rows": rich_train, "valid_rows": rich_valid,
            "train_users": train_users, "valid_users": valid_users,
            "train_labels": train_labels, "valid_labels": valid_labels,
            "train_predictions": train_matrix, "valid_predictions": valid_matrix}


def _predict_chunked(root, experiment, features, dimension, chunk_size=50000):
    """Predict in bounded chunks to avoid interaction-tensor memory spikes."""
    path = _checkpoint(root, experiment)
    with np.load(path, allow_pickle=False) as saved:
        if saved["V"].shape[0] != dimension or saved["W"].shape != (dimension,):
            raise ValueError(f"{experiment} checkpoint is incompatible with its encoder")
        model = FM(dimension, k=saved["V"].shape[1], seed=0)
        model.V[...] = saved["V"]; model.W[...] = saved["W"]; model.b = np.float32(saved["b"])
    result = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        result[start:end] = model.predict(features[start:end]).astype(np.float64, copy=False)
    return result


def _selected(config):
    components = config.get("components")
    if components != list(FAMILY_ORDER) or config.get("composition_code") != "11111":
        raise ValueError("context composition requires all five families and code 11111")
    selected = np.ones(len(FAMILY_ORDER), dtype=bool)
    weights = np.asarray(config["weights"], dtype=np.float64)
    if len(weights) != len(FAMILY_ORDER) or np.any(weights <= 0):
        raise ValueError("context composition requires five positive family weights")
    weights /= weights.sum()
    return selected, weights


def _gate(context, base_logits, gate_matrix, selected, temperature):
    logits = np.asarray(base_logits)[None, :] + np.asarray(context) @ np.asarray(gate_matrix)
    logits = logits / max(float(temperature), 1e-6)
    logits[:, ~selected] = -60.0
    logits -= logits.max(axis=1, keepdims=True)
    gate = np.exp(np.clip(logits, -60.0, 30.0))
    gate[:, ~selected] = 0.0
    gate /= np.maximum(gate.sum(axis=1, keepdims=True), 1e-12)
    return gate


def _scores(predictions, context, base_logits, gate_matrix, selected, temperature):
    gate = _gate(context, base_logits, gate_matrix, selected, temperature)
    return np.sum(gate * predictions, axis=1), gate


def _pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    by_user = {}
    for index, user in enumerate(users):
        by_user.setdefault(user, []).append(index)
    result = []
    for indices in by_user.values():
        positives = [i for i in indices if labels[i] > 0]
        negatives = [i for i in indices if labels[i] <= 0]
        if not negatives:
            continue
        for positive in positives:
            result.append((positive, int(rng.choice(negatives))))
    return np.asarray(result, dtype=np.int64)


def fit_context_composition(root, config, rich_rows=None):
    """Fit a regularized BPR gate on train rows and return validation scores."""
    if config.get("composition_model") != "context_gate":
        raise ValueError("context composer requires composition_model=context_gate")
    if config.get("composition_loss", "bpr_gate") != "bpr_gate":
        raise ValueError("only the reviewed bpr_gate composition loss is allowed")
    data = load_all_family_predictions(root, rich_rows=rich_rows)
    train_context, mean, std = build_context_features(
        data["train_rows"], data["train_rows"], data["train_predictions"])
    valid_context, _, _ = build_context_features(
        data["train_rows"], data["valid_rows"], data["valid_predictions"], mean, std)
    selected, weights = _selected(config)
    base = np.full(len(FAMILY_ORDER), -60.0, dtype=np.float64)
    base[selected] = np.log(np.maximum(weights[selected], 1e-8))
    gate_matrix = np.zeros((len(FEATURE_NAMES), len(FAMILY_ORDER)), dtype=np.float64)
    pairs = _pairs(data["train_users"], data["train_labels"], int(config.get("composition_seed", 0)))
    if len(pairs) < 2:
        raise ValueError("context composer needs at least two train user groups")
    rng = np.random.default_rng(int(config.get("composition_seed", 0)))
    order = rng.permutation(len(pairs)); cut = max(1, int(len(order) * 0.8))
    fit_pairs, monitor_pairs = pairs[order[:cut]], pairs[order[cut:]]
    lr = float(config.get("gate_lr", 0.02)); l2 = float(config.get("gate_l2", 0.01))
    epochs = int(config.get("gate_epochs", 6)); patience = int(config.get("gate_patience", 2))
    temperature = float(config.get("gate_temperature", 1.0)); batch_size = 2048
    best = None; best_loss = float("inf"); bad = 0
    for _ in range(epochs):
        shuffled = rng.permutation(len(fit_pairs)); losses = []
        for start in range(0, len(shuffled), batch_size):
            pair = fit_pairs[shuffled[start:start + batch_size]]
            pi, ni = pair[:, 0], pair[:, 1]
            sp, gp = _scores(data["train_predictions"][pi], train_context[pi], base,
                              gate_matrix, selected, temperature)
            sn, gn = _scores(data["train_predictions"][ni], train_context[ni], base,
                              gate_matrix, selected, temperature)
            delta = np.clip(sn - sp, -30.0, 30.0)
            q = 1.0 / (1.0 + np.exp(-delta))
            dgp = gp * (data["train_predictions"][pi] - sp[:, None])
            dgn = gn * (data["train_predictions"][ni] - sn[:, None])
            grad_logits_p = -q[:, None] * dgp
            grad_logits_n = q[:, None] * dgn
            grad_base = (grad_logits_p + grad_logits_n).mean(axis=0)
            grad_gate = (train_context[pi].T @ grad_logits_p +
                         train_context[ni].T @ grad_logits_n) / max(len(pair), 1)
            grad_base[~selected] = 0.0
            grad_gate[:, ~selected] = 0.0
            base[selected] -= lr * (grad_base[selected] + l2 * base[selected])
            gate_matrix[:, selected] -= lr * (grad_gate[:, selected] + l2 * gate_matrix[:, selected])
            losses.append(float(np.mean(np.logaddexp(0.0, delta))))
        monitor_p = monitor_pairs[:, 0]; monitor_n = monitor_pairs[:, 1]
        smp, _ = _scores(data["train_predictions"][monitor_p], train_context[monitor_p], base,
                         gate_matrix, selected, temperature)
        smn, _ = _scores(data["train_predictions"][monitor_n], train_context[monitor_n], base,
                         gate_matrix, selected, temperature)
        monitor_loss = float(np.mean(np.logaddexp(0.0, smn - smp)))
        monitor_loss += l2 * float(np.sum(gate_matrix[:, selected] ** 2))
        if monitor_loss < best_loss - 1e-6:
            best_loss = monitor_loss; best = (base.copy(), gate_matrix.copy()); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    base, gate_matrix = best
    valid_scores, valid_gate = _scores(data["valid_predictions"], valid_context, base,
                                       gate_matrix, selected, temperature)
    state = {"base_logits": base, "gate_matrix": gate_matrix,
             "feature_mean": mean, "feature_std": std,
             "selected": selected, "temperature": temperature,
             "feature_schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES,
             "loss": "bpr_gate", "train_monitor_loss": best_loss,
             "schema_analysis": analyze_train_schema(data["train_rows"]),
             "gate_mean": valid_gate.mean(axis=0)}
    return data, state, valid_scores


def save_context_checkpoint(root, state, config, metrics):
    """Persist the learned composition gate separately from family checkpoints."""
    out_dir = outputs_dir()
    os.makedirs(out_dir, exist_ok=True)
    seed = int(config.get("composition_seed", 0))
    path = os.path.join(out_dir, f"composition_context_candidate_{seed:03d}.npz")
    metadata = {"config": config, "feature_schema": state["feature_schema"],
                "feature_names": list(state["feature_names"]), "loss": state["loss"],
                "metrics": {k: float(v) for k, v in metrics.items()},
                "schema_analysis": state["schema_analysis"]}
    np.savez_compressed(path, base_logits=state["base_logits"],
                        gate_matrix=state["gate_matrix"], feature_mean=state["feature_mean"],
                        feature_std=state["feature_std"], selected=state["selected"],
                        temperature=np.asarray([state["temperature"]]),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return path


def predict_from_context_checkpoint(root, checkpoint, split="valid"):
    if split != "valid":
        raise PermissionError("context composition prediction is validation-only until manual finalization")
    checkpoint = os.path.abspath(os.path.join(root, checkpoint)) if not os.path.isabs(checkpoint) else checkpoint
    if os.path.commonpath([root, checkpoint]) != root or not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata.get("feature_schema") != FEATURE_SCHEMA or metadata.get("loss") != "bpr_gate":
            raise ValueError("unsupported context composition checkpoint")
        base = saved["base_logits"]; matrix = saved["gate_matrix"]
        mean = saved["feature_mean"]; std = saved["feature_std"]
        selected = saved["selected"].astype(bool); temperature = float(saved["temperature"][0])
    data = load_all_family_predictions(root)
    context, _, _ = build_context_features(
        data["train_rows"], data["valid_rows"], data["valid_predictions"], mean, std)
    scores, _ = _scores(data["valid_predictions"], context, base, matrix, selected, temperature)
    return scores


# ---------------------------------------------------------------------------
# Task-based composition layer.  This is separate from the earlier reviewed
# context_gate implementation so old artifacts remain readable, while the
# new workflow has one explicit, all-five-family contract.
PURE_FEATURES = ("duration_ms", "tab", "hourmin", "user_history_count",
                 "video_exposure_count")
INTERACTIONS = ("p_bpr_duration", "p_listwise_duration",
                "p_history_user_history", "p_cwm_duration", "p_multitask_tab")


def _task_feature_values(train_rows, rows, predictions, feature_set, interaction):
    """Build row features from train-derived statistics only."""
    video_counts = {}
    tab_values = {}
    durations = []
    for item in train_rows:
        base = item["base"]
        video_counts[base[2]] = video_counts.get(base[2], 0) + 1
        tab_values.setdefault(base[4], len(tab_values) + 1)
        durations.append(float(base[5]))
    dmean = float(np.mean(durations)) if durations else 0.0
    dstd = float(np.std(durations)) if durations else 1.0
    dstd = dstd if dstd > 1e-12 else 1.0

    def raw(item):
        base = item["base"]
        duration = (np.log1p(float(base[5])) - np.log1p(dmean)) / max(dstd, 1.0)
        tab = float(tab_values.get(base[4], 0))
        hour = float(item.get("hourmin", 0)) / 1439.0
        history = np.log1p(float(item.get("hist", (0, 0, 0))[0]))
        exposure = np.log1p(float(video_counts.get(base[2], 0)))
        return {"duration_ms": duration, "tab": tab, "hourmin": hour,
                "user_history_count": history, "video_exposure_count": exposure}

    names = []
    # A list is the explicit Task 2 contract.  The legacy feature_set bundles
    # remain readable for old artifacts, but new candidates use this path.
    if isinstance(feature_set, list):
        names = list(feature_set)
        if any(name not in PURE_FEATURES for name in names):
            raise ValueError("selected task features are not allowlisted")
        feature_set = "__selected__"
    if feature_set == "none" or feature_set == "__selected__":
        names = names
    elif feature_set == "pure_v1":
        names = list(PURE_FEATURES)
    elif feature_set == "safe_context_v1":
        matrix, mean, std = build_context_features(train_rows, rows, predictions)
        names = list(FEATURE_NAMES)
        if interaction != "none":
            raise ValueError("safe_context_v1 does not accept prediction interactions")
        return matrix, names, mean, std
    elif feature_set in PURE_FEATURES:
        names = [feature_set]
    else:
        raise ValueError(f"unsupported pure feature set: {feature_set}")

    values = [raw(item) for item in rows]
    matrix = np.asarray([[v[name] for name in names] for v in values], dtype=np.float64)
    if interaction != "none":
        if interaction not in INTERACTIONS:
            raise ValueError(f"unsupported interaction: {interaction}")
        if predictions is None:
            raise ValueError("prediction interaction requires all family predictions")
        family, feature = {
            "p_bpr_duration": ("bpr_fm", "duration_ms"),
            "p_listwise_duration": ("listwise_fm", "duration_ms"),
            "p_history_user_history": ("history_fm", "user_history_count"),
            "p_cwm_duration": ("cwm_fm", "duration_ms"),
            "p_multitask_tab": ("multitask_fm", "tab"),
        }[interaction]
        index = FAMILY_ORDER.index(family)
        extra = np.asarray([v[feature] for v in values], dtype=np.float64) * predictions[:, index]
        matrix = np.column_stack([matrix, extra])
        names.append(interaction)
    if matrix.ndim == 1:
        matrix = matrix.reshape(len(rows), -1)
    mean = matrix.mean(axis=0) if len(matrix) and matrix.shape[1] else np.zeros(matrix.shape[1])
    std = matrix.std(axis=0) if len(matrix) and matrix.shape[1] else np.ones(matrix.shape[1])
    std = np.where(std > 1e-12, std, 1.0)
    return ((matrix - mean) / std if matrix.shape[1] else matrix), names, mean, std


def build_task_features(train_rows, rows, predictions, config, mean=None, std=None):
    matrix, names, fitted_mean, fitted_std = _task_feature_values(
        train_rows, rows, predictions, config.get("feature_set", "none"),
        config.get("interaction", "none"))
    if mean is not None or std is not None:
        mean = np.asarray(mean, dtype=np.float64)
        std = np.where(np.asarray(std, dtype=np.float64) > 1e-12,
                       np.asarray(std, dtype=np.float64), 1.0)
        if len(names) != len(mean):
            raise ValueError("composition feature schema changed between fit and predict")
        matrix = (matrix * fitted_std + fitted_mean - mean) / std
        fitted_mean, fitted_std = mean, std
    return matrix, names, fitted_mean, fitted_std


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 30.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _prediction_analysis(data):
    predictions = np.asarray(data["train_predictions"], dtype=np.float64)
    correlation = np.corrcoef(predictions.T) if len(predictions) > 1 else np.eye(5)
    return {
        "families": list(FAMILY_ORDER),
        "row_alignment": bool(len(data["train_predictions"]) == len(data["train_users"])
                                and len(data["valid_predictions"]) == len(data["valid_users"])),
        "train_rows": int(len(data["train_predictions"])),
        "valid_rows": int(len(data["valid_predictions"])),
        "scale": {family: {"mean": float(predictions[:, i].mean()),
                            "std": float(predictions[:, i].std()),
                            "min": float(predictions[:, i].min()),
                            "max": float(predictions[:, i].max())}
                  for i, family in enumerate(FAMILY_ORDER)},
        "user_level_variance": {family: float(_user_zscore(predictions[:, i], data["train_users"]).var())
                                 for i, family in enumerate(FAMILY_ORDER)},
        "missing_nan_inf": int(np.size(predictions) - np.isfinite(predictions).sum()),
        "prediction_correlation": correlation.tolist(),
    }


def _composition_forward(predictions, features, state, model):
    base = state["base_logits"]
    if model == "nonnegative_linear":
        weights = _softmax(base)
        return predictions @ weights + float(state["bias"]) + features @ state["feature_weights"], {"weights": weights}
    if model == "gated_linear":
        gate_features = features if features.shape[1] else np.zeros((len(features), 1), dtype=np.float64)
        logits = base[None, :] + gate_features @ state["gate_matrix"]
        gate = _softmax(logits)
        return np.sum(gate * predictions, axis=1) + float(state["bias"]), {"gate": gate}
    if model == "small_mlp":
        mlp_features = np.column_stack((predictions, features))
        hidden1 = np.maximum(mlp_features @ state["mlp_w1"] + state["mlp_b1"], 0.0)
        hidden2 = np.maximum(hidden1 @ state["mlp_w2"] + state["mlp_b2"], 0.0)
        scores = (hidden2 @ state["mlp_w3"]).reshape(-1) + float(state["mlp_b3"][0])
        return scores, {"hidden1": hidden1, "hidden2": hidden2}
    raise ValueError(f"unsupported composition model: {model}")


def _loss_gradient(scores, users, labels, pairs):
    """0.6 within-user listwise CE + 0.4 same-user BPR gradient."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=object)
    labels = np.asarray(labels, dtype=np.float64)
    grad = np.zeros(len(scores), dtype=np.float64)
    list_loss = 0.0; groups = 0
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1])]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        ix = order[start:end]
        pos = labels[ix] > 0
        if not np.any(pos):
            continue
        local = scores[ix]
        prob = _softmax(local[None, :])[0]
        target = pos.astype(np.float64) / float(pos.sum())
        grad[ix] += 0.6 * (prob - target)
        list_loss += float(-np.sum(target * np.log(np.maximum(prob, 1e-12))))
        groups += 1
    bpr_loss = 0.0
    if len(pairs):
        p, n = pairs[:, 0], pairs[:, 1]
        delta = np.clip(scores[n] - scores[p], -30.0, 30.0)
        q = 1.0 / (1.0 + np.exp(-delta))
        grad[p] -= 0.4 * q / len(pairs)
        grad[n] += 0.4 * q / len(pairs)
        bpr_loss = float(np.mean(np.logaddexp(0.0, delta)))
    return grad, 0.6 * list_loss / max(groups, 1) + 0.4 * bpr_loss


def fit_composition_layer(root, config, rich_rows=None):
    """Train only the final composition parameters on train rows."""
    if config.get("composition_code") != "11111":
        raise ValueError("composition layer requires all five families")
    if config.get("composition_loss", "0.6_listwise_0.4_bpr") != "0.6_listwise_0.4_bpr":
        raise ValueError("composition target/loss is fixed to long_view 0.6 listwise + 0.4 BPR")
    data = load_all_family_predictions(root, rich_rows=rich_rows)
    input_weights = np.asarray(config.get("prediction_input_weights", [1.0] * 5), dtype=np.float64)
    if (input_weights.shape != (5,) or not np.isfinite(input_weights).all()
            or np.any(input_weights <= 0)):
        raise ValueError("prediction_input_weights must contain five positive finite values")
    input_weights = input_weights / input_weights.sum()
    # Downstream tasks receive five weighted prediction channels. Task 1
    # learns its own weights and therefore consumes unweighted channels.
    if config.get("task_id") != "weight_learning":
        data["train_predictions"] = data["train_predictions"] * input_weights[None, :]
        data["valid_predictions"] = data["valid_predictions"] * input_weights[None, :]
    prediction_analysis = _prediction_analysis(data)
    train_features, names, feature_mean, feature_std = build_task_features(
        data["train_rows"], data["train_rows"], data["train_predictions"], config)
    valid_features, _, _, _ = build_task_features(
        data["train_rows"], data["valid_rows"], data["valid_predictions"], config,
        feature_mean, feature_std)
    model = config.get("composition_model", "nonnegative_linear")
    if model == "fixed_weight":
        model = "nonnegative_linear"
    seed = int(config.get("composition_seed", 0)); rng = np.random.default_rng(seed)
    initial_weights = [.2] * 5 if config.get("task_id") == "weight_learning" else config.get("weights", [.2] * 5)
    weights = np.asarray(initial_weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8); weights /= weights.sum()
    if config.get("task_id") == "weight_learning":
        initial_raw = config.get("initial_raw_weights")
        if initial_raw is None:
            base_logits = np.zeros(5, dtype=np.float64)
        else:
            if (not isinstance(initial_raw, list) or len(initial_raw) != 5
                    or not np.isfinite(np.asarray(initial_raw, dtype=np.float64)).all()):
                raise ValueError("Task 1 initial_raw_weights must be five finite values")
            base_logits = np.asarray(initial_raw, dtype=np.float64).copy()
    else:
        base_logits = (np.zeros(5, dtype=np.float64)
                       if config.get("prediction_input_weights") is not None else np.log(weights))
    state = {"base_logits": base_logits, "bias": float(config.get("bias", 0.0)),
             "feature_weights": np.zeros(train_features.shape[1], dtype=np.float64),
             "feature_schema": names, "feature_mean": feature_mean,
             "feature_std": feature_std, "model": model, "loss": "0.6_listwise_0.4_bpr",
             "target": "long_view"}
    if model == "gated_linear":
        state["gate_matrix"] = np.zeros((max(train_features.shape[1], 1), 5), dtype=np.float64)
    elif model == "small_mlp":
        input_dim = 5 + train_features.shape[1]
        state["mlp_input_dim"] = input_dim
        state["mlp_hidden1"] = 64
        state["mlp_hidden2"] = 32
        state["mlp_w1"] = rng.normal(0.0, 0.02, (input_dim, 64))
        state["mlp_b1"] = np.zeros(64, dtype=np.float64)
        state["mlp_w2"] = rng.normal(0.0, 0.02, (64, 32))
        state["mlp_b2"] = np.zeros(32, dtype=np.float64)
        state["mlp_w3"] = np.zeros((32, 1), dtype=np.float64)
        state["mlp_b3"] = np.asarray([float(config.get("bias", 0.0))], dtype=np.float64)
    # Stable user split is derived from IDs, never from validation labels.
    fit_mask = np.asarray([hashlib.sha1(str(u).encode()).digest()[0] % 5 != 0
                           for u in data["train_users"]], dtype=bool)
    monitor_mask = ~fit_mask
    if fit_mask.sum() < 2 or monitor_mask.sum() < 2:
        fit_mask[:] = True; monitor_mask[:] = True
    fit_idx = np.flatnonzero(fit_mask)
    fit_pairs = _pairs(data["train_users"][fit_mask], data["train_labels"][fit_mask], seed)
    fit_pairs = fit_pairs[:200000]
    if len(fit_pairs):
        fit_pairs = np.column_stack((fit_idx[fit_pairs[:, 0]], fit_idx[fit_pairs[:, 1]])).astype(np.int64)
    lr = float(config.get("learning_rate", config.get("lr", 0.02))); l2 = float(config.get("l2", 0.01))
    adam_m = np.zeros(5, dtype=np.float64); adam_v = np.zeros(5, dtype=np.float64)
    adam_mb = adam_vb = 0.0; adam_t = 0
    epochs = int(config.get("epochs", 8)); patience = int(config.get("patience", 2))
    best_state = None; best_loss = float("inf"); bad = 0; epoch_history = []
    monitor_idx = np.flatnonzero(monitor_mask)
    for epoch in range(epochs):
        scores, cache = _composition_forward(data["train_predictions"][fit_idx],
                                              train_features[fit_idx], state, model)
        local_pairs = fit_pairs
        local_pairs = (np.column_stack((np.searchsorted(fit_idx, fit_pairs[:, 0]),
                                        np.searchsorted(fit_idx, fit_pairs[:, 1]))).astype(np.int64)
                       if len(fit_pairs) else fit_pairs)
        grad, train_loss = _loss_gradient(scores, data["train_users"][fit_idx],
                                          data["train_labels"][fit_idx], local_pairs)
        pred = data["train_predictions"][fit_idx]
        if model == "nonnegative_linear":
            w = cache["weights"]
            gw = np.sum(grad[:, None] * pred, axis=0) / len(grad)
            gw_raw = w * (gw - float(np.dot(gw, w)))
            adam_t += 1; beta1, beta2, eps = 0.9, 0.999, 1e-8
            g = gw_raw + l2 * state["base_logits"]
            adam_m = beta1 * adam_m + (1 - beta1) * g
            adam_v = beta2 * adam_v + (1 - beta2) * (g * g)
            mh = adam_m / (1 - beta1 ** adam_t); vh = adam_v / (1 - beta2 ** adam_t)
            state["base_logits"] -= lr * mh / (np.sqrt(vh) + eps)
            gb = float(grad.mean()) + l2 * state["bias"]
            adam_mb = beta1 * adam_mb + (1 - beta1) * gb
            adam_vb = beta2 * adam_vb + (1 - beta2) * (gb * gb)
            state["bias"] -= lr * (adam_mb / (1 - beta1 ** adam_t)) / (np.sqrt(adam_vb / (1 - beta2 ** adam_t)) + eps)
            if train_features.shape[1]:
                state["feature_weights"] -= lr * (train_features[fit_idx].T @ grad / len(grad) + l2 * state["feature_weights"])
        elif model == "gated_linear":
            gate = cache["gate"]; dlogit = grad[:, None] * gate * (pred - np.sum(gate * pred, axis=1)[:, None])
            state["base_logits"] -= lr * (dlogit.mean(axis=0) + l2 * state["base_logits"])
            if train_features.shape[1]:
                gate_features = train_features[fit_idx] if train_features.shape[1] else np.zeros((len(grad), 1), dtype=np.float64)
                state["gate_matrix"] -= lr * (gate_features.T @ dlogit / len(grad) + l2 * state["gate_matrix"])
        else:
            x = np.column_stack((pred, train_features[fit_idx]))
            h1, h2 = cache["hidden1"], cache["hidden2"]
            d3 = grad[:, None]
            dw3 = h2.T @ d3 / len(grad) + l2 * state["mlp_w3"]
            db3 = d3.mean(axis=0)
            dh2 = (d3 @ state["mlp_w3"].T) * (h2 > 0)
            dw2 = h1.T @ dh2 / len(grad) + l2 * state["mlp_w2"]
            db2 = dh2.mean(axis=0)
            dh1 = (dh2 @ state["mlp_w2"].T) * (h1 > 0)
            dw1 = x.T @ dh1 / len(grad) + l2 * state["mlp_w1"]
            db1 = dh1.mean(axis=0)
            state["mlp_w3"] -= lr * dw3; state["mlp_b3"] -= lr * db3
            state["mlp_w2"] -= lr * dw2; state["mlp_b2"] -= lr * db2
            state["mlp_w1"] -= lr * dw1; state["mlp_b1"] -= lr * db1
            state["bias"] = float(state["mlp_b3"][0])
        monitor_scores, _ = _composition_forward(data["train_predictions"][monitor_idx],
                                                  train_features[monitor_idx], state, model)
        monitor_pairs = _pairs(data["train_users"][monitor_idx], data["train_labels"][monitor_idx], seed + epoch)
        _, monitor_loss = _loss_gradient(monitor_scores, data["train_users"][monitor_idx],
                                         data["train_labels"][monitor_idx], monitor_pairs)
        if monitor_loss < best_loss - 1e-6:
            best_loss = monitor_loss; best_state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                                     for k, v in state.items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        epoch_scores, _ = _composition_forward(data["valid_predictions"], valid_features, state, model)
        epoch_metrics = evaluate(data["valid_users"], data["valid_labels"], epoch_scores)
        epoch_history.append({"epoch": epoch + 1, "train_loss": float(train_loss),
                              "composition_monitor_loss": float(monitor_loss),
                              "validation_GAUC": float(epoch_metrics["GAUC"]),
                              "validation_nDCG@5": float(epoch_metrics["nDCG@5"]),
                              "validation_primary": float(epoch_metrics["primary"]),
                              "raw_weights": state["base_logits"].tolist(),
                              "normalized_weights": _softmax(state["base_logits"]).tolist(),
                              "bias": float(state["bias"])})
    if best_state is None:
        best_state = state
    state = best_state
    valid_scores, _ = _composition_forward(data["valid_predictions"], valid_features, state, model)
    state["train_monitor_loss"] = float(best_loss)
    state["raw_weights"] = np.asarray(state["base_logits"], dtype=np.float64).copy()
    state["normalized_weights"] = _softmax(state["base_logits"])
    state["optimizer"] = "adam"
    state["prediction_input_weights"] = input_weights
    state["feature_names"] = tuple(names)
    state["prediction_analysis"] = prediction_analysis
    state["feature_variance"] = {name: float(train_features[:, i].var())
                                  for i, name in enumerate(names)}
    state["training_history"] = epoch_history
    return data, state, valid_scores


def save_composition_layer_checkpoint(root, state, config, metrics):
    """Save only composition parameters; family checkpoint files are untouched."""
    out_dir = outputs_dir(); os.makedirs(out_dir, exist_ok=True)
    name = str(config.get("name", "candidate")).replace("/", "_").replace("\\", "_")
    path = os.path.join(out_dir, f"composition_{name}.npz")
    metadata = {"model": state["model"], "architecture": ("Dense(64)-ReLU-Dense(32)-ReLU-Dense(1)"
                if state["model"] == "small_mlp" else state["model"]),
                "loss": state["loss"], "target": state["target"],
                "feature_names": list(state["feature_names"]), "config": config,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "raw_weights": state.get("raw_weights", []).tolist(),
                "normalized_weights": state.get("normalized_weights", []).tolist(),
                "optimizer": state.get("optimizer", "adam"),
                "prediction_input_weights": state.get("prediction_input_weights", []).tolist(),
                "training_history": state.get("training_history", [])}
    arrays = {"base_logits": state["base_logits"], "bias": np.asarray([state["bias"]]),
              "feature_weights": state["feature_weights"], "feature_mean": state["feature_mean"],
              "feature_std": state["feature_std"],
              "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))}
    for key in ("gate_matrix", "mlp_w1", "mlp_b1", "mlp_w2", "mlp_b2", "mlp_w3", "mlp_b3"):
        if key in state: arrays[key] = state[key]
    np.savez_compressed(path, **arrays)
    return path


def predict_from_composition_checkpoint(root, checkpoint, split="valid"):
    if split != "valid":
        raise PermissionError("composition prediction is validation-only until manual finalization")
    checkpoint = os.path.abspath(os.path.join(root, checkpoint)) if not os.path.isabs(checkpoint) else checkpoint
    if os.path.commonpath([root, checkpoint]) != root or not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        state = {"base_logits": saved["base_logits"], "bias": float(saved["bias"][0]),
                 "feature_weights": saved["feature_weights"], "feature_mean": saved["feature_mean"],
                 "feature_std": saved["feature_std"], "model": metadata["model"],
                 "loss": metadata["loss"], "target": metadata["target"]}
        for key in ("gate_matrix", "mlp_w1", "mlp_b1", "mlp_w2"):
            if key in saved: state[key] = saved[key]
        config = metadata["config"]
    data = load_all_family_predictions(root)
    features, _, _, _ = build_task_features(data["train_rows"], data["valid_rows"],
                                            data["valid_predictions"], config,
                                            state["feature_mean"], state["feature_std"])
    return _composition_forward(data["valid_predictions"], features, state, state["model"])[0]
