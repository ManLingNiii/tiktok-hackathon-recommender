"""安全的 validation-only runner for KuaiRand-Pure.

只保留 train/valid rows；test split 不建立、不評估、不輸出。
官方 evaluate.py 與 baseline.py 均不修改。
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
try:
    from dataset_config import data_dir, dataset_name, runs_dir, outputs_dir
except ImportError:
    from agent.dataset_config import data_dir, dataset_name, runs_dir, outputs_dir
DATA = data_dir()
RUNS = runs_dir()
OUT = outputs_dir()
sys.path.insert(0, KIT)
from baseline import FM
from data import encode
from evaluate import evaluate
try:
    from checkpoint_manager import save_if_best
    from validation_guard import evaluate_confirmation
except ImportError:
    from agent.checkpoint_manager import save_if_best
    from agent.validation_guard import evaluate_confirmation


def load_train_valid(data_dir):
    vid2author = {}
    suffix = "_1k" if dataset_name() in {"1k", "kuairand_1k"} else "_pure"
    with open(os.path.join(data_dir, f"video_features_basic{suffix}.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    # Keep only development partitions. No test container is created.
    splits = {"train": [], "valid": []}
    for filename in (f"log_standard_4_08_to_4_21{suffix}.csv", f"log_standard_4_22_to_5_08{suffix}.csv"):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                valid_end = 20220508 if suffix == "_1k" else 20220428
                name = "train" if 20220408 <= date <= 20220421 else "valid" if 20220422 <= date <= valid_end else None
                if name is None:
                    continue
                splits[name].append((date, r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"),
                                     r["tab"], float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0))
    return splits


def run(seed=0, epochs=None, patience=None):
    cfg = json.loads(os.environ.get("AGENT_MODEL_CONFIG", "{}"))
    seed = int(cfg.get("seed", seed))
    if epochs is None:
        epochs = int(os.environ.get("AGENT_BASELINE_EPOCHS", "3" if dataset_name() in {"1k", "kuairand_1k"} else "40"))
    if patience is None:
        patience = int(os.environ.get("AGENT_BASELINE_PATIENCE", "1" if dataset_name() in {"1k", "kuairand_1k"} else "4"))
    splits = load_train_valid(DATA)
    enc, dim = encode({**splits, "test": []})
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    batch_size = int(cfg.get("batch_size", 8192))
    model = FM(dim, k=int(cfg.get("k", 16)), lr=float(cfg.get("lr", 0.001)),
               l2=float(cfg.get("l2", 0.0)), seed=seed)
    rng = np.random.default_rng(seed)
    best = -1.0
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [model.step(Xtr[idx[i:i + batch_size]], ytr[idx[i:i + batch_size]])
                  for i in range(0, len(idx), batch_size)]
        metrics = evaluate(uva, yva, model.predict(Xva))
        history.append(metrics)
        print(f"epoch {epoch:02d} | loss {np.mean(losses):.4f} | validation {metrics}", flush=True)
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state
    final = {key: float(value) for key, value in
             evaluate(uva, yva, model.predict(Xva)).items()}
    confirmation = {key: float(value) for key, value in
                    evaluate_confirmation(uva, yva, model.predict(Xva)).items()}
    os.makedirs(RUNS, exist_ok=True)
    checkpoint = save_if_best(
        "baseline_fm",
        {"V": model.V, "W": model.W, "b": model.b},
        final,
        config={"k": int(cfg.get("k", 16)), "lr": float(cfg.get("lr", 0.001)),
                "l2": float(cfg.get("l2", 0.0)), "batch_size": batch_size, "seed": seed,
                "epochs": len(history), "patience": patience},
        source="agent/validation_only.py",
    )
    record = {"experiment": "baseline_fm", "dataset": dataset_name(), "status": "success", "iteration": 0,
              "split": "validation_only", "seed": seed, "metrics": final,
              "confirmation_metrics": confirmation,
              "epochs": len(history), "test_access": False, "timestamp": time.time(),
              "checkpoint": checkpoint["checkpoint"],
              "checkpoint_saved": checkpoint["checkpoint_saved"]}
    with open(os.path.join(RUNS, "validation_only_iteration_000.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
