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
DATA = os.path.join(KIT, "KuaiRand-Pure", "data")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, KIT)
from baseline import FM
from data import encode
from evaluate import evaluate


def load_train_valid(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    # Keep only development partitions. No test container is created.
    splits = {"train": [], "valid": []}
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                name = "train" if 20220408 <= date <= 20220421 else "valid" if 20220422 <= date <= 20220428 else None
                if name is None:
                    continue
                splits[name].append((date, r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"),
                                     r["tab"], float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0))
    return splits


def run(seed=0, epochs=40, patience=4):
    splits = load_train_valid(DATA)
    enc, dim = encode({**splits, "test": []})
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    model = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best = -1.0
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [model.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]]) for i in range(0, len(idx), 8192)]
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
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "validation_best_fm.npz"), V=model.V, W=model.W, b=model.b)
    record = {"iteration": 0, "split": "validation_only", "seed": seed, "metrics": final,
              "epochs": len(history), "test_access": False, "timestamp": time.time()}
    with open(os.path.join(RUNS, "validation_only_iteration_000.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
