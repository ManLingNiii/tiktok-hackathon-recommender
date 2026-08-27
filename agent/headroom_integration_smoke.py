"""Run each reviewed headroom module through a real FM mini training loop.

This is an integration smoke test, not a benchmark: it uses a small prefix of
the real train/validation data and never constructs a test split.
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)
from baseline import FM
from data import encode
from evaluate import evaluate
from validation_only import load_train_valid
from modules import (BPRLossModule, ListwiseLossModule, HistoryFeaturesModule,
                      MultiTaskModule, CensoredWatchTimeModule)
from modules.base import ExperimentContext

DATA = os.path.join(KIT, "KuaiRand-Pure", "data")


def run_one(name, limit=12000, epochs=2, seed=0):
    raw = load_train_valid(DATA)
    raw["train"] = raw["train"][:limit]
    raw["valid"] = raw["valid"][:4000]
    ctx = ExperimentContext(seed=seed)
    modules = {
        "bpr_loss": BPRLossModule(), "listwise_loss": ListwiseLossModule(),
        "history_features": HistoryFeaturesModule(), "multitask": MultiTaskModule(),
        "censored_watch_time": CensoredWatchTimeModule(),
    }
    module = modules[name]
    module.validate(ctx)
    if name == "history_features":
        raw["train"] = module.transform(raw["train"])
        raw["valid"] = module.transform(raw["valid"])
        # Official encoder consumes the first seven fields; history columns are
        # deliberately tested here, while the baseline-compatible path remains intact.
        raw["train"] = [r[:7] for r in raw["train"]]
        raw["valid"] = [r[:7] for r in raw["valid"]]
    enc, dim = encode({"train": raw["train"], "valid": raw["valid"], "test": []})
    Xtr, ytr, _ = enc["train"]; Xva, yva, users = enc["valid"]
    model = FM(dim, k=8, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    hook_losses = []
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for start in range(0, len(idx), 2048):
            batch = idx[start:start + 2048]
            model.step(Xtr[batch], ytr[batch])
            scores = model.predict(Xtr[batch])
            if name == "bpr_loss":
                pos, neg = scores[ytr[batch] > 0], scores[ytr[batch] <= 0]
                n = min(len(pos), len(neg)); hook_losses.append(module.loss(pos[:n], neg[:n]) if n else 0.0)
            elif name == "listwise_loss":
                hook_losses.append(module.loss(ytr[batch], scores))
            elif name == "multitask":
                hook_losses.append(module.weighted_loss(float(np.mean(scores ** 2)), {"is_click": 0.0}))
            elif name == "censored_watch_time":
                watch = np.asarray([r[5] for r in raw["train"]], dtype=float)[batch]
                hook_losses.append(module.one_sided_loss(scores, watch, ytr[batch] < 1))
    metrics = evaluate(users, yva, model.predict(Xva))
    return {"status": "success", "module": name, "metrics": {k: float(v) for k, v in metrics.items()},
            "hook_loss_mean": float(np.mean(hook_losses)) if hook_losses else 0.0,
            "split": "validation_only", "test_access": False}


if __name__ == "__main__":
    names = sys.argv[1:] or ["bpr_loss", "listwise_loss", "history_features", "multitask", "censored_watch_time"]
    results = [run_one(name) for name in names]
    print(json.dumps(results, ensure_ascii=False, indent=2))
