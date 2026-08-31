"""Evaluate the frozen seed-0 family checkpoints on validation only."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)
from data import encode
from evaluate import evaluate
from validation_guard import evaluate_confirmation
from validation_only import load_train_valid
from rich_data import encode_rich, load_rich
from modules.composition import _predict, _predict_history_external_train_valid


def main():
    data = os.path.join(KIT, "KuaiRand-Pure", "data")
    registry_path = os.path.join(ROOT, "submission_ready", "checkpoint_registry.json")
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)
    splits = load_train_valid(data)
    train_rows, valid_rows = load_rich(data)
    basic, dim = encode({**splits, "test": []})
    users = np.asarray([row[1] for row in splits["valid"]], dtype=object)
    labels = basic["valid"][1]
    results = {}
    for family, entry in registry["families"].items():
        path = os.path.abspath(os.path.join(ROOT, entry["checkpoint"]))
        if family == "bpr_fm":
            scores = _predict(ROOT, family, basic["valid"][0], dim)
        elif family == "history_fm":
            train_base = [row["base"] + (row["y"],) for row in train_rows]
            valid_base = [row["base"] + (row["y"],) for row in valid_rows]
            _, scores = _predict_history_external_train_valid(
                ROOT, train_base, valid_base)
        else:
            _, valid, family_dim = encode_rich(
                train_rows, valid_rows, include_history=False)
            scores = _predict(ROOT, family, valid[0], family_dim)
        validation = evaluate(users, labels, scores)
        confirmation = evaluate_confirmation(users, labels, scores)
        results[family] = {
            "checkpoint": path,
            "metrics": {k: float(v) for k, v in validation.items()},
            "confirmation_metrics": {k: float(v) for k, v in confirmation.items()},
            "validation_only": True,
            "test_access": False,
        }
    print(json.dumps({"dataset": "pure", "seed": 0, "families": results}, indent=2))


if __name__ == "__main__":
    main()
