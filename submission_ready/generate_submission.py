"""Generate a submission-format CSV from the frozen best ensemble.

Default operation is validation-only.  Public-test generation is an explicit
manual finalization action and is intentionally not used by the agent loop.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from baseline import FM
from data import encode, load
from dataset_config import data_dir
from prediction_adapter import predict_from_manifest
from validation_only import load_train_valid

HEADER = ["row_id", "user_id", "video_id", "score"]
MANIFEST = os.path.join(ROOT, "submission_ready", "composition_manifest.json")


def rows_and_features(split, allow_final_test=False):
    if split == "valid":
        splits = load_train_valid(data_dir())
        encoded, dimension = encode({**splits, "test": []})
        return splits["valid"], encoded["valid"][0], dimension
    if not allow_final_test:
        raise PermissionError("test generation requires explicit --allow-final-test")
    # This path is for a human-approved finalization only; it is never called
    # by autonomous_agent.py and cannot affect experiment selection.
    splits = load(data_dir())
    encoded, dimension = encode(splits)
    return splits["test"], encoded["test"][0], dimension


def generate(output, split="valid", allow_final_test=False):
    rows, features, dimension = rows_and_features(split, allow_final_test)
    scores = predict_from_manifest(MANIFEST, FM, dimension, features, split=split)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for row_id, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow([row_id, row[1], row[2], f"{float(score):.12g}"])
    with open(MANIFEST, encoding="utf-8") as fh:
        model = json.load(fh).get("model")
    return {"path": os.path.abspath(output), "split": split, "rows": len(rows),
            "model": model, "test_access": split == "test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    if args.manifest != MANIFEST:
        globals()["MANIFEST"] = os.path.abspath(args.manifest)
    print(generate(args.output, args.split, args.allow_final_test))
