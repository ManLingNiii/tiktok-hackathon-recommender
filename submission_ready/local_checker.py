"""Local checker for the official submission CSV contract."""
import argparse
import csv
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from data import encode, load
from dataset_config import data_dir
from evaluate import evaluate
from validation_only import load_train_valid

HEADER = ["row_id", "user_id", "video_id", "score"]


def expected_rows(split, allow_final_test=False):
    if split == "valid":
        splits = load_train_valid(data_dir())
        encoded, _ = encode({**splits, "test": []})
        return splits["valid"], encoded["valid"][1], encoded["valid"][2]
    if not allow_final_test:
        raise PermissionError("test checking requires explicit --allow-final-test")
    splits = load(data_dir())
    encoded, _ = encode(splits)
    return splits["test"], encoded["test"][1], encoded["test"][2]


def check(path, split="valid", allow_final_test=False, score=False):
    rows, labels, users = expected_rows(split, allow_final_test)
    values = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        if next(reader, None) != HEADER:
            raise ValueError(f"header must be {','.join(HEADER)}")
        for number, record in enumerate(reader):
            if len(record) != 4:
                raise ValueError(f"row {number + 2} must contain 4 fields")
            row_id, user_id, video_id, value = record
            if int(row_id) != number:
                raise ValueError(f"row {number + 2} has non-contiguous row_id")
            if number >= len(rows) or user_id != rows[number][1] or video_id != rows[number][2]:
                raise ValueError(f"row {number + 2} user/video alignment mismatch")
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"row {number + 2} score must be finite")
            values.append(parsed)
    if len(values) != len(rows):
        raise ValueError(f"row count {len(values)} != expected {len(rows)}")
    result = {"status": "valid", "path": os.path.abspath(path), "split": split,
              "rows": len(values), "test_access": split == "test"}
    if score and split == "valid":
        metrics = evaluate(users, labels, values)
        result["metrics"] = {key: float(value) for key, value in metrics.items()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    print(check(args.path, args.split, args.allow_final_test, args.score))
