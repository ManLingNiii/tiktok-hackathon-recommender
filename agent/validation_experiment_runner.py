"""Fixed, validation-only experiment runner.

The runner is deliberately deterministic and fail-closed. New experiments
must be added to experiment_specs.json and must execute validation_only.py or
another reviewed module; test is never a valid development split.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
RUNS = os.path.join(ROOT, "runs")
SPECS_PATH = os.path.join(os.path.dirname(__file__), "experiment_specs.json")
PRIMARY_RE = re.compile(r"['\"]primary['\"]:\s*([0-9.eE+-]+)")
GAUC_RE = re.compile(r"['\"]GAUC['\"]:\s*(?:np\.float32\()?([0-9.eE+-]+)")
NDCG_RE = re.compile(r"['\"]nDCG@5['\"]:\s*(?:np\.float32\()?([0-9.eE+-]+)")


def load_specs():
    with open(SPECS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def validate_spec(name, spec):
    if not spec.get("allowed", False):
        raise ValueError(f"experiment is not allowlisted: {name}")
    if "test" in json.dumps(spec).lower():
        raise ValueError("test split/access is forbidden in development")
    command = spec.get("command", [])
    allowed = [["agent/validation_only.py"], ["agent/experiments/bpr_fm.py"]]
    if len(command) >= 2 and command[:1] == ["agent/formal_trainer.py"]:
        if len(command) != 3 or command[1] != "--mode" or command[2] not in {"listwise","history","multitask","cwm"}:
            raise ValueError("invalid formal trainer mode")
    elif command not in allowed:
        raise ValueError("only reviewed validation-only modules may run")


def parse_metrics(output):
    matches = list(PRIMARY_RE.finditer(output))
    if not matches:
        raise ValueError("validation metrics not found")
    tail = output[max(0, matches[-1].start() - 500):matches[-1].end() + 20]
    def value(pattern):
        found = pattern.search(tail)
        return float(found.group(1)) if found else None
    return {"GAUC": value(GAUC_RE), "nDCG@5": value(NDCG_RE),
            "primary": float(matches[-1].group(1))}


def run(name="baseline_fm"):
    specs = load_specs()
    if name not in specs:
        raise ValueError(f"unknown experiment: {name}")
    spec = specs[name]
    validate_spec(name, spec)
    os.makedirs(RUNS, exist_ok=True)
    start = time.time()
    recovery_events = []
    try:
        proc = subprocess.run(
            [sys.executable, "-u", *spec["command"]],
            cwd=ROOT, text=True, capture_output=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        recovery_events.append("timeout_after_600_seconds")
        proc = subprocess.CompletedProcess([], 124, exc.stdout or "", exc.stderr or "")
    output = proc.stdout + proc.stderr
    iteration = len([x for x in os.listdir(RUNS) if x.startswith("experiment_") and x.endswith(".json")])
    log_path = os.path.join(RUNS, f"experiment_{iteration:03d}_{name}.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(output)
    record = {
        "iteration": iteration,
        "experiment": name,
        "hypothesis": spec["description"],
        "split": "validation_only",
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "metrics": None,
        "manual_interventions": 0,
        "recovery_events": recovery_events,
        "test_access": False,
        "duration_seconds": round(time.time() - start, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if proc.returncode == 0:
        try:
            record["metrics"] = parse_metrics(output)
        except ValueError as exc:
            record["status"] = "failed"
            record["returncode"] = 2
            record["recovery_events"].append(str(exc))
    # Keep a small machine-readable leaderboard for Gemini/team review.
    history = []
    for filename in os.listdir(RUNS):
        if filename.startswith("experiment_") and filename.endswith(".json"):
            try:
                with open(os.path.join(RUNS, filename), encoding="utf-8") as fh:
                    old = json.load(fh)
                if old.get("status") == "success" and old.get("metrics"):
                    history.append(old)
            except (OSError, json.JSONDecodeError):
                continue
    history.append(record)
    successful = [x for x in history if x.get("status") == "success" and x.get("metrics")]
    if successful:
        best = max(successful, key=lambda x: x["metrics"]["primary"])
        record["best_so_far"] = {"experiment": best["experiment"], "primary": best["metrics"]["primary"]}
        with open(os.path.join(RUNS, "validation_leaderboard.json"), "w", encoding="utf-8") as fh:
            json.dump({"metric": "primary", "epsilon": 0.002, "experiments": successful,
                       "best": best}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(RUNS, f"experiment_{iteration:03d}_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "baseline_fm")
