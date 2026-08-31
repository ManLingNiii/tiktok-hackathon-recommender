"""Fixed, validation-only experiment runner.

The runner is deliberately deterministic and fail-closed. New experiments
must be added to experiment_specs.json and must execute validation_only.py or
another reviewed module; test is never a valid development split.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
try:
    from dataset_config import dataset_name, runs_dir
except ImportError:
    from agent.dataset_config import dataset_name, runs_dir
try:
    from headroom_interface import validate_result
except ImportError:
    from agent.headroom_interface import validate_result
try:
    from config_generator import validate_composition_config
except ImportError:
    from agent.config_generator import validate_composition_config


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
RUNS = runs_dir()
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
        if len(command) != 3 or command[1] != "--mode" or command[2] not in {"listwise","history","multitask","cwm","composition"}:
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


def parse_child_record(output, name):
    decoder = json.JSONDecoder(); records = []
    for i, char in enumerate(output):
        if char != "{": continue
        try:
            value, _ = decoder.raw_decode(output[i:])
            if isinstance(value, dict) and value.get("experiment") == name and value.get("status"):
                records.append(value)
        except json.JSONDecodeError:
            continue
    return records[-1] if records else {}


def code_audit(run_id):
    """Record the project code snapshot used for this validation run."""
    status = subprocess.run(["git", "status", "--short", "--", "agent"],
                            cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                            capture_output=True)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "HEAD", "--", "agent"],
                          cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                          capture_output=True)
    status_text = status.stdout or ""
    diff_text = diff.stdout or ""
    changed_files = [line[3:] if len(line) >= 3 else line
                     for line in status_text.splitlines() if line.strip()]
    patch_path = os.path.join(RUNS, f"{run_id}_code_diff.patch")
    with open(patch_path, "w", encoding="utf-8") as fh:
        fh.write(diff_text)
    return {"changed_files": changed_files, "patch_path": patch_path,
            "sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
            "git_status_exit_code": status.returncode}


def run(name="baseline_fm"):
    specs = load_specs()
    if name not in specs:
        raise ValueError(f"unknown experiment: {name}")
    spec = specs[name]
    validate_spec(name, spec)
    if name == "composition_fm":
        raw_config = os.environ.get("AGENT_MODEL_CONFIG", "{}")
        try:
            validate_composition_config(json.loads(raw_config))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid composition recipe: {exc}") from exc
    os.makedirs(RUNS, exist_ok=True)
    start = time.time()
    run_id = (f"{dataset_name()}-"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
              f"{name}-{uuid.uuid4().hex[:8]}")
    audit = code_audit(run_id)
    planner_raw = os.environ.get("AGENT_PLANNER_DECISION")
    try:
        planner_decision = json.loads(planner_raw) if planner_raw else None
    except json.JSONDecodeError:
        planner_decision = {"raw": planner_raw, "parse_error": True}
    recovery_events = []
    try:
        proc = subprocess.run(
            [sys.executable, "-u", *spec["command"]],
            cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True,
            timeout=3600 if dataset_name() in {"1k", "kuairand_1k"} else 1200,
            env={**os.environ, "AGENT_MODEL_CONFIG": os.environ.get("AGENT_MODEL_CONFIG", "{}")},
        )
    except subprocess.TimeoutExpired as exc:
        recovery_events.append("timeout_after_1200_seconds")
        proc = subprocess.CompletedProcess([], 124, exc.stdout or "", exc.stderr or "")
    output = proc.stdout + proc.stderr
    iteration = len([x for x in os.listdir(RUNS) if x.startswith("experiment_") and x.endswith(".json")])
    log_path = os.path.join(RUNS, f"experiment_{iteration:03d}_{name}.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(output)
    record = {
        "run_id": run_id,
        "iteration": iteration,
        "experiment": name,
        "dataset": dataset_name(),
        "hypothesis": (planner_decision or {}).get("hypothesis") or spec["description"],
        "split": "validation_only",
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "metrics": None,
        "changed_files": audit["changed_files"],
        "code_diff": audit,
        "model_configuration": json.loads(os.environ.get("AGENT_MODEL_CONFIG", "{}")),
        "manual_interventions": 0,
        "recovery_events": recovery_events,
        "recovery_event": recovery_events,
        "error": None,
        "planner_decision": planner_decision,
        "test_access": False,
        "duration_seconds": round(time.time() - start, 2),
        "runtime": round(time.time() - start, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if proc.returncode == 0:
        try:
            child = parse_child_record(output, name)
            record["metrics"] = child.get("metrics") or parse_metrics(output)
            for key in ("config", "checkpoint", "checkpoint_saved", "family_best_metrics",
                        "confirmation_metrics", "members", "checkpoint_members", "loss",
                        "target", "composition_model", "composition_loss", "feature_schema",
                        "schema_analysis", "composition_manifest", "prediction_analysis",
                        "feature_variance", "training_history", "raw_weights", "normalized_weights",
                        "prediction_input_weights", "optimizer"):
                if key in child:
                    record[key] = child[key]
            record["model_configuration"] = child.get("config", record["model_configuration"])
            if child.get("checkpoint"):
                record["checkpoint_path"] = child["checkpoint"]
            if child:
                validate_result({**record, "status": "success",
                                 "checkpoint": child.get("checkpoint", record.get("checkpoint")),
                                 "checkpoint_saved": child.get("checkpoint_saved", False),
                                 "family_best_metrics": child.get("family_best_metrics", record["metrics"])})
        except ValueError as exc:
            record["status"] = "failed"
            record["returncode"] = 2
            record["recovery_events"].append(str(exc))
            record["recovery_event"] = record["recovery_events"]
            record["error"] = str(exc)
    elif proc.returncode != 0:
        record["error"] = (proc.stderr or proc.stdout or "runner failed")[-4000:]
        record["recovery_event"] = record["recovery_events"]
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
