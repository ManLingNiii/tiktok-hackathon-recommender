"""Deprecated entrypoint.

The loop deliberately treats the Starter Kit as the source of truth and never
opens hidden-test labels. Later iterations can add a proposal/reflection step.
Do not use this file for development: it runs the Starter Kit baseline, which
evaluates the public test split. Use validation_experiment_runner.py instead.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)


def run_official_baseline():
    command = [sys.executable, "-u", "baseline.py", "--model", "fm"]
    proc = subprocess.run(command, cwd=KIT, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    with open(os.path.join(RUNS, "agent_iteration_000.log"), "w", encoding="utf-8") as fh:
        fh.write(output)
    record = {
        "iteration": 0,
        "hypothesis": "Re-run the organizer-provided FM baseline as the agent reference.",
        "command": "python -u baseline.py --model fm",
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "manual_interventions": 0,
        "error": None if proc.returncode == 0 else output[-4000:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(RUNS, "agent_iteration_000.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(output, end="")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit("Deprecated: use agent/validation_experiment_runner.py (validation-only).")
