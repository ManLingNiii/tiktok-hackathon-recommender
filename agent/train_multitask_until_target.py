"""Run only the reviewed Multi-task family until the validation target is met."""
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER = os.path.join(ROOT, "agent", "validation_experiment_runner.py")
SPACE = os.path.join(ROOT, "agent", "configs", "search_space.json")
RUNS = os.path.join(ROOT, "runs", "pure")
TARGET = 0.60347
LOG = os.path.join(RUNS, "multitask_training_log.txt")


def append(text):
    os.makedirs(RUNS, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def run(config):
    env = os.environ.copy()
    env["AGENT_DATASET"] = "pure"
    env["AGENT_MODEL_CONFIG"] = json.dumps(config)
    # The exact BPR primary candidate must start from the same fresh seed as
    # the reviewed standalone BPR experiment; other candidates may warm-start.
    env["AGENT_USE_PRETRAINED"] = "0" if config.get("primary_objective") == "bpr" else "1"
    proc = subprocess.run([sys.executable, "-u", RUNNER, "multitask_fm"],
                          cwd=ROOT, env=env, text=True, capture_output=True,
                          timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    decoder = json.JSONDecoder(); found=[]
    for i, c in enumerate(proc.stdout):
        if c != "{": continue
        try:
            value, _ = decoder.raw_decode(proc.stdout[i:])
            if value.get("experiment") == "multitask_fm" and value.get("status") == "success": found.append(value)
        except (json.JSONDecodeError, AttributeError): pass
    if not found: raise RuntimeError("runner returned no multitask result")
    return found[-1]


def main():
    with open(SPACE, encoding="utf-8") as fh: candidates = json.load(fh)["candidates"]
    candidates = [x for x in candidates if "multitask_fm" in x.get("families", [])
                  and x.get("name") in {"multitask_primary_bpr_isolated"}]
    append("\n=== Multi-task Pure run started ===\nDataset: KuaiRand-Pure (configured project data directory)\nTarget Primary: 0.60347\nInput/output contract: validation_experiment_runner, validation_only split, JSON record + checkpoint\nAnalysis: prior Multi-task peaked at 0.60242373 while standalone BPR reached 0.60367590; the remaining gap was caused by a different primary optimizer, not lack of data.\nChanges: exact reviewed BPR primary optimizer with isolated auxiliary head and fresh initialization; validation early stopping and confirmation gate remain enabled.\nReason: remove pointwise/click gradient coupling while retaining the same primary ranking objective and preserving an auxiliary output path.")
    best = -1.0
    for config in candidates:
        append(f"\nCONFIG {config['name']}\nChange: k={config.get('k')}, lr={config.get('lr')}, l2={config.get('l2')}, epochs={config.get('epochs')}, patience={config.get('patience')}\nReason: controlled hyperparameter/architecture candidate from search_space.json.")
        try:
            result = run(config)
            primary = float(result["metrics"]["primary"]); best = max(best, primary)
            append("Result: " + json.dumps(result, ensure_ascii=False))
            append(f"Decision: {'PROMOTED target reached' if primary >= TARGET else 'continue search'}")
            print(json.dumps(result, ensure_ascii=False))
            if primary >= TARGET: return 0
        except Exception as exc:
            append(f"Result: ERROR {exc}\nDecision: continue with next registered config")
            print(json.dumps({"config": config["name"], "status": "recovery_required", "error": str(exc)}))
    append(f"Target not reached in current registered candidates; best={best}. No test data used.")
    return 2


if __name__ == "__main__": raise SystemExit(main())
