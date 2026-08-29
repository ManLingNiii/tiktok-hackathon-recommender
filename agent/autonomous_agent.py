"""Fail-closed autonomous validation loop for the competition-style agent.

The loop may choose only experiments already present in experiment_specs.json.
It never receives arbitrary shell commands and never opens the test split.
"""
import argparse
import json
import os
import subprocess
import sys
import time
try:
    from dataset_config import runs_dir, outputs_dir
    from checkpoint_manager import startup_weights
    from config_generator import resolve_config
except ImportError:
    from agent.dataset_config import runs_dir, outputs_dir
    from agent.checkpoint_manager import startup_weights
    from agent.config_generator import resolve_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER = os.path.join(ROOT, "agent", "validation_experiment_runner.py")
PLANNER = os.path.join(ROOT, "agent", "gemini_planner.py")
SPECS = os.path.join(ROOT, "agent", "experiment_specs.json")
LEADERBOARD = os.path.join(runs_dir(), "validation_leaderboard.json")
STATE = os.path.join(runs_dir(), "autonomous_state.json")


def load_specs():
    with open(SPECS, encoding="utf-8") as fh:
        return json.load(fh)


def propose(tried):
    env = os.environ.copy()
    env.setdefault("PLANNER_BACKEND", "local")
    env["PLANNER_EXCLUDE"] = ",".join(sorted(tried))
    env["LEADERBOARD_PATH"] = LEADERBOARD
    proc = subprocess.run([sys.executable, "-u", PLANNER], cwd=ROOT, env=env,
                          text=True, capture_output=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:] or "planner failed")
    return json.loads(proc.stdout)


def choose_experiment(plan, specs):
    # The planner returns a module; map it to exactly one reviewed spec.
    candidates = [name for name, spec in specs.items()
                  if spec.get("allowed") and spec.get("module") == plan.get("module")]
    if len(candidates) != 1 or set(plan.get("splits", [])) - {"train", "valid"}:
        raise ValueError("planner proposal does not resolve to one validation spec")
    config = plan.get("config")
    if config is not None:
        resolve_config(config, candidates[0])
    return candidates[0]


def run_experiment(name, config=None, planner_decision=None):
    env = os.environ.copy()
    if config:
        selected = resolve_config(config, name)
        env["AGENT_MODEL_CONFIG"] = json.dumps(selected)
    if planner_decision is not None:
        env["AGENT_PLANNER_DECISION"] = json.dumps(planner_decision, ensure_ascii=False)
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, "-u", RUNNER, name], cwd=ROOT,
                              env=env, text=True, capture_output=True,
                              timeout=3660 if env.get("AGENT_DATASET") in {"1k", "kuairand_1k"} else 1260)
    except subprocess.TimeoutExpired as exc:
        # A runner can finish the child trainer and write its log just before
        # the wrapper itself stalls. Recover only a complete child success
        # record; never treat partial epoch output as a result.
        candidates = []
        for filename in os.listdir(runs_dir()):
            if filename.startswith("experiment_") and filename.endswith(f"_{name}.log"):
                path = os.path.join(runs_dir(), filename)
                if os.path.getmtime(path) >= started:
                    candidates.append(path)
        for path in sorted(candidates, key=os.path.getmtime, reverse=True):
            with open(path, encoding="utf-8") as fh:
                recovered = parse_records(fh.read(), name)
            if recovered:
                result = recovered[-1]
                result.setdefault("recovery_events", []).append("recovered_complete_child_after_wrapper_timeout")
                result["recovered_from_timeout"] = True
                return result
        raise RuntimeError("autonomous runner timeout without a complete child result") from exc
    if proc.returncode:
        raise RuntimeError(proc.stdout[-2000:] + proc.stderr[-2000:])
    records = parse_records(proc.stdout, name)
    if not records:
        raise ValueError("runner returned no JSON record")
    return records[-1]


def parse_records(output, name):
    decoder = json.JSONDecoder(); parsed = []
    for index, char in enumerate(output or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
            if isinstance(value, dict):
                parsed.append(value)
        except json.JSONDecodeError:
            continue
    return [value for value in parsed
            if value.get("experiment") == name and value.get("status") == "success"]


def recover_completed_logs(events, tried, specs):
    """Recover complete child results left behind by a wrapper timeout."""
    known = {f"{x.get('experiment')}:{x.get('plan', {}).get('config', 'default')}"
             for x in events if x.get("experiment")}
    module_by_experiment = {name: spec.get("module") for name, spec in specs.items()}
    for filename in sorted(os.listdir(runs_dir())):
        if not (filename.startswith("experiment_") and filename.endswith(".log")):
            continue
        path = os.path.join(runs_dir(), filename)
        for experiment in module_by_experiment:
            if experiment == "baseline_fm" or not filename.endswith(f"_{experiment}.log"):
                continue
            with open(path, encoding="utf-8") as fh:
                records = parse_records(fh.read(), experiment)
            if not records:
                continue
            result = records[-1]
            config = result.get("config", {}).get("name", "default")
            key = f"{experiment}:{config}"
            if key in known:
                continue
            plan = {"module": module_by_experiment[experiment], "experiment": experiment,
                    "splits": ["train", "valid"], "files": ["agent/experiment_specs.json"],
                    "config": config, "strategy": "recovered_complete_child_log"}
            events.append({"iteration": len(events) + 1, "plan": plan,
                           "experiment": experiment, "result": result,
                           "promoted": False,
                           "recovery": "recovered_complete_child_after_wrapper_timeout"})
            tried.add(key); known.add(key)
    return events


def main(max_iterations, sleep_seconds, use_pretrained=True):
    specs = load_specs()
    # Continue from each family's latest best checkpoint by default.
    os.environ["AGENT_USE_PRETRAINED"] = "1" if use_pretrained else "0"
    os.makedirs(runs_dir(), exist_ok=True)
    startup = startup_weights([name for name in specs if name != "baseline_fm"])
    with open(os.path.join(runs_dir(), "agent_startup_weights.json"), "w", encoding="utf-8") as fh:
        json.dump({"use_pretrained": use_pretrained, "families": startup},
                  fh, ensure_ascii=False, indent=2)
    # Bundle-based experiments (such as the imported listwise ensemble) do not
    # have one family-best npz; their reviewed member files are checked by the
    # runner when the experiment is selected.
    missing = [name for name, item in startup.items()
               if not item["exists"] and name != "listwise_ensemble"]
    if missing and use_pretrained:
        raise FileNotFoundError("missing family-best startup weights: " + ", ".join(missing))
    events = []
    if not os.path.exists(LEADERBOARD):
        # Always establish the reference before asking Gemini to optimize.
        run_experiment("baseline_fm")
    tried = set()
    promoted_models = set()
    if os.path.exists(STATE):
        try:
            old = json.load(open(STATE, encoding="utf-8"))
            tried = {f"{x.get('experiment')}:{x.get('plan', {}).get('config', 'default')}"
                     for x in old.get("events", []) if x.get("experiment")}
            promoted_models = {x.get("experiment") for x in old.get("events", [])
                               if x.get("promoted") and x.get("experiment")}
            events = old.get("events", [])
        except (OSError, json.JSONDecodeError):
            events.append({"status": "recovery_required", "error": "invalid autonomous state"})
    iteration = 0
    events = recover_completed_logs(events, tried, specs)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump({"events": events}, fh, ensure_ascii=False, indent=2)
    while max_iterations <= 0 or iteration < max_iterations:
        iteration += 1
        try:
            os.environ["PLANNER_PROMOTED"] = ",".join(sorted(promoted_models))
            plan = propose(tried)
            experiment = choose_experiment(plan, specs)
            key = f"{experiment}:{plan.get('config', 'default')}"
            if key in tried:
                raise ValueError(f"experiment already tried in this autonomous run: {experiment}")
            result = run_experiment(experiment, plan.get("config"), plan)
            tried.add(key)
            leaderboard = json.load(open(LEADERBOARD, encoding="utf-8"))
            baselines = [x for x in leaderboard["experiments"]
                         if x.get("experiment") == "baseline_fm" and x.get("metrics")]
            # Use the latest successful baseline that has the confirmation
            # guard; older leaderboard records predate that safeguard.
            baseline = next((x for x in reversed(baselines)
                             if x.get("confirmation_metrics", {}).get("primary") is not None), None)
            if baseline is None and baselines:
                baseline = baselines[-1]
            epsilon = leaderboard.get("epsilon", 0.002)
            candidate_primary = result.get("metrics", {}).get("primary", -1)
            baseline_primary = baseline["metrics"]["primary"] if baseline else -1
            full_validation_pass = candidate_primary >= baseline_primary + epsilon
            # A promotion must also generalize to the fixed validation
            # confirmation users.  Missing confirmation metrics never pass.
            baseline_confirmation = baseline.get("confirmation_metrics", {}) if baseline else {}
            candidate_confirmation = result.get("confirmation_metrics", {})
            confirmation_pass = bool(
                baseline_confirmation.get("primary") is not None and
                candidate_confirmation.get("primary") is not None and
                candidate_confirmation["primary"] >= baseline_confirmation["primary"] + epsilon
            )
            promoted = full_validation_pass and confirmation_pass
            event = {"iteration": iteration, "plan": plan, "experiment": experiment,
                     "result": result, "promoted": promoted,
                     "promotion_gate": {
                         "epsilon": epsilon,
                         "full_validation_pass": full_validation_pass,
                         "confirmation_pass": confirmation_pass,
                         "baseline_primary": baseline_primary,
                         "candidate_primary": candidate_primary,
                         "baseline_confirmation_primary": baseline_confirmation.get("primary"),
                         "candidate_confirmation_primary": candidate_confirmation.get("primary"),
                     }}
            events.append(event)
            if promoted:
                promoted_models.add(experiment)
            with open(STATE, "w", encoding="utf-8") as fh:
                json.dump({"events": events}, fh, ensure_ascii=False, indent=2)
            print(json.dumps(events[-1], ensure_ascii=False))
            # Stop at the first candidate that passes both validation gates.
            # The promotion remains validation-only and is auditable.
            if promoted:
                break
        except Exception as exc:
            if "exhausted current configs" in str(exc):
                events.append({"iteration": iteration, "status": "search_exhausted",
                               "reason": str(exc), "promoted_models": sorted(promoted_models),
                               "note": "registered search space exhausted; stop to avoid validation overfitting"})
                with open(STATE, "w", encoding="utf-8") as fh:
                    json.dump({"events": events}, fh, ensure_ascii=False, indent=2)
                print(json.dumps(events[-1], ensure_ascii=False))
                break
            else:
                events.append({"iteration": iteration, "status": "recovery_required",
                           "error": str(exc)})
            with open(STATE, "w", encoding="utf-8") as fh:
                json.dump({"events": events}, fh, ensure_ascii=False, indent=2)
            print(json.dumps(events[-1], ensure_ascii=False))
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="0 means run until all registered experiment/config candidates are exhausted")
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument("--use-pretrained", action="store_true", default=True,
                        help="use each family-best checkpoint as the first weights (default)")
    parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained",
                        help="start from random weights for an isolated comparison")
    args = parser.parse_args()
    raise SystemExit(main(args.max_iterations, args.sleep_seconds, args.use_pretrained))
