"""Fail-closed autonomous validation loop for the competition-style agent.

The loop may choose only experiments already present in experiment_specs.json.
It never receives arbitrary shell commands and never opens the test split.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
# The composition module loads the organizer-provided FM class.  Add the
# read-only starter-kit path before importing agent modules so this entrypoint
# works both as ``python agent/autonomous_agent.py`` and as an imported module.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
if KIT not in sys.path:
    sys.path.insert(0, KIT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from dataset_config import runs_dir, outputs_dir
    from checkpoint_manager import startup_weights
    from config_generator import (resolve_config, validate_composition_config, TASK_BUDGETS,
                                  STRATEGY_CATALOG, composition_recipe_key)
    from research_diagnosis import diagnose, validation_plateau
    from feature_catalog import ensure_feature_catalog
    from modules.composition import save_best_composition_manifest
except ImportError:
    from agent.dataset_config import runs_dir, outputs_dir
    from agent.checkpoint_manager import startup_weights
    from agent.config_generator import (resolve_config, validate_composition_config, TASK_BUDGETS,
                                        STRATEGY_CATALOG, composition_recipe_key)
    from agent.research_diagnosis import diagnose, validation_plateau
    from agent.feature_catalog import ensure_feature_catalog
    from agent.modules.composition import save_best_composition_manifest

RUNNER = os.path.join(ROOT, "agent", "validation_experiment_runner.py")
PLANNER = os.path.join(ROOT, "agent", "gemini_planner.py")
SPECS = os.path.join(ROOT, "agent", "experiment_specs.json")
LEADERBOARD = os.path.join(runs_dir(), "validation_leaderboard.json")
STATE = os.path.join(runs_dir(), "autonomous_state.json")
_configured_task_state = os.environ.get(
    "AGENT_TASK_STATE", os.path.join(runs_dir(), "task_workflow_state.json"))
TASK_STATE = os.path.abspath(_configured_task_state)
if os.path.commonpath([ROOT, TASK_STATE]) != ROOT:
    raise ValueError("AGENT_TASK_STATE must remain inside the project root")
TASK_ORDER = tuple(TASK_BUDGETS)
PROMOTION_EPSILON = 0.002


def hypothesis_status(improvement):
    """Classify evidence for the hypothesis independently of promotion."""
    if improvement >= PROMOTION_EPSILON:
        return "strongly_supported"
    if improvement > 0:
        return "supported"
    if improvement == 0:
        return "unsupported"
    return "rejected"


def load_specs():
    with open(SPECS, encoding="utf-8") as fh:
        return json.load(fh)


def propose(tried, task_id, iteration_in_task, previous_config=None, task_state_strategies=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("PLANNER_BACKEND", "local")
    env["PLANNER_EXCLUDE"] = ",".join(sorted(tried))
    env["LEADERBOARD_PATH"] = LEADERBOARD
    env["PLANNER_TASK_ID"] = task_id
    env["PLANNER_TASK_INDEX"] = str(iteration_in_task)
    env["PLANNER_PREVIOUS_CONFIG"] = json.dumps(previous_config or {}, ensure_ascii=False)
    env["PLANNER_STRATEGY_EXCLUDE"] = json.dumps(
        sorted(strategy_id for strategy_id, info in (task_state_strategies or {}).items()
               if isinstance(info, dict) and info.get("status") == "strategy_converged")
    )
    active = [strategy_id for strategy_id, info in (task_state_strategies or {}).items()
              if isinstance(info, dict) and info.get("status") == "in_progress"]
    env["PLANNER_CURRENT_STRATEGY"] = active[-1] if active else ""
    proc = subprocess.run([sys.executable, "-u", PLANNER], cwd=ROOT, env=env,
                          text=True, encoding="utf-8", errors="replace",
                          capture_output=True, timeout=120)
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
        if isinstance(config, dict):
            if not config.get("name"):
                raise ValueError("invalid planner config")
            if candidates[0] == "composition_fm":
                validate_composition_config(config)
            else:
                resolve_config(config.get("name"), candidates[0])
        else:
            resolve_config(config, candidates[0])
    return candidates[0]


def plan_config_name(plan):
    config = plan.get("config", "default")
    return config.get("name", "default") if isinstance(config, dict) else config


def run_experiment(name, config=None, planner_decision=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if config:
        selected = config if isinstance(config, dict) else resolve_config(config, name)
        env["AGENT_MODEL_CONFIG"] = json.dumps(selected)
    if planner_decision is not None:
        env["AGENT_PLANNER_DECISION"] = json.dumps(planner_decision, ensure_ascii=False)
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, "-u", RUNNER, name], cwd=ROOT,
                              env=env, text=True, encoding="utf-8", errors="replace",
                              capture_output=True,
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
    records = [value for value in parsed
               if value.get("experiment") == name and value.get("status") == "success"]
    # The runner's outer audit record contains the child result as a nested
    # object.  Prefer that outer record so run_id/checkpoint/loss/audit fields
    # are not silently discarded by selecting the nested trainer record.
    audited = [value for value in records if value.get("run_id")]
    return audited or records


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


def main(max_iterations, sleep_seconds, use_pretrained=True, composition_only=False):
    """Run the six reviewed research tasks with a hard 50-iteration budget."""
    specs = load_specs()
    os.environ["AGENT_USE_PRETRAINED"] = "1" if use_pretrained else "0"
    os.environ["COMPOSITION_ONLY"] = "1" if composition_only else "0"
    os.makedirs(runs_dir(), exist_ok=True)
    feature_catalog = ensure_feature_catalog(ROOT)
    startup = startup_weights([name for name in specs if name != "baseline_fm"])
    with open(os.path.join(runs_dir(), "agent_startup_weights.json"), "w", encoding="utf-8") as fh:
        json.dump({"use_pretrained": use_pretrained, "families": startup,
                   "feature_catalog": feature_catalog,
                   "frozen_for_composition": list(TASK_ORDER_FAMILIES())}, fh,
                  ensure_ascii=False, indent=2)
    missing = [name for name in TASK_ORDER_FAMILIES() if not startup.get(name, {}).get("exists")]
    if missing and use_pretrained:
        raise FileNotFoundError("missing frozen family checkpoints: " + ", ".join(missing))
    # Establish a baseline only through the fixed runner.  It is not counted
    # as one of the six research tasks.
    needs_baseline = True
    if os.path.exists(LEADERBOARD):
        try:
            board = json.load(open(LEADERBOARD, encoding="utf-8"))
            needs_baseline = not any(x.get("experiment") == "baseline_fm"
                                     and x.get("metrics", {}).get("primary") is not None
                                     and x.get("confirmation_metrics", {}).get("primary") is not None
                                     for x in board.get("experiments", []))
        except (OSError, json.JSONDecodeError):
            needs_baseline = True
    if needs_baseline:
        run_experiment("baseline_fm")

    # A new versioned state file prevents old bitmask runs from silently
    # changing the semantics of this workflow.
    state = {"workflow_version": 2, "max_iterations": 50, "events": [],
             "tasks": {task: {"task_id": task, "status": "pending", "iteration_count": 0,
                              "max_iterations": budget, "best_primary": None,
                              "previous_best_primary": None, "stagnant_iterations": 0,
                              "next_task": TASK_ORDER[i + 1] if i + 1 < len(TASK_ORDER) else None}
                      for i, (task, budget) in enumerate(TASK_BUDGETS.items())}}
    if os.path.exists(TASK_STATE):
        try:
            old = json.load(open(TASK_STATE, encoding="utf-8"))
            if old.get("workflow_version") == 2:
                state = old
        except (OSError, json.JSONDecodeError):
            state["recovery_event"] = "invalid task workflow state; started a clean version-2 state"
    baseline_primary = None
    try:
        board = json.load(open(LEADERBOARD, encoding="utf-8"))
        baseline_rows = [x for x in board.get("experiments", [])
                         if x.get("experiment") == "baseline_fm" and x.get("metrics", {}).get("primary") is not None]
        if baseline_rows:
            baseline_primary = float(baseline_rows[-1]["metrics"]["primary"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        baseline_primary = None
    for task in TASK_ORDER:
        state["tasks"][task].setdefault("baseline_primary", baseline_primary)
    events = state.get("events", [])
    tried = {x.get("recipe_key") for x in events if x.get("recipe_key")}
    iteration = len([x for x in events if x.get("iteration")])
    max_run = min(50, max_iterations) if max_iterations > 0 else 50
    while iteration < max_run:
        task_id = next((task for task in TASK_ORDER
                        if state["tasks"][task]["status"] in {"pending", "in_progress"}), None)
        if task_id is None:
            break
        task_state = state["tasks"][task_id]
        if task_state["iteration_count"] >= task_state["max_iterations"]:
            task_state["status"] = "converged_by_budget"
            continue
        task_state["status"] = "in_progress"
        task_index = task_state["iteration_count"]
        previous_config = task_state.get("best_config") or task_state.get("starting_config")
        if task_id == "weight_learning" and state.get("best_raw_weights"):
            previous_config = dict(previous_config or {})
            previous_config["initial_raw_weights"] = list(state["best_raw_weights"])
        if task_id in {"additive_interaction", "dnn_composition"}:
            # Task 2 and DNN must warm-start from Task 1's learned weights,
            # not from the planner's nominal candidate weights.
            learned = state.get("best_learned_weights")
            if learned:
                previous_config = dict(previous_config or {})
                previous_config["weights"] = list(learned)
        dnn_warm_start_weights = None
        try:
            current_results = [x.get("result", {}) for x in events if x.get("result")]
            plateau = validation_plateau(current_results)
            os.environ["PLANNER_TASK_ID"] = task_id
            os.environ["PLANNER_TASK_INDEX"] = str(task_index)
            os.environ["PLANNER_PREVIOUS_CONFIG"] = json.dumps(previous_config or {}, ensure_ascii=False)
            os.environ["PLANNER_PLATEAU_STREAK"] = str(plateau["streak"])
            os.environ["PLANNER_PLATEAU_THRESHOLD"] = str(plateau["threshold"])
            os.environ["PLANNER_PLATEAU_TRIGGERED"] = "1" if plateau["triggered"] else "0"
            planner_excluded = set(tried)
            planner_excluded.update(
                f"{x.get('task_id')}:{composition_recipe_key(x['config'])}"
                for x in events if isinstance(x.get("config"), dict) and x["config"].get("name"))
            planner_excluded.update(
                f"composition_fm:{x.get('config', {}).get('name')}"
                for x in events if isinstance(x.get("config"), dict) and x["config"].get("name"))
            plan = propose(planner_excluded, task_id, task_index, previous_config,
                           task_state.get("strategies", {}))
            experiment = choose_experiment(plan, specs)
            config = plan.get("config")
            if task_id in {"additive_interaction", "dnn_composition"} and state.get("best_learned_weights"):
                config = dict(config)
                config["prediction_input_weights"] = list(state["best_learned_weights"])
                validate_composition_config(config)
            recipe_key = f"{task_id}:{composition_recipe_key(config)}"
            if recipe_key in tried:
                raise ValueError("same task recipe was proposed twice")
            result = run_experiment(experiment, config, plan)
            candidate_primary = float(result["metrics"]["primary"])
            previous_best = task_state.get("best_primary")
            if previous_best is None:
                previous_best = float(task_state.get("baseline_primary", candidate_primary))
            improvement = candidate_primary - float(previous_best)
            is_new_best = candidate_primary > float(previous_best)
            learned_weights = result.get("normalized_weights")
            if learned_weights and len(learned_weights) == 5:
                learned_weights = [float(x) for x in learned_weights]
            else:
                learned_weights = None
            learned_raw_weights = result.get("raw_weights")
            if learned_raw_weights and len(learned_raw_weights) == 5:
                learned_raw_weights = [float(x) for x in learned_raw_weights]
            else:
                learned_raw_weights = None
            h_status = hypothesis_status(improvement)
            strategy_id = str(plan.get("strategy_id") or plan.get("strategy") or "unspecified")
            strategy_state = task_state.setdefault("strategies", {}).setdefault(
                strategy_id, {"status": "in_progress", "stagnant_iterations": 0,
                              "best_primary": None, "iteration_count": 0})
            strategy_state["iteration_count"] += 1
            if improvement < PROMOTION_EPSILON:
                strategy_state["stagnant_iterations"] += 1
            else:
                strategy_state["stagnant_iterations"] = 0
            strategy_converged = strategy_state["stagnant_iterations"] >= 3
            if strategy_converged:
                strategy_state["status"] = "strategy_converged"
            task_state["previous_best_primary"] = float(previous_best)
            if is_new_best:
                # Retention and promotion are separate decisions.  A recipe
                # that improves validation, even by less than the promotion
                # epsilon, must become the next round's starting point.
                task_state["best_primary"] = candidate_primary
                task_state["best_config"] = config
                task_state["best_checkpoint"] = result.get("checkpoint")
                if learned_weights is not None:
                    task_state["best_learned_weights"] = learned_weights
                    if task_id == "weight_learning":
                        state["best_learned_weights"] = learned_weights
                        if learned_raw_weights is not None:
                            task_state["best_raw_weights"] = learned_raw_weights
                            state["best_raw_weights"] = learned_raw_weights
            if improvement >= PROMOTION_EPSILON:
                task_state["stagnant_iterations"] = 0
                task_status = "improved_continue"
            elif is_new_best:
                # Keep plateau control strict: a sub-epsilon improvement is
                # retained, but still counts toward the three-round plateau
                # window and is not a promotion.
                task_state["stagnant_iterations"] += 1
                task_status = "improved_unpromoted_continue"
            else:
                task_state["stagnant_iterations"] += 1
                task_status = "stagnant_continue"
            task_state["iteration_count"] += 1
            task_state["last_improvement"] = improvement
            known_strategies = set(task_state.get("strategies", {}))
            exhausted_strategies = known_strategies >= set(STRATEGY_CATALOG.get(task_id, ()))
            if (task_state["iteration_count"] >= task_state["max_iterations"]
                    or exhausted_strategies):
                task_state["status"] = "converged" if task_state["stagnant_iterations"] >= 3 else "converged_by_budget"
                task_status = "task_transition"
                next_task = task_state.get("next_task")
                if next_task:
                    state["tasks"][next_task]["status"] = "in_progress"
                    state["tasks"][next_task]["stagnant_iterations"] = 0
                    state["tasks"][next_task]["starting_config"] = task_state.get("best_config")
                    if state.get("best_learned_weights"):
                        state["tasks"][next_task]["starting_config"] = dict(
                            state["tasks"][next_task]["starting_config"] or {})
                        state["tasks"][next_task]["starting_config"]["weights"] = list(
                            state["best_learned_weights"])
            evidence = {"previous_best_primary": float(previous_best),
                        "current_metrics": result.get("metrics"),
                        "confirmation_metrics": result.get("confirmation_metrics"),
                        "loss": result.get("loss"),
                        "prediction_analysis": result.get("prediction_analysis", {}),
                        "feature_variance": result.get("feature_variance", {}),
                        "failure_or_recovery": result.get("recovery_events", [])}
            event = {"run_id": result.get("run_id"), "iteration": iteration + 1, "task_id": task_id,
                     "hypothesis": plan.get("hypothesis"), "evidence": evidence,
                     "change_plan": plan.get("change_plan"), "family_list": list(TASK_ORDER_FAMILIES()),
                     "config": config, "model_configuration": config,
                     "weights": config.get("weights"), "features": [config.get("feature_set", "none")],
                     "normalization": "user_zscore", "composition_model": config.get("composition_model"),
                     "loss": config.get("composition_loss"), "target": "long_view",
                     "result": result, "GAUC": result["metrics"].get("GAUC"),
                     "nDCG@5": result["metrics"].get("nDCG@5"), "primary": candidate_primary,
                     "improvement": improvement,
                     "raw_weights": result.get("raw_weights"),
                     "prediction_input_weights": config.get("prediction_input_weights"),
                     "initial_raw_weights": config.get("initial_raw_weights"),
                     "normalized_weights": learned_weights,
                     "learned_weights_source": "task1_adam_final" if learned_weights else None,
                     "task1_adam_final_weights": state.get("best_learned_weights"),
                     "feature_priority_weights": (state.get("best_learned_weights")
                                                   if task_id == "additive_interaction" else None),
                     "dnn_warm_start_weights": dnn_warm_start_weights,
                     "hypothesis_status": h_status,
                     "strategy_id": strategy_id,
                     "strategy_status": strategy_state["status"],
                     "promotion_status": ("retained_candidate" if is_new_best else "candidate"),
                     "promoted": False,
                     "retained_as_best": is_new_best,
                     "stagnant_iterations": task_state["stagnant_iterations"],
                     "status": task_status, "next_action": task_state.get("next_task"),
                     "test_access": False, "errors": result.get("error"),
                     "recovery_events": result.get("recovery_events", []),
                     "manual_intervention_count": 0, "runtime": result.get("runtime", result.get("duration_seconds")),
                     "planner_decision": plan, "checkpoint_path": result.get("checkpoint"),
                     "recipe_key": recipe_key, "promotion_gate": {"epsilon": PROMOTION_EPSILON,
                         "validation_primary_improved": improvement >= PROMOTION_EPSILON,
                         "retained_as_best": is_new_best,
                         "confirmation_primary": result.get("confirmation_metrics", {}).get("primary")},
                     "research_analysis": diagnose(current_results + [result])}
            events.append(event); tried.add(recipe_key); state["events"] = events
            if task_id == "multi_seed_confirmation":
                vals = [float(x["primary"]) for x in events
                        if x.get("task_id") == task_id and x.get("primary") is not None]
                confirmations = [float(x["result"]["confirmation_metrics"]["primary"])
                                 for x in events if x.get("task_id") == task_id
                                 and x.get("result", {}).get("confirmation_metrics", {}).get("primary") is not None]
                robust_so_far = [x for x in events
                                 if x.get("task_id") == task_id
                                 and x.get("result", {}).get("metrics", {}).get("primary") is not None]
                if vals:
                    state["tasks"][task_id]["robustness_summary"] = {
                        "mean_primary": sum(vals) / len(vals),
                        "std_primary": math.sqrt(sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)),
                        "min_primary": min(vals), "max_primary": max(vals),
                        "confirmation_drift": (sum(vals) / len(vals) - sum(confirmations) / len(confirmations)) if confirmations else None,
                        "stable": set(x["config"].get("composition_seed") for x in robust_so_far) >= {0, 1, 2},
                    }
            with open(TASK_STATE, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
            if task_id == "additive_interaction":
                # Explicit Task 2 artifact: selected features are retained
                # only through the task-best recipe, never by priority alone.
                selected = list((task_state.get("best_config") or {}).get("selected_features", []))
                family_names = ["bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm"]
                weights = (state.get("best_learned_weights")
                           or task_state.get("best_learned_weights")
                           or (task_state.get("best_config") or {}).get("weights", [.2] * 5))
                mapping = {
                    "duration_ms": ("bpr_fm", "cwm_fm"), "tab": ("listwise_fm", "multitask_fm"),
                    "hourmin": ("listwise_fm",), "user_history_count": ("bpr_fm", "history_fm"),
                    "video_exposure_count": ("bpr_fm",),
                }
                priority = {feature: sum(float(weights[family_names.index(family)])
                                         for family in families)
                            for feature, families in mapping.items()}
                with open(os.path.join(runs_dir(), "task2_feature_selection.json"), "w", encoding="utf-8") as fh:
                    json.dump({"task_id": task_id, "selected_features": selected,
                               "feature_priority": priority,
                               "feature_priority_weights": list(weights),
                               "feature_priority_weights_source": (
                                   "task1_adam_final" if state.get("best_learned_weights")
                                   else "task_config_fallback"),
                               "candidate_features": sorted(priority, key=lambda x: (-priority[x], x)),
                               "selection_rule": "validation_primary_improvement >= 0.002",
                               "validation_only": True, "test_access": False}, fh,
                              ensure_ascii=False, indent=2)
            print(json.dumps(event, ensure_ascii=False))
        except Exception as exc:
            event = {"iteration": iteration + 1, "task_id": task_id,
                     "status": "recovery_required", "error": str(exc), "test_access": False,
                     "manual_interventions": 0, "recovery_events": [str(exc)]}
            events.append(event); state["events"] = events
            with open(TASK_STATE, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
            print(json.dumps(event, ensure_ascii=False))
            return 1
        iteration += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
    robust = [x for x in events if x.get("task_id") == "multi_seed_confirmation"
              and x.get("result", {}).get("metrics", {}).get("primary") is not None]
    vals = [float(x["result"]["metrics"]["primary"]) for x in robust]
    confs = [float(x["result"]["confirmation_metrics"]["primary"]) for x in robust
             if x.get("result", {}).get("confirmation_metrics", {}).get("primary") is not None]
    summary = None
    retained = False
    retained_candidate = None
    if vals and confs:
        mean_primary = sum(vals) / len(vals)
        mean_confirmation = sum(confs) / len(confs)
        summary = {"seeds": sorted({x["config"].get("composition_seed") for x in robust}),
                   "mean_primary": mean_primary,
                   "std_primary": math.sqrt(sum((x - mean_primary) ** 2 for x in vals) / len(vals)),
                   "min_primary": min(vals), "max_primary": max(vals),
                   "mean_confirmation_primary": mean_confirmation,
                   "confirmation_drift": abs(mean_primary - mean_confirmation)}
        try:
            board = json.load(open(LEADERBOARD, encoding="utf-8"))
            baseline_rows = [x for x in board.get("experiments", [])
                             if x.get("experiment") == "baseline_fm"]
            baseline = baseline_rows[-1] if baseline_rows else {}
            base_primary = float(baseline.get("metrics", {}).get("primary", 0.0))
            base_confirmation = float(baseline.get("confirmation_metrics", {}).get("primary", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            base_primary = base_confirmation = 0.0
        retained = (set(summary["seeds"]) >= {0, 1, 2}
                    and mean_primary >= base_primary + PROMOTION_EPSILON
                    and mean_confirmation >= base_confirmation + PROMOTION_EPSILON
                    and summary["confirmation_drift"] <= 0.008)
        if retained:
            retained_candidate = max(robust, key=lambda x: x["result"]["metrics"]["primary"])
            manifest_result = save_best_composition_manifest(
                ROOT, retained_candidate["config"], retained_candidate["result"]["metrics"],
                retained_candidate["result"].get("confirmation_metrics"),
                composition_checkpoint=retained_candidate["result"].get("checkpoint_path"),
                schema_analysis={"source": "train_only", "task": "multi_seed_confirmation",
                                 "feature_names": retained_candidate.get("features", [])})
            state["retained_manifest"] = manifest_result
    final = {"status": "promoted" if retained else "not_promoted", "iterations": iteration,
             "tasks": state["tasks"], "retained": retained,
             "robustness_summary": summary,
             "winner": {"run_id": retained_candidate.get("run_id"),
                        "checkpoint": retained_candidate.get("checkpoint_path"),
                        "primary": retained_candidate.get("primary")}
                       if retained_candidate else None,
             "selection": "promote only when all three seeds and validation/confirmation gates pass",
             "best_recipe": retained_candidate.get("config") if retained_candidate else None,
             "best_checkpoint": retained_candidate.get("result", {}).get("checkpoint") if retained_candidate else None,
             "failure_diagnosis": ("promotion gate passed" if retained else "no candidate met validation, confirmation, and multi-seed gates"),
             "unexplored_strategies": [],
             "next_research_plan": "manual submission finalization" if retained else "review remaining allowlisted strategies"}
    events.append({"final_decision": final}); state["events"] = events
    with open(TASK_STATE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    print(json.dumps(events[-1], ensure_ascii=False))
    return 0


def TASK_ORDER_FAMILIES():
    return ("bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="maximum task-based competition-safe iterations (hard cap: 50)")
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument("--use-pretrained", action="store_true", default=True,
                        help="use each family-best checkpoint as the first weights (default)")
    parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained",
                        help="start from random weights for an isolated comparison")
    parser.add_argument("--composition-only", action="store_true",
                        help="evaluate only final composition weights; never train family checkpoints")
    args = parser.parse_args()
    raise SystemExit(main(args.max_iterations, args.sleep_seconds, args.use_pretrained,
                          args.composition_only))
