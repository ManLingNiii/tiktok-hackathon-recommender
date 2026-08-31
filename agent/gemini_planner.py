"""Planner boundary with a local deterministic backend for offline rehearsal.

Set PLANNER_BACKEND=gemini to use Google Gemini after the local loop is proven.
"""
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
if KIT not in sys.path:
    sys.path.insert(0, KIT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment_registry import validate_plan
from config_generator import (config_candidates, config_catalog,
                              adaptive_composition_candidate,
                              next_allowlisted_candidate, next_task_candidate,
                              TASK_BUDGETS, STRATEGY_CATALOG)
from research_diagnosis import diagnose
from modules.context_composition import analyze_train_schema
try:
    from dataset_config import data_dir
    from rich_data import load_rich
except ImportError:
    from agent.dataset_config import data_dir
    from agent.rich_data import load_rich


ALLOWED_EXPERIMENTS = "bpr_fm, listwise_fm, history_fm, multitask_fm, cwm_fm, composition_fm"
LOCAL_PLANS = [
    ("bpr_loss", "bpr_fm"), ("listwise_loss", "listwise_fm"),
    ("history_features", "history_fm"), ("multitask", "multitask_fm"),
    ("censored_watch_time", "cwm_fm"), ("composition", "composition_fm"),
]
SEARCH_SPACE = os.path.join(os.path.dirname(__file__), "configs", "search_space.json")


def train_schema_context():
    """Inspect only the approved train schema before forming a hypothesis."""
    try:
        train_rows, _ = load_rich(data_dir())
        return analyze_train_schema(train_rows)
    except (OSError, ValueError, KeyError) as exc:
        return {"schema": "unavailable", "error": str(exc), "source": "train_only"}


def research_context():
    path = os.environ.get("LEADERBOARD_PATH", "runs/pure/validation_leaderboard.json")
    try:
        with open(path, encoding="utf-8") as fh:
            board = json.load(fh)
        registry_path = os.path.join(ROOT, "submission_ready", "checkpoint_registry.json")
        with open(registry_path, encoding="utf-8") as fh:
            registry = json.load(fh)
        initial_family_metrics = {
            family: {
                "checkpoint": entry.get("checkpoint"),
                "metrics": entry.get("validation_metrics", {}),
                "confirmation_metrics": entry.get("confirmation_metrics", {}),
            }
            for family, entry in registry.get("families", {}).items()
        }
        rows = [{"experiment": x.get("experiment"), "config": x.get("config", {}), "metrics": x.get("metrics"),
                 "confirmation_metrics": x.get("confirmation_metrics", {}),
                 "status": x.get("status"), "error": x.get("error"),
                 "test_access": x.get("test_access", False),
                 "recovery_events": x.get("recovery_events", [])}
                for x in board.get("experiments", [])]
        return {"epsilon": board.get("epsilon", 0.002), "best": board.get("best", {}).get("experiment"),
                "experiments": rows[-20:], "plateau_control": {
                    "threshold": float(os.environ.get("PLANNER_PLATEAU_THRESHOLD", "0.002")),
                    "streak": int(os.environ.get("PLANNER_PLATEAU_STREAK", "0")),
                    "triggered": os.environ.get("PLANNER_PLATEAU_TRIGGERED") == "1",
                    "unlock_other_tasks": os.environ.get("PLANNER_UNLOCK_OTHER_TASKS") == "1",
                }, "train_schema": train_schema_context(),
                "initial_family_checkpoints": initial_family_metrics}
    except (OSError, json.JSONDecodeError):
        return {"epsilon": 0.002, "best": None, "experiments": []}


def parse_json_response(text):
    """Accept strict JSON and common fenced JSON, but fail safely otherwise."""
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response")
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    object_match = re.search(r"\{.*\}", raw, re.S)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    preview = raw[:240].replace("\n", " ")
    raise RuntimeError(f"Gemini response was not valid JSON; preview={preview!r}")


def propose_next_experiment(model="gemini-3.7-flash"):
    excluded = {x.strip() for x in os.environ.get("PLANNER_EXCLUDE", "").split(",") if x.strip()}
    promoted = {x.strip() for x in os.environ.get("PLANNER_PROMOTED", "").split(",") if x.strip()}
    context = research_context()
    context["diagnosis"] = diagnose(context.get("experiments", []))
    if os.environ.get("PLANNER_PLATEAU_TRIGGERED") == "1":
        # The autonomous controller computes this from the current run, which
        # is more precise than the rolling leaderboard context.  Make the
        # strategy text agree with the enforced task-unlock decision.
        context["diagnosis"] = {
            **context["diagnosis"],
            "focus": "task_expansion",
            "conclusion": ("validation Primary plateaued for three consecutive "
                           "improvements; unlock another reviewed task family"),
            "next_strategy": "unlock_auxiliary_tasks",
            "plateau_streak": int(os.environ.get("PLANNER_PLATEAU_STREAK", "0")),
            "plateau_threshold": float(os.environ.get("PLANNER_PLATEAU_THRESHOLD", "0.002")),
            "plateau_triggered": True,
        }
    if os.environ.get("PLANNER_BACKEND", "local").lower() == "local":
        task = os.environ.get("PLANNER_TASK_ID", "weight_learning")
        index = int(os.environ.get("PLANNER_TASK_INDEX", "0"))
        previous_raw = os.environ.get("PLANNER_PREVIOUS_CONFIG", "")
        try:
            previous = json.loads(previous_raw) if previous_raw else None
        except json.JSONDecodeError:
            previous = None
        config = next_task_candidate(task, index, excluded, previous=previous)
        if config is None:
            raise RuntimeError(f"task {task} exhausted its reviewed candidates")
        excluded_strategies = set(json.loads(os.environ.get("PLANNER_STRATEGY_EXCLUDE", "[]")))
        available_strategies = [x for x in STRATEGY_CATALOG.get(task, ()) if x not in excluded_strategies]
        if not available_strategies:
            raise RuntimeError(f"task {task} exhausted its reviewed strategies")
        current_strategy = os.environ.get("PLANNER_CURRENT_STRATEGY")
        strategy_id = (current_strategy if current_strategy in available_strategies
                       else available_strategies[0])
        evidence = context.get("experiments", [])[-1:] or [{"metrics": {}}]
        hypothesis = {
            "weight_learning": "五個 frozen family prediction 先完成 row alignment 與 user-level normalization，再學習 nonnegative weights，建立可重現的融合 baseline。",
            "additive_interaction": f"依 Task 1 family weights 排序 pure feature candidates，逐步加入 {config.get('selected_features', [])}，只保留 validation Primary 通過 gate 的 feature。",
            "dnn_composition": f"在五個 frozen predictions 與 selected pure data 上使用低容量 {config.get('composition_model')}，學習非線性 composition 並以 confirmation 防止 overfitting。",
            "multi_seed_confirmation": f"使用 seed={config.get('composition_seed')} 重估上一階段 DNN recipe；只有 mean/min primary 與 confirmation drift 都可接受才保留。",
        }[task]
        if previous:
            hypothesis += " 下一輪承接上一 task 的最佳 recipe，僅改變本輪 allowlist 指定的單一因素。"
        plan = {"module": "composition", "experiment": "composition_fm",
                "task_id": task, "iteration_in_task": index + 1,
                "strategy_id": strategy_id,
                "splits": ["train", "valid"], "files": ["agent/experiment_specs.json", "agent/configs/search_space.json"],
                "config": config, "strategy": context["diagnosis"].get("next_strategy", "task_based_search"),
                "hypothesis": hypothesis,
                "evidence": evidence,
                "change_plan": {"families": list(config["components"]), "weights": config["weights"],
                                "features": config.get("selected_features", [config.get("feature_set", "none")]),
                                "normalization": "user_zscore", "composition_model": config["composition_model"],
                                "loss": config["composition_loss"], "target": "long_view"},
                "expected_effect": "提升 validation primary 並由 confirmation guard 檢查泛化。",
                "success_criteria": {"primary_improvement": ">= 0.002"},
                "next_if_failure": "記錄失敗原因並由 task transition 規則改變下一個 allowlisted recipe。",
                "plateau_control": context.get("plateau_control", {}),
                "schema_analysis": context.get("train_schema", {}),
                "diagnosis": context["diagnosis"]}
        validate_plan(plan)
        return plan
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in this terminal")
    client = genai.Client(api_key=key)
    models = [model] + [x.strip() for x in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash"
    ).split(",") if x.strip() and x.strip() != model]
    catalog = config_catalog([experiment for _, experiment in LOCAL_PLANS])
    prompt = (
        "Return JSON only with keys module, experiment, splits, files, hypothesis, config. "
        f"Choose exactly one composition_fm task from: {list(TASK_BUDGETS)}. "
        f"Choose one registered config from this per-experiment catalog: {json.dumps(catalog)}. "
        f"Do not choose these already-tried experiment/config keys: {sorted(excluded)}. "
        "All five family checkpoints are frozen and must always be used; never exclude a family. "
        f"Research context from validation only: {json.dumps(context, ensure_ascii=False)}. "
        "Use train_schema to first reason about which user/video/context fields are available and safe. "
        "Use the supplied task_id and task budget; propose exactly one new composition recipe. "
        "Return a concise falsifiable hypothesis explaining why the proposed experiment should improve the metrics. "
        "For composition_fm, config may contain only allowlisted task fields: name, task_id, families, "
        "composition_code=11111, component_ids=[1,2,3,4,5], components (all five), positive weights "
        "summing to 1, bias, composition_seed, composition_model, composition_loss=0.6_listwise_0.4_bpr, "
        "feature_set, selected_features, interaction, lr, l2, epochs, patience. For additive_interaction, "
        "selected_features must contain only one newly proposed feature or a previously retained prefix. "
        "Do not return checkpoints or paths. "
        "Target is long_view. Use train and valid only. Do not propose test, hidden_test, arbitrary commands, "
        "deprecated bundle experiments, family checkpoint changes, or evaluator changes."
    )
    errors = []
    for current_model in models:
        for attempt in range(3):
            try:
                chat = client.chats.create(model=current_model)
                response = chat.send_message(prompt)
                plan = parse_json_response(response.text)
                validate_plan(plan)
                return plan
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                errors.append(f"{current_model} attempt {attempt + 1}: {code or type(exc).__name__}")
                if code not in (429, 500, 502, 503, 504):
                    raise
                if attempt < 2:
                    time.sleep(2 ** attempt)
    raise RuntimeError("Gemini models unavailable after retries: " + "; ".join(errors))


if __name__ == "__main__":
    print(json.dumps(propose_next_experiment(), ensure_ascii=False, indent=2))
