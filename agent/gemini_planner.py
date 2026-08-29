"""Planner boundary with a local deterministic backend for offline rehearsal.

Set PLANNER_BACKEND=gemini to use Google Gemini after the local loop is proven.
"""
import json
import os
import re
import time

from experiment_registry import validate_plan
from config_generator import config_candidates, config_catalog


ALLOWED_EXPERIMENTS = "bpr_fm, listwise_fm, listwise_ensemble, history_fm, multitask_fm, cwm_fm"
LOCAL_PLANS = [
    ("bpr_loss", "bpr_fm"), ("listwise_loss", "listwise_fm"),
    ("listwise_ensemble", "listwise_ensemble"),
    ("history_features", "history_fm"), ("multitask", "multitask_fm"),
    ("censored_watch_time", "cwm_fm"),
]
SEARCH_SPACE = os.path.join(os.path.dirname(__file__), "configs", "search_space.json")


def research_context():
    path = os.environ.get("LEADERBOARD_PATH", "runs/validation_leaderboard.json")
    try:
        with open(path, encoding="utf-8") as fh:
            board = json.load(fh)
        rows = [{"experiment": x.get("experiment"), "config": x.get("config", {}), "metrics": x.get("metrics"),
                 "status": x.get("status"), "recovery_events": x.get("recovery_events", [])}
                for x in board.get("experiments", [])]
        return {"epsilon": board.get("epsilon", 0.002), "best": board.get("best", {}).get("experiment"),
                "experiments": rows[-20:]}
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
    if os.environ.get("PLANNER_BACKEND", "local").lower() == "local":
        scores = {x[1]: [] for x in LOCAL_PLANS}
        for row in context["experiments"]:
            if row.get("experiment") in scores and row.get("metrics", {}).get("primary") is not None:
                scores[row["experiment"]].append(row["metrics"]["primary"])
        # Prefer unexplored experiments; once a family has history, prioritize
        # the weakest result for a targeted follow-up rather than fixed order.
        available = [(module, experiment, config["name"])
                     for module, experiment in LOCAL_PLANS
                     for config in config_candidates(experiment)
                     if experiment not in promoted
                     and f"{experiment}:{config['name']}" not in excluded]
        available.sort(key=lambda item: (bool(scores[item[1]]),
                                         min(scores[item[1]]) if scores[item[1]] else float("inf")))
        if available:
            module, experiment, config = available[0]
            plan = {"module": module, "experiment": experiment,
                    "splits": ["train", "valid"],
                    "files": ["agent/experiment_specs.json"],
                    "config": config,
                    "strategy": "explore_untried_then_follow_up_weakest_primary"}
            validate_plan(plan)
            return plan
        raise RuntimeError("local planner exhausted current configs for unpromoted experiments")
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
        "Return JSON only with keys module, experiment, splits, files, config. "
        f"Choose exactly one experiment from: {ALLOWED_EXPERIMENTS}. "
        f"Choose one registered config from this per-experiment catalog: {json.dumps(catalog)}. "
        f"Do not choose these already-tried experiment/config keys: {sorted(excluded)}. "
        f"Already-promoted model families: {sorted(promoted)}; continue training all others. "
        f"Research context from validation only: {json.dumps(context, ensure_ascii=False)}. "
        "Return a registered config name in the config key. Target is long_view. Use train and valid only. "
        "Do not propose test, hidden_test, arbitrary commands, or evaluator changes."
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
