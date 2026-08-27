"""Safe Gemini planner: Chat API only, no arbitrary command execution."""
import json
import os
import re
import time

from google import genai

from experiment_registry import validate_plan


ALLOWED_EXPERIMENTS = "listwise_fm, history_fm, multitask_fm, cwm_fm"


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
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in this terminal")
    client = genai.Client(api_key=key)
    models = [model] + [x.strip() for x in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash"
    ).split(",") if x.strip() and x.strip() != model]
    prompt = (
        "Return JSON only with keys module, experiment, splits, files. "
        f"Choose exactly one experiment from: {ALLOWED_EXPERIMENTS}. "
        "Target is long_view. Use train and valid only. "
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
