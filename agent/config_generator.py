"""Bounded, deterministic AutoML configuration generator.

This is the only place where new parameter combinations are created. Values
come from the reviewed axes in configs/search_space.json; no model code or
shell command is generated dynamically.
"""
import hashlib
import itertools
import json
import math
import os
import random


ROOT = os.path.abspath(os.path.dirname(__file__))
SEARCH_SPACE = os.path.join(ROOT, "configs", "search_space.json")
COMPOSITION_FAMILIES = ["bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm"]
PURE_FEATURE_ALLOWLIST = ("duration_ms", "tab", "hourmin", "user_history_count",
                          "video_exposure_count")
FEATURE_FAMILY_MAP = {
    "duration_ms": ("bpr_fm", "cwm_fm"),
    "tab": ("listwise_fm", "multitask_fm"),
    "hourmin": ("listwise_fm",),
    "user_history_count": ("bpr_fm", "history_fm"),
    "video_exposure_count": ("bpr_fm",),
}
TASKS = (
    ("weight_learning", 12),
    ("additive_interaction", 16),
    ("dnn_composition", 13),
    ("multi_seed_confirmation", 9),
)
TASK_BUDGETS = dict(TASKS)
STRATEGY_CATALOG = {
    "weight_learning": ("weight_balance", "ranking_emphasis", "family_disagreement"),
    "additive_interaction": ("priority_feature", "feature_refinement", "prediction_interaction"),
    "dnn_composition": ("small_mlp_feature_set", "regularization_refinement", "normalization_refinement"),
    "multi_seed_confirmation": ("seed_confirmation", "confirmation_refinement"),
}


def composition_recipe_key(config):
    """Canonical recipe identity excluding the display-only candidate name."""
    return json.dumps({k: v for k, v in config.items() if k != "name"},
                      sort_keys=True, separators=(",", ":"))


def _space():
    with open(SEARCH_SPACE, encoding="utf-8") as fh:
        return json.load(fh)


def _family_allowed(spec, family):
    families = spec.get("families")
    if family == "composition_fm":
        return families == [family]
    return not families or family in families


def _generated(family):
    space = _space().get("generated", {})
    if not space.get("enabled", False) or family not in space.get("families", {}):
        return []
    spec = space["families"][family]
    axes = spec.get("axes", {})
    keys = list(axes)
    combos = []
    for values in itertools.product(*(axes[key] for key in keys)):
        config = dict(zip(keys, values))
        # Keep generated candidates deterministic and bounded. The hash is
        # only an identifier; all actual values remain visible in the config.
        fingerprint = json.dumps(config, sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]
        config["name"] = f"auto_{family}_{suffix}"
        config["families"] = [family]
        combos.append(config)
    # Hash ordering samples the Cartesian product across all axes instead of
    # taking only the first corner of the product when a family cap applies.
    combos.sort(key=lambda item: item["name"])
    return combos[:int(spec.get("max_candidates", 24))]


def _generated_compositions():
    """Create legacy catalog entries without ever dropping a family."""
    rng = random.Random(20260829)
    families = ["bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm"]
    result = []
    for index in range(50):
        selected = list(families)
        raw = [rng.uniform(0.1, 1.0) for _ in selected]
        total = sum(raw)
        weights = [round(value / total, 8) for value in raw]
        weights[-1] = round(1.0 - sum(weights[:-1]), 8)
        code = "".join("1" if family in selected else "0" for family in families)
        result.append({"name": f"random_composition_{index:02d}",
                       "families": ["composition_fm"],
                       "composition_code": code,
                       "component_ids": [1, 2, 3, 4, 5],
                       "components": selected, "weights": weights,
                       "composition_seed": index})
    return result


def adaptive_composition_candidate(index, focus="balanced", previous=None):
    """Generate exactly one task-aware candidate; all five families are fixed."""
    focus_hash = int(hashlib.sha1(str(focus).encode()).hexdigest()[:6], 16) % 1000
    index = int(index)
    task = focus if focus in TASK_BUDGETS else "five_family_weight_learning"
    families = list(COMPOSITION_FAMILIES)
    templates = _space().get("task_templates", {})
    spec = templates.get(task, {})
    axes = spec.get("axes", {})
    rng = random.Random(20260829 + index * 7919 + focus_hash)

    def pick(key, default):
        values = axes.get(key, [default])
        return values[index % len(values)]

    previous = previous if isinstance(previous, dict) else {}
    # A retained recipe is the starting point for the next proposal, but the
    # current task's reviewed axis must still be allowed to move.  In
    # particular, using previous weights unconditionally makes every
    # weight-learning proposal identical after the first retained candidate.
    if task == "weight_learning":
        raw = [float(x) for x in pick("weights", previous.get("weights", [.2, .2, .2, .2, .2]))]
    else:
        raw = [float(x) for x in (previous.get("weights") or
                                  pick("weights", [.2, .2, .2, .2, .2]))]
    if len(raw) != 5 or min(raw) <= 0:
        raw = [.2] * 5
    total = sum(raw)
    weights = [round(x / total, 8) for x in raw]
    weights[-1] = round(1.0 - sum(weights[:-1]), 8)
    previous_selected = [x for x in previous.get("selected_features", [])
                         if x in PURE_FEATURE_ALLOWLIST]
    if task == "additive_interaction":
        # Task 2 is sequential feature selection: priority orders the
        # allowlist, while the planner/evaluator decides whether to retain it.
        family_weights = dict(zip(COMPOSITION_FAMILIES, raw))
        ranked = sorted(
            (feature for feature in PURE_FEATURE_ALLOWLIST if feature not in previous_selected),
            key=lambda feature: (-sum(family_weights[f] for f in FEATURE_FAMILY_MAP[feature]), feature),
        )
        selected = previous_selected + (ranked[:1] if ranked else [])
    else:
        selected = previous_selected
    selected_input = list(selected)
    config = {
        "name": f"task_{task}_{index:03d}",
        "task_id": task,
        "families": ["composition_fm"], "composition_code": "11111",
        "component_ids": [1, 2, 3, 4, 5], "components": families,
        "weights": weights, "bias": 0.0,
        "composition_seed": int(pick("seed", index % 3)),
        "composition_model": pick("composition_model", previous.get("composition_model", "nonnegative_linear")),
        "composition_loss": "0.6_listwise_0.4_bpr",
        "feature_set": (selected_input if task in {"additive_interaction", "dnn_composition"}
                        else pick("feature_set", "none")),
        "selected_features": selected_input,
        "interaction": pick("interaction", "none" if task in {"weight_learning", "dnn_composition"}
                             else previous.get("interaction", "none")),
        "optimizer": "adam", "learning_rate": float(pick("lr", 0.02)),
        "lr": float(pick("lr", 0.02)), "l2": float(pick("l2", 0.01)),
        "epochs": int(pick("epochs", 8)), "patience": int(pick("patience", 2)),
    }
    if task == "weight_learning" and isinstance(previous.get("initial_raw_weights"), list):
        config["initial_raw_weights"] = list(previous["initial_raw_weights"])
    return config


def validate_composition_config(config):
    """Validate Gemini/local composition tuning without exposing family weights."""
    if not isinstance(config, dict):
        raise ValueError("composition config must be an object")
    # ``families`` is the workflow-level family selector used by the normal
    # registry.  Keep it explicit in the allowlist so an adaptive composition
    # can travel through the same Governor/runner contract as a catalog entry.
    allowed = {"name", "task_id", "families", "composition_code", "component_ids", "components",
               "weights", "initial_raw_weights", "prediction_input_weights", "bias", "composition_seed", "composition_model", "composition_loss",
               "feature_set", "selected_features", "interaction", "optimizer", "learning_rate", "lr", "l2", "epochs", "patience",
               "gate_lr", "gate_l2", "gate_epochs", "gate_patience", "gate_temperature"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"composition config contains forbidden fields: {sorted(unknown)}")
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("composition config requires a name")
    families = config.get("families", ["composition_fm"])
    if families != ["composition_fm"]:
        raise ValueError("composition config must select exactly composition_fm")
    components = config.get("components")
    ids = config.get("component_ids")
    weights = config.get("weights")
    if components != COMPOSITION_FAMILIES:
        raise ValueError("composition must use all five families in registered order")
    expected_ids = [COMPOSITION_FAMILIES.index(x) + 1 for x in components]
    if ids != expected_ids:
        raise ValueError("composition component_ids do not match family order")
    if config.get("composition_code") != "11111":
        raise ValueError("composition_code must be 11111; family exclusion is forbidden")
    if "task_id" in config and config["task_id"] not in TASK_BUDGETS:
        raise ValueError("task_id is not an allowlisted research task")
    if not isinstance(weights, list) or len(weights) != len(components):
        raise ValueError("composition weights must match components")
    numeric = [float(x) for x in weights]
    if any(not (0.0 < x <= 1.0) for x in numeric) or abs(sum(numeric) - 1.0) > 1e-6:
        raise ValueError("composition weights must be positive and sum to one")
    if "initial_raw_weights" in config:
        initial_raw = config["initial_raw_weights"]
        if (not isinstance(initial_raw, list) or len(initial_raw) != 5
                or any(not math.isfinite(float(x)) for x in initial_raw)):
            raise ValueError("initial_raw_weights must contain five finite logits")
    if "prediction_input_weights" in config:
        input_weights = config["prediction_input_weights"]
        if (not isinstance(input_weights, list) or len(input_weights) != 5
                or any(not math.isfinite(float(x)) or float(x) <= 0 for x in input_weights)
                or abs(sum(float(x) for x in input_weights) - 1.0) > 1e-6):
            raise ValueError("prediction_input_weights must be five positive normalized weights")
    seed = config.get("composition_seed", 0)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("composition_seed must be a non-negative integer")
    model = config.get("composition_model", "nonnegative_linear")
    if model not in {"nonnegative_linear", "gated_linear", "small_mlp"}:
        raise ValueError("composition_model is not allowlisted")
    loss = config.get("composition_loss", "0.6_listwise_0.4_bpr")
    if loss != "0.6_listwise_0.4_bpr":
        raise ValueError("composition loss must be 0.6_listwise_0.4_bpr")
    if config.get("optimizer", "adam") != "adam":
        raise ValueError("composition optimizer must be allowlisted Adam")
    feature_set = config.get("feature_set", "none")
    if isinstance(feature_set, list):
        if (len(set(feature_set)) != len(feature_set)
                or any(feature not in PURE_FEATURE_ALLOWLIST for feature in feature_set)):
            raise ValueError("feature_set list must contain unique allowlisted features")
    elif feature_set not in {"none", "pure_v1", "safe_context_v1",
                             "duration_ms", "tab", "hourmin",
                             "user_history_count", "video_exposure_count"}:
        raise ValueError("feature_set is not allowlisted")
    selected_features = config.get("selected_features", [])
    if (not isinstance(selected_features, list)
            or len(set(selected_features)) != len(selected_features)
            or any(feature not in PURE_FEATURE_ALLOWLIST for feature in selected_features)):
        raise ValueError("selected_features must be a unique list from the pure feature allowlist")
    if config.get("task_id") in {"additive_interaction", "dnn_composition"}:
        if isinstance(feature_set, str) and feature_set in {"pure_v1", "safe_context_v1"}:
            raise ValueError("Task 2/3 must use explicit selected_features, not a feature bundle")
        if feature_set != selected_features:
            raise ValueError("feature_set must exactly equal selected_features for Task 2/3")
    if config.get("interaction", "none") not in {"none", "p_bpr_duration", "p_listwise_duration",
                                                   "p_history_user_history", "p_cwm_duration",
                                                   "p_multitask_tab"}:
        raise ValueError("interaction is not allowlisted")
    for key in ("gate_lr", "gate_l2", "gate_temperature"):
        if key in config and (not math.isfinite(float(config[key])) or float(config[key]) <= 0):
            raise ValueError(f"{key} must be positive and finite")
    for key in ("gate_epochs", "gate_patience"):
        if key in config and (isinstance(config[key], bool) or int(config[key]) != config[key]
                              or int(config[key]) <= 0):
            raise ValueError(f"{key} must be a positive integer")
    return True


def config_candidates(experiment=None):
    data = _space()
    fixed = [x for x in data.get("candidates", [])
             if not experiment or _family_allowed(x, experiment)]
    generated = (_generated_compositions() if experiment == "composition_fm"
                 else _generated(experiment) if experiment else [])
    seen = {x["name"] for x in fixed}
    return fixed + [x for x in generated if x["name"] not in seen]


def next_allowlisted_candidate(experiment, excluded):
    """Return one untried fixed allowlist entry without expanding generated space."""
    excluded = set(excluded)
    for candidate in _space().get("candidates", []):
        if not _family_allowed(candidate, experiment):
            continue
        if f"{experiment}:{candidate.get('name')}" in excluded:
            continue
        return candidate
    return None


def next_task_candidate(task_id, index, excluded, previous=None):
    """Generate one deterministic candidate from the reviewed task template."""
    if task_id not in TASK_BUDGETS:
        raise ValueError(f"unknown task: {task_id}")
    excluded = set(excluded)
    for offset in range(max(1, TASK_BUDGETS[task_id] - int(index))):
        candidate = adaptive_composition_candidate(int(index) + offset, task_id, previous=previous)
        if f"{task_id}:{composition_recipe_key(candidate)}" in excluded:
            continue
        if f"composition_fm:{candidate['name']}" in excluded:
            continue
        validate_composition_config(candidate)
        return candidate
    return None


def resolve_config(name, experiment):
    if experiment == "composition_fm" and isinstance(name, str) and name.startswith("task_"):
        parts = name.split("_")
        if len(parts) >= 4 and parts[-1].isdigit():
            return adaptive_composition_candidate(int(parts[-1]), "_".join(parts[1:-1]))
    matches = [x for x in config_candidates(experiment) if x.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"unregistered or ambiguous model config: {name}")
    selected = matches[0]
    if not _family_allowed(selected, experiment):
        raise ValueError(f"config {name} is incompatible with {experiment}")
    return selected


def config_catalog(experiments):
    return {experiment: [x["name"] for x in config_candidates(experiment)]
            for experiment in experiments}
