"""Validation-only checkpoint management.

Keep one canonical, best-so-far checkpoint per experiment family.  A run may
train and early-stop in memory, but it writes an artifact only when its final
validation primary improves that family's recorded best.
"""
import json
import os
import tempfile

import numpy as np
try:
    from dataset_config import outputs_dir, runs_dir
except ImportError:
    from agent.dataset_config import outputs_dir, runs_dir


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS = outputs_dir()
RUNS = runs_dir()
BEST_META = os.path.join(RUNS, "best_models.json")
TOLERANCE = 1e-5


def _load_metadata():
    try:
        with open(BEST_META, encoding="utf-8") as fh:
            value = json.load(fh)
        if "models" in value and isinstance(value["models"], dict):
            return value
        # Migrate the old single-global-best shape without losing its evidence.
        if value.get("experiment"):
            return {"models": {value["experiment"]: value}}
    except (OSError, json.JSONDecodeError):
        pass
    return {"models": {}}


def _write_metadata(value):
    os.makedirs(RUNS, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="best_models_", suffix=".json", dir=RUNS)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
        os.replace(temp_path, BEST_META)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def canonical_path(experiment):
    return os.path.join(OUTPUTS, f"{experiment}_best.npz")


def load_best_parameters(experiment):
    """Load a family's canonical parameters for an explicit warm-start.

    Warm-starting is opt-in so ordinary config comparisons remain independent
    runs and do not inherit a previous candidate's validation decisions.
    """
    metadata = _load_metadata().get("models", {}).get(experiment, {})
    path = metadata.get("checkpoint") or canonical_path(experiment)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        required = {"V", "W", "b"}
        if not required.issubset(data.files):
            return None
        return {key: np.asarray(data[key]).copy() for key in required}


def startup_weights(experiments):
    """Return the family checkpoints that an agent will use at startup.

    This is intentionally read-only: it makes the warm-start contract
    inspectable without rewriting or promoting any weights.
    """
    models = _load_metadata().get("models", {})
    report = {}
    for experiment in experiments:
        meta = models.get(experiment, {})
        path = meta.get("checkpoint") or canonical_path(experiment)
        params = load_best_parameters(experiment)
        report[experiment] = {
            "checkpoint": os.path.abspath(path),
            "exists": bool(params is not None),
            "primary": meta.get("primary"),
            "config": meta.get("config", {}),
        }
    return report


def save_if_best(experiment, arrays, metrics, config=None, source=None):
    """Save the family checkpoint only if validation primary improves.

    Returns metadata for the family, including whether this call wrote the
    checkpoint.  ``arrays`` is a mapping accepted by ``numpy.savez``.
    """
    metrics = {key: float(value) for key, value in metrics.items()}
    data = _load_metadata()
    models = data.setdefault("models", {})
    previous = models.get(experiment, {})
    previous_primary = previous.get("primary")
    primary = metrics.get("primary", float("-inf"))
    path = canonical_path(experiment)
    saved = previous_primary is None or primary > float(previous_primary) + TOLERANCE
    if saved:
        os.makedirs(OUTPUTS, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f"{experiment}_", suffix=".npz", dir=OUTPUTS)
        os.close(fd)
        try:
            np.savez(temp_path, **arrays)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        models[experiment] = {
            "experiment": experiment,
            "primary": primary,
            "metrics": metrics,
            "config": config or {},
            "checkpoint": path,
            "source": source or "validation_only",
        }
        _write_metadata(data)
        current = models[experiment]
    else:
        current = previous
    return {
        **current,
        "checkpoint_saved": bool(saved),
        "candidate_metrics": metrics,
    }
