"""Controlled cross-method prediction composition on validation rows only."""
import json
import os
import tempfile

import numpy as np

from baseline import FM
from data import encode
from validation_only import load_train_valid
from rich_data import encode_rich, load_rich
try:
    from multitask_inference import predict as predict_multitask
except ImportError:
    from modules.multitask_inference import predict as predict_multitask


def _history_external_features(train_rows, valid_rows):
    """Encode the history-final top-1 checkpoint schema.

    The external history track uses the five base fields plus one continuous
    author_affinity_rate input.  Its checkpoint stores that input separately
    as Vh/Wh, so it must not be passed through the generic rich encoder.
    """
    edges = np.quantile(np.asarray([x[5] for x in train_rows]), np.linspace(0, 1, 11)[1:-1])
    vocab = [dict() for _ in range(5)]

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    for x in train_rows:
        for i, value in enumerate(raw(x)):
            if value not in vocab[i]:
                vocab[i][value] = len(vocab[i])
    dims = [len(v) + 1 for v in vocab]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int32)
        for n, row in enumerate(rows):
            for i, value in enumerate(raw(row)):
                X[n, i] = vocab[i].get(value, len(vocab[i])) + offsets[i]
        return X

    # This mirrors history-final/train.py: validation history is frozen at the
    # end of train, while each train row sees only prior train interactions.
    q = edges
    global_rate = sum(x[6] for x in train_rows) / max(len(train_rows), 1)
    counts = {}
    totals = {}
    positives = {}
    def make(rows, prefix):
        out = np.empty((len(rows), 1), dtype=np.float32)
        recent = {}
        for i, x in enumerate(rows):
            u, author = x[1], x[3]
            bucket = int(np.searchsorted(q, x[5]))
            key = (u, author)
            n = totals.get(u, 0)
            z = counts.get(key, 0)
            ap = positives.get((u, author), 0.0)
            out[i, 0] = (ap + 20.0 * global_rate) / (z + 20.0)
            if prefix:
                y = float(x[6])
                counts[key] = z + 1
                totals[u] = n + 1
                positives[(u, author)] = ap + y
                recent.setdefault(u, []).append(y)
        return out
    return encode(train_rows), encode(valid_rows), make(train_rows, True), make(valid_rows, False), int(sum(dims))


def _score_history_external(saved, encoded, affinity):
    """Score rows using the external history-final checkpoint schema."""
    V, W, Vh, Wh, b = saved["V"], saved["W"], saved["Vh"], saved["Wh"], saved["b"]
    scores = np.empty(len(encoded), dtype=np.float64)
    for start in range(0, len(encoded), 8192):
        batch = encoded[start:start + 8192]
        E = V[batch]
        S = E.sum(1)
        H = affinity[start:start + 8192]
        Sh = Vh[None] * H[:, :, None]
        scores[start:start + len(batch)] = (b + W[batch].sum(1)
            + .5 * ((S * S).sum(1) - (E * E).sum((1, 2)))
            + H @ Wh + (S[:, None, :] * Sh).sum((1, 2)))
    return scores


def _predict_history_external(root, train_rows, valid_rows):
    path = _checkpoint(root, "history_fm")
    with np.load(path, allow_pickle=False) as saved:
        required = {"V", "W", "Vh", "Wh", "b"}
        if not required.issubset(saved.files):
            raise ValueError("history_fm checkpoint is not the history-final schema")
        _, Xv, _, Hv, dim = _history_external_features(train_rows, valid_rows)
        if saved["V"].shape[0] != dim or saved["W"].shape != (dim,):
            raise ValueError("history-final checkpoint is incompatible with the base encoder")
        return _score_history_external(saved, Xv, Hv)


def _predict_history_external_train_valid(root, train_rows, valid_rows):
    """Return train/valid predictions for the merged external History model."""
    path = _checkpoint(root, "history_fm")
    Xtr, Xv, Htr, Hv, dim = _history_external_features(train_rows, valid_rows)
    with np.load(path, allow_pickle=False) as saved:
        required = {"V", "W", "Vh", "Wh", "b"}
        if not required.issubset(saved.files):
            raise ValueError("history_fm checkpoint is not the history-final schema")
        if saved["V"].shape[0] != dim or saved["W"].shape != (dim,):
            raise ValueError("history-final checkpoint is incompatible with the base encoder")
        return (_score_history_external(saved, Xtr, Htr),
                _score_history_external(saved, Xv, Hv))


FAMILY_IDS = {"bpr_fm": 1, "listwise_fm": 2, "history_fm": 3,
              "multitask_fm": 4, "cwm_fm": 5}
ID_FAMILIES = {value: key for key, value in FAMILY_IDS.items()}
ALLOWED_COMPONENTS = set(FAMILY_IDS)
FAMILY_ORDER = ("bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm")
COMPOSITION_MANIFEST = os.path.join("submission_ready", "composition_manifest.json")
CHECKPOINT_REGISTRY = os.path.join("submission_ready", "checkpoint_registry.json")


def _load_registry(root):
    path = os.path.join(root, CHECKPOINT_REGISTRY)
    with open(path, encoding="utf-8") as fh:
        registry = json.load(fh)
    if registry.get("seed") != 0 or registry.get("validation_only") is not True:
        raise ValueError("checkpoint registry must be validation-only seed=0")
    entries = registry.get("families", {})
    if set(entries) != set(FAMILY_IDS):
        raise ValueError("checkpoint registry must contain exactly the five registered families")
    return registry


def _checkpoint(root, experiment):
    registry = _load_registry(root)
    relative = registry["families"][experiment]["checkpoint"]
    path = os.path.abspath(os.path.join(root, relative))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("checkpoint registry path escapes project root")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _predict(root, experiment, features, dimension):
    path = _checkpoint(root, experiment)
    if experiment == "multitask_fm":
        # The registered multitask artifact includes the auxiliary click head;
        # keep composition inference identical to the standalone adapter.
        return predict_multitask(features, checkpoint=path).astype(np.float64)
    with np.load(path, allow_pickle=False) as saved:
        if saved["V"].shape[0] != dimension or saved["W"].shape != (dimension,):
            raise ValueError(f"{experiment} checkpoint is incompatible with its encoder")
        model = FM(dimension, k=saved["V"].shape[1], seed=0)
        model.V[...] = saved["V"]; model.W[...] = saved["W"]; model.b = np.float32(saved["b"])
        scores = model.predict(features).astype(np.float64)
    return scores


def _user_zscore(values, user_ids):
    """Normalize scores within each user, preserving the original row order."""
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(user_ids, dtype=object)
    if len(values) != len(users):
        raise ValueError("values and user_ids must be row-aligned")
    if not len(values):
        return np.zeros_like(values)
    # Sorting once avoids allocating an N-sized boolean mask for every user.
    # This is important for five-family composition on the full Pure split.
    order = np.argsort(users, kind="stable")
    sorted_values = values[order]
    sorted_users = users[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1])]
    ends = np.r_[starts[1:], len(sorted_values)]
    counts = ends - starts
    means = np.add.reduceat(sorted_values, starts) / counts
    expanded_means = np.repeat(means, counts)
    variance = np.add.reduceat((sorted_values - expanded_means) ** 2, starts) / counts
    std = np.sqrt(variance)
    normalized = np.divide(sorted_values - expanded_means, np.repeat(std, counts),
                           out=np.zeros_like(sorted_values), where=np.repeat(std, counts) > 1e-12)
    output = np.zeros_like(values)
    output[order] = normalized
    return output


def family_prediction_payload(family, checkpoint, predictions, user_ids, labels):
    """Return the common row-aligned interface used by composition and tests."""
    predictions = np.asarray(predictions, dtype=np.float64)
    user_ids = np.asarray(user_ids, dtype=object)
    labels = np.asarray(labels)
    if not (len(predictions) == len(user_ids) == len(labels)):
        raise ValueError("family prediction payload is not row-aligned")
    return {"family": family, "checkpoint": os.path.abspath(checkpoint),
            "predictions": predictions, "user_ids": user_ids, "labels": labels,
            "row_count": int(len(predictions)), "validation_only": True,
            "test_access": False}


def compose(predictions, user_ids, bitmask="11111", weights=None, normalization="user_zscore", bias=0.0):
    """Fuse all five frozen family predictions using a nonnegative fusion layer.

    ``bitmask`` remains in the audit schema for backwards-compatible reporting,
    but it is deliberately not a selector anymore: composition is fail-closed
    unless all five registered families are present and the code is ``11111``.
    """
    if bitmask != "11111":
        raise ValueError("composition always uses all five families; composition_code must be 11111")
    missing = [family for family in FAMILY_ORDER if family not in predictions]
    if missing:
        raise ValueError(f"all five family predictions are required; missing: {missing}")
    lengths = {len(np.asarray(predictions[family])) for family in FAMILY_ORDER}
    if len(lengths) != 1 or len(np.asarray(user_ids)) not in lengths:
        raise ValueError("all family predictions and user_ids must be row-aligned")
    if normalization != "user_zscore":
        raise ValueError("only the reviewed user_zscore normalization is allowed")
    if weights is None:
        raw = np.ones(len(FAMILY_ORDER), dtype=np.float64)
    else:
        raw = np.asarray(weights, dtype=np.float64)
        if len(raw) != len(FAMILY_ORDER) or np.any(~np.isfinite(raw)) or np.any(raw <= 0):
            raise ValueError("weights must be finite, positive, and contain five values")
    if float(raw.sum()) <= 0 or np.isclose(raw.sum(), 0.0):
        raise ValueError("weights must have a positive sum")
    normalized_weights = raw / raw.sum()
    final = np.full(len(np.asarray(user_ids)), float(bias), dtype=np.float64)
    for family, weight in zip(FAMILY_ORDER, normalized_weights):
        final += float(weight) * _user_zscore(predictions[family], user_ids)
    return final, {"bitmask": bitmask, "enabled_families": list(FAMILY_ORDER),
                   "raw_weights": raw.tolist(),
                   "normalized_weights": normalized_weights.tolist(),
                   "normalization": normalization, "bias": float(bias)}


def predict_composition(root, config):
    component_ids = config.get("component_ids")
    if component_ids is not None:
        if (not isinstance(component_ids, list)
                or any(isinstance(x, bool) or int(x) != x for x in component_ids)):
            raise ValueError("composition component_ids must be integer family IDs")
        try:
            numeric_components = [ID_FAMILIES[int(x)] for x in component_ids]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("composition contains an unknown family ID") from exc
        declared = config.get("components")
        if declared is not None and declared != numeric_components:
            raise ValueError("composition IDs do not match declared family names")
        components = numeric_components
    else:
        components = config.get("components")
    if list(components) != list(FAMILY_ORDER):
        raise ValueError("composition must contain all five families in registered order")
    if config.get("composition_code") != "11111":
        raise ValueError("composition_code must be 11111; family exclusion is forbidden")
    seed = config.get("composition_seed", config.get("seed", 0))
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("composition_seed must be a non-negative integer")
    weights = config.get("weights")
    if not isinstance(components, list) or len(components) != len(weights):
        raise ValueError("composition requires five equally sized components and weights")
    if any(x not in ALLOWED_COMPONENTS for x in components) or len(set(components)) != len(components):
        raise ValueError("composition contains an unallowlisted or duplicate component")
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("composition weights must be finite, positive, and sum to one")
    # Basic BPR uses the official encoder; formal families use the reviewed
    # rich encoder. Both preserve the exact validation row order.
    registry = _load_registry(root)
    basic = ["bpr_fm"]
    rich = list(FAMILY_ORDER[1:])
    splits = load_train_valid(os.path.join(root, "kuairand-starter-kit", "KuaiRand-Pure", "data"))
    valid_rows = splits["valid"]
    user_ids = np.asarray([row[1] for row in valid_rows], dtype=object)
    predictions = {}
    if basic:
        encoded, dim = encode({**splits, "test": []})
        predictions["bpr_fm"] = _predict(root, "bpr_fm", encoded["valid"][0], dim)
    if rich:
        train_rows, valid_rows = load_rich(os.path.join(root, "kuairand-starter-kit", "KuaiRand-Pure", "data"))
        for family in rich:
            if family == "history_fm":
                predictions[family] = _predict_history_external(root, [x["base"] + (x["y"],) for x in train_rows],
                                                                [x["base"] + (x["y"],) for x in valid_rows])
            else:
                et, ev, dim = encode_rich(train_rows, valid_rows, include_history=False)
                predictions[family] = _predict(root, family, ev[0], dim)
    scores, metadata = compose(predictions, user_ids, "11111", weights,
                               config.get("normalization", "user_zscore"),
                               config.get("bias", 0.0))
    return scores


def save_best_composition_manifest(root, config, metrics, confirmation_metrics=None,
                                   composition_checkpoint=None, schema_analysis=None):
    """Persist the best validation-approved composition recipe, not new weights."""
    path = os.path.join(root, COMPOSITION_MANIFEST)
    previous = {}
    try:
        with open(path, encoding="utf-8") as fh:
            previous = json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass
    primary = float(metrics.get("primary", float("-inf")))
    previous_components = previous.get("composition", {}).get("components", [])
    previous_valid = (previous.get("model") == "composition_fm"
                      and all(x in FAMILY_IDS for x in previous_components)
                      and previous.get("composition", {}).get("component_ids")
                      == [FAMILY_IDS[x] for x in previous_components])
    if previous_valid and previous.get("validation_metrics", {}).get("primary") is not None:
        if primary <= float(previous["validation_metrics"]["primary"]):
            return {"path": path, "saved": False, "manifest": previous}
    components = list(FAMILY_ORDER)
    weights = config["weights"]
    registry = _load_registry(root)
    if composition_checkpoint is not None:
        composition_checkpoint = os.path.abspath(composition_checkpoint)
        if os.path.commonpath([root, composition_checkpoint]) != root:
            raise ValueError("composition checkpoint path escapes project root")
        composition_checkpoint = os.path.relpath(composition_checkpoint, root)
    composition_code = "11111"
    manifest = {
        "status": "prepared_not_submitted",
        "dataset": "KuaiRand-Pure",
        "target": "long_view",
        "selection_metric": "mean(GAUC, nDCG@5)",
        "validation_only": True,
        "test_used": False,
        "model": "composition_fm",
        "composition_model": config.get("composition_model", "nonnegative_linear"),
        "composition_loss": config.get("composition_loss", "0.6_listwise_0.4_bpr"),
        "composition": {"composition_code": composition_code,
                         "component_ids": [FAMILY_IDS[x] for x in components],
                         "components": components, "weights": weights,
                         "bias": float(config.get("bias", 0.0))},
        "normalization": "user_zscore",
        "feature_schema": config.get("feature_set", "none"),
        "family_id_map": {str(value): key for key, value in FAMILY_IDS.items()},
        "composition_seed": int(config.get("composition_seed", config.get("seed", 0))),
        "initial_seed": registry["seed"],
        "checkpoint_registry": CHECKPOINT_REGISTRY,
        "initial_checkpoint_registry": registry["families"],
        "validation_metrics": {k: float(v) for k, v in metrics.items()},
        "confirmation_metrics": ({k: float(v) for k, v in confirmation_metrics.items()}
                                  if confirmation_metrics else None),
        "checkpoint_members": [
            {"family": family, "path": registry["families"][family]["checkpoint"], "weight": weight}
            for family, weight in zip(components, weights)
        ],
        "composition_checkpoint": composition_checkpoint,
        "schema_analysis": schema_analysis,
        "generator": "submission_ready/generate_submission.py",
        "local_checker": "submission_ready/local_checker.py",
        "submission_csv": None,
        "final_test_check": "deferred_until_manual_confirmation",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="composition_manifest_", suffix=".json",
                                     dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return {"path": path, "saved": True, "manifest": manifest}
