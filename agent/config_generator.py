"""Bounded, deterministic AutoML configuration generator.

This is the only place where new parameter combinations are created. Values
come from the reviewed axes in configs/search_space.json; no model code or
shell command is generated dynamically.
"""
import hashlib
import itertools
import json
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
SEARCH_SPACE = os.path.join(ROOT, "configs", "search_space.json")


def _space():
    with open(SEARCH_SPACE, encoding="utf-8") as fh:
        return json.load(fh)


def _family_allowed(spec, family):
    families = spec.get("families")
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


def config_candidates(experiment=None):
    data = _space()
    fixed = [x for x in data.get("candidates", [])
             if not experiment or _family_allowed(x, experiment)]
    generated = _generated(experiment) if experiment else []
    seen = {x["name"] for x in fixed}
    return fixed + [x for x in generated if x["name"] not in seen]


def resolve_config(name, experiment):
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
