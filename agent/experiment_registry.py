"""Allowlist for experiments proposed by an LLM."""
import json
import os


ROOT = os.path.dirname(__file__)
with open(os.path.join(ROOT, "headroom_registry.json"), encoding="utf-8") as fh:
    REGISTRY = json.load(fh)


def validate_plan(plan):
    """Fail closed unless a plan names an allowed module and no test split."""
    module = plan.get("module")
    if module not in REGISTRY["allowed_modules"]:
        raise ValueError(f"module not allowed: {module!r}")
    if any(s in plan.get("splits", []) for s in REGISTRY["forbidden_splits"]):
        raise ValueError("test split is forbidden during development")
    if any(p in plan.get("files", []) for p in REGISTRY["forbidden_paths"]):
        raise ValueError("forbidden path in experiment plan")
    return True
