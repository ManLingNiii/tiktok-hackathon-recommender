"""Stable contract for adding a teammate headroom implementation.

Each new family should provide one registered command in
``experiment_specs.json`` and return the same result fields as the reviewed
trainers.  Checkpoint persistence remains centralized in checkpoint_manager.
"""
from __future__ import annotations

REQUIRED_RESULT_FIELDS = {
    "experiment", "dataset", "status", "split", "metrics", "test_access",
    "checkpoint", "checkpoint_saved", "family_best_metrics",
}
REQUIRED_METRICS = {"GAUC", "nDCG@5", "primary"}


def validate_result(record: dict) -> None:
    """Fail closed when a teammate trainer violates the result contract."""
    missing = sorted(REQUIRED_RESULT_FIELDS - set(record))
    if missing:
        raise ValueError("headroom result missing fields: " + ", ".join(missing))
    metrics = record.get("metrics") or {}
    missing_metrics = sorted(REQUIRED_METRICS - set(metrics))
    if missing_metrics:
        raise ValueError("headroom metrics missing fields: " + ", ".join(missing_metrics))
    if record.get("status") != "success":
        raise ValueError("headroom result is not successful")
    if record.get("split") != "validation_only":
        raise ValueError("headroom result must identify validation_only")
    if record.get("test_access") is not False:
        raise ValueError("headroom result must explicitly report test_access=false")


def trainer_checklist() -> tuple[str, ...]:
    """Human-readable checklist exposed to integration tooling and teammates."""
    return (
        "register one experiment in agent/experiment_specs.json",
        "use train/valid loaders and the shared evaluation function",
        "accept AGENT_MODEL_CONFIG and batch_size/seed where applicable",
        "save through checkpoint_manager.save_if_best",
        "return REQUIRED_RESULT_FIELDS and REQUIRED_METRICS",
        "never generate submission output from the validation trainer",
    )
