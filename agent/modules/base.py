"""Stable interfaces shared by all allowlisted headroom modules."""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Protocol


@dataclass
class ExperimentContext:
    target: str = "long_view"
    split: str = "validation_only"
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class HeadroomModule(Protocol):
    name: str

    def validate(self, ctx: ExperimentContext) -> None: ...
    def fit(self, train_rows: Iterable[Any], ctx: ExperimentContext) -> Any: ...
    def predict(self, state: Any, rows: Iterable[Any], ctx: ExperimentContext) -> Any: ...


def require_safe_context(ctx: ExperimentContext) -> None:
    if ctx.target != "long_view":
        raise ValueError("only long_view is allowed")
    if ctx.split != "validation_only":
        raise ValueError("headroom modules may run only on validation_only")
