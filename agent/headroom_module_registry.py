"""Single import/validation point for reviewed headroom modules."""
from .modules import (BPRLossModule, ListwiseLossModule, HistoryFeaturesModule,
                      MultiTaskModule, CensoredWatchTimeModule)
from .modules.base import ExperimentContext

MODULES = {
    "bpr_loss": BPRLossModule,
    "listwise_loss": ListwiseLossModule,
    "history_features": HistoryFeaturesModule,
    "multitask": MultiTaskModule,
    "censored_watch_time": CensoredWatchTimeModule,
}


def build(name, config=None):
    if name not in MODULES:
        raise ValueError(f"module is not allowlisted: {name}")
    config = config or {}
    module = MODULES[name]()
    module.validate(ExperimentContext(
        target=config.get("target", "long_view"),
        split=config.get("split", "validation_only"),
        seed=config.get("seed", 0), config=config))
    return module
