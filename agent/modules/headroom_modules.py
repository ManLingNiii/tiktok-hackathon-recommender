"""Four reviewed, composable headroom modules.

These modules expose feature/loss transformations only. The runner remains the
single owner of data splitting, fitting, metric computation, and artifact writes.
"""
from dataclasses import dataclass
import numpy as np
from .base import ExperimentContext, require_safe_context
from .loss_adapter import bpr_loss, listwise_loss, listwise_logit_gradient, normalize_watch_time


@dataclass
class BPRLossModule:
    name: str = "bpr_loss"

    def validate(self, ctx): require_safe_context(ctx)
    def loss(self, positive_scores, negative_scores): return bpr_loss(positive_scores, negative_scores)


@dataclass
class ListwiseLossModule:
    name: str = "listwise_loss"

    def validate(self, ctx): require_safe_context(ctx)
    def loss(self, labels, scores): return listwise_loss(labels, scores)
    def gradient(self, labels, scores): return listwise_logit_gradient(labels, scores)


def add_history_features(rows):
    """Append leakage-safe prior user/tab/author counts.

    Rows must be sorted or sortable by (date, user_id); current-row labels are
    added only after features are computed.
    """
    rows = sorted(list(rows), key=lambda r: (str(r[0]), int(r[1])))
    user, tab, author = {}, {}, {}
    out = []
    for row in rows:
        date, uid, vid, aid, t, dur, label = row[:7]
        key_tab, key_author = (uid, t), (uid, aid)
        out.append(tuple(row) + (user.get(uid, 0), tab.get(key_tab, 0), author.get(key_author, 0)))
        user[uid] = user.get(uid, 0) + int(label)
        tab[key_tab] = tab.get(key_tab, 0) + int(label)
        author[key_author] = author.get(key_author, 0) + int(label)
    return out


@dataclass
class HistoryFeaturesModule:
    name: str = "history_features"

    def validate(self, ctx): require_safe_context(ctx)
    def transform(self, rows): return add_history_features(rows)


@dataclass
class MultiTaskModule:
    name: str = "multitask"
    auxiliary_weights: dict = None

    def validate(self, ctx): require_safe_context(ctx)
    def __post_init__(self):
        if self.auxiliary_weights is None:
            self.auxiliary_weights = {"is_click": 0.1, "is_like": 0.1, "is_follow": 0.05}
    def weighted_loss(self, main_loss, auxiliary_losses):
        return float(main_loss + sum(self.auxiliary_weights.get(k, 0.0) * v
                                     for k, v in auxiliary_losses.items()))


@dataclass
class CensoredWatchTimeModule:
    name: str = "censored_watch_time"
    margin: float = 0.0

    def validate(self, ctx): require_safe_context(ctx)
    def one_sided_loss(self, predicted, observed, censored):
        """For censored rows penalize only predictions below observed time."""
        pred, obs, cens = map(np.asarray, (predicted, observed, censored))
        uncensored = np.square(pred[~cens] - obs[~cens])
        censored_penalty = np.maximum(0.0, obs[cens] - pred[cens] + self.margin) ** 2
        parts = np.concatenate((uncensored, censored_penalty))
        return float(np.mean(parts)) if parts.size else 0.0

    def normalize(self, play_time_ms, duration_ms):
        return normalize_watch_time(play_time_ms, duration_ms)

    def normalized_one_sided_loss(self, predicted, play_time_ms, duration_ms, censored):
        target = self.normalize(play_time_ms, duration_ms)
        pred = np.asarray(predicted, dtype=np.float64)
        cens = np.asarray(censored, dtype=bool)
        err = np.where(cens, np.maximum(0.0, target - pred + self.margin), pred - target)
        return float(np.mean(err * err)) if err.size else 0.0
