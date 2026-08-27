"""Loss adapters; training code can inject these without changing evaluation."""
import numpy as np


def pointwise_logloss(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, scores) - labels * scores))


def bpr_loss(pos_scores, neg_scores):
    """Stable BPR objective: -log sigmoid(pos-neg)."""
    delta = np.asarray(pos_scores, dtype=np.float64) - np.asarray(neg_scores, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, -delta)))


def listwise_loss(labels, scores):
    """Cross entropy over one user's exposure list."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.sum(y) <= 0:
        return 0.0
    logp = s - np.logaddexp.reduce(s)
    return float(-np.sum((y / np.sum(y)) * logp))


def listwise_logit_gradient(labels, scores):
    """Gradient of listwise cross-entropy w.r.t. logits for one exposure group."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.sum(y) <= 0:
        return np.zeros_like(s, dtype=np.float32)
    p = np.exp(s - np.max(s)); p /= np.sum(p)
    return (p - y / np.sum(y)).astype(np.float32)


def normalize_watch_time(play_time_ms, duration_ms):
    """Return bounded watch ratio; avoids millisecond-scale gradient explosions."""
    p = np.asarray(play_time_ms, dtype=np.float64)
    d = np.maximum(np.asarray(duration_ms, dtype=np.float64), 1.0)
    return np.clip(p / d, 0.0, 1.0).astype(np.float32)
