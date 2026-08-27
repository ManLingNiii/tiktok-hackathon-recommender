"""Pairwise BPR utilities for within-user ranking experiments."""
import numpy as np


def sample_pairs(users, labels, scores, seed=0):
    """Return (positive_index, negative_index) pairs within each user."""
    rng = np.random.default_rng(seed)
    by_user = {}
    for i, user in enumerate(users):
        by_user.setdefault(user, []).append(i)
    pairs = []
    for indices in by_user.values():
        pos = [i for i in indices if labels[i] > 0]
        neg = [i for i in indices if labels[i] <= 0]
        if pos and neg:
            pairs.extend((p, n) for p, n in zip(pos, rng.choice(neg, len(pos))))
    return pairs


def bpr_loss(pos_scores, neg_scores):
    """Mean -log(sigmoid(pos-neg)), numerically stable."""
    margin = np.asarray(pos_scores) - np.asarray(neg_scores)
    return float(np.mean(np.logaddexp(0.0, -margin)))
