"""Listwise softmax loss utilities for within-user exposure lists."""
import numpy as np


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def listwise_loss(users, labels, scores):
    """Cross-entropy against the user's normalized long_view relevance."""
    total = 0.0
    groups = 0
    by_user = {}
    for user, label, score in zip(users, labels, scores):
        by_user.setdefault(user, [[], []])
        by_user[user][0].append(float(label))
        by_user[user][1].append(float(score))
    for labels_u, scores_u in by_user.values():
        mass = sum(labels_u)
        if mass <= 0:
            continue
        target = np.asarray(labels_u, dtype=np.float64) / mass
        pred = softmax(scores_u)
        total -= float(np.sum(target * np.log(pred + 1e-12)))
        groups += 1
    return total / groups if groups else 0.0
