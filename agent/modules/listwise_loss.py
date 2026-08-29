"""Listwise softmax utilities merged from the reviewed external bundle."""
import numpy as np


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def listwise_loss(users, labels, scores):
    """Cross-entropy against each user's normalized long_view relevance."""
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


def complete_user_batches(users, batch_size, rng):
    """Yield shuffled batches without splitting a user's exposure group."""
    users = np.asarray(users, dtype=object)
    order = np.argsort(users, kind="stable")
    starts = np.r_[0, 1 + np.flatnonzero(users[order][1:] != users[order][:-1])]
    ends = np.r_[starts[1:], len(order)]
    groups = [order[start:end] for start, end in zip(starts, ends)]
    current = []
    size = 0
    for group_index in rng.permutation(len(groups)):
        group = groups[group_index]
        if current and size + len(group) > batch_size:
            yield np.concatenate(current)
            current, size = [], 0
        if len(group) > batch_size:
            if current:
                yield np.concatenate(current)
                current, size = [], 0
            for start in range(0, len(group), batch_size):
                yield group[start:start + batch_size]
        else:
            current.append(group)
            size += len(group)
    if current:
        yield np.concatenate(current)


def grouped_listwise_gradient(users, labels, scores, temperature=1.0):
    """Return stable per-logit gradients for complete exposure groups."""
    users = np.asarray(users, dtype=object)
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    temperature = max(float(temperature), 0.25)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1])]
    ends = np.r_[starts[1:], len(order)]
    logits = scores[order] / temperature
    targets = labels[order]
    maxima = np.maximum.reduceat(logits, starts)
    exponentials = np.exp(logits - np.repeat(maxima, ends - starts))
    denominators = np.add.reduceat(exponentials, starts)
    positives = np.add.reduceat(targets, starts)
    probabilities = exponentials / np.repeat(denominators, ends - starts)
    target_distribution = np.divide(
        targets, np.repeat(positives, ends - starts),
        out=np.zeros_like(targets),
        where=np.repeat(positives > 0, ends - starts),
    )
    gradient_sorted = (probabilities - target_distribution) / temperature
    gradient_sorted[np.repeat(positives <= 0, ends - starts)] = 0.0
    gradient = np.empty_like(gradient_sorted)
    gradient[order] = gradient_sorted
    return gradient
