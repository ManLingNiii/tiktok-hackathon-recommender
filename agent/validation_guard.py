"""Deterministic validation-only confirmation split.

The official validation score remains the primary metric.  A fixed subset of
validation users is used only as a confirmation guard so repeated config search
cannot promote a candidate that improves one repeatedly inspected score while
failing on an untouched portion of validation.
"""
import hashlib

import numpy as np

from evaluate import evaluate


def confirmation_mask(user_ids, bucket=4):
    """Return a stable 1/bucket user-level mask without using test data."""
    values = []
    for user_id in user_ids:
        digest = hashlib.sha256(str(user_id).encode("utf-8")).digest()[0]
        values.append(digest % bucket == 0)
    mask = np.asarray(values, dtype=bool)
    # Avoid an accidental empty confirmation set on a tiny smoke-test slice.
    if mask.size and not mask.any():
        mask[np.argmin(np.asarray([str(x) for x in user_ids], dtype=object))] = True
    return mask


def evaluate_confirmation(user_ids, labels, scores):
    mask = confirmation_mask(user_ids)
    return evaluate(np.asarray(user_ids, dtype=object)[mask],
                    np.asarray(labels)[mask], np.asarray(scores)[mask])
