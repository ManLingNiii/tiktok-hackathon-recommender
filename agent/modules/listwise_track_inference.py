"""Inference adapter for the external codex/listwise-track checkpoint.

The listwise track checkpoint stores the shared FM state (V, W, b), so ranking
inference can reuse the same encoded feature matrix as the local agent without
loading test data or retraining the model.
"""

from __future__ import annotations

import os
import numpy as np


DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "outputs", "pure", "listwise_initial.npz",
)


def predict(X: np.ndarray, checkpoint: str = DEFAULT_CHECKPOINT,
            batch_size: int = 200_000) -> np.ndarray:
    """Return branch-model logits for an already encoded feature matrix."""
    path = os.path.abspath(checkpoint)
    with np.load(path, allow_pickle=False) as saved:
        V = saved["V"]
        W = saved["W"]
        b = np.float32(saved["b"])
        if X.ndim != 2 or V.shape[0] <= int(X.max(initial=-1)):
            raise ValueError("encoded features are incompatible with checkpoint")
        outputs = []
        for start in range(0, len(X), batch_size):
            E = V[X[start:start + batch_size]]
            S = E.sum(axis=1)
            interaction = 0.5 * ((S ** 2).sum(axis=1) - (E ** 2).sum(axis=(1, 2)))
            outputs.append(b + W[X[start:start + batch_size]].sum(axis=1) + interaction)
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
