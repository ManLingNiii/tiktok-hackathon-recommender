"""Inference adapter for the canonical multitask checkpoint."""

from __future__ import annotations

import os
import numpy as np

DEFAULT_CHECKPOINT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "pure",
    "multitask_initial.npz",
))


def predict(X: np.ndarray, checkpoint: str = DEFAULT_CHECKPOINT,
            auxiliary_mix: float = 1.0, batch_size: int = 200_000) -> np.ndarray:
    """Predict the multitask primary score plus its trained click residual."""
    path = os.path.abspath(checkpoint)
    with np.load(path, allow_pickle=False) as saved:
        V, W, b, auxW = saved["V"], saved["W"], np.float32(saved["b"]), saved["auxW"]
        if X.ndim != 2 or (X.size and (X.min() < 0 or X.max() >= V.shape[0])):
            raise ValueError("encoded features are incompatible with checkpoint")
        scores = []
        for start in range(0, len(X), batch_size):
            block = X[start:start + batch_size]
            E = V[block]; S = E.sum(axis=1)
            base = b + W[block].sum(axis=1) + 0.5 * ((S ** 2).sum(axis=1) - (E ** 2).sum(axis=(1, 2)))
            aux = 1.0 / (1.0 + np.exp(-np.clip(auxW[block].sum(axis=1), -30, 30)))
            scores.append(base + auxiliary_mix * (aux - np.mean(aux)))
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
