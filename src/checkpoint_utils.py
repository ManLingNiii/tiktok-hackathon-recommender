"""NumPy checkpoint export helpers for the BPR training entry points."""
import os
import subprocess

import numpy as np


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_numpy_checkpoint(path, model, *, dim, k, lr, l2, seed,
                          best_epoch, valid_metrics, n_neg, method,
                          dataset="KuaiRand-Pure", split_version="v1"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, V=model.V.copy(), W=model.W.copy(), b=np.asarray(model.b),
             dim=np.int64(dim), k=np.int64(k), lr=np.float64(lr), l2=np.float64(l2),
             seed=np.int64(seed), best_epoch=np.int64(best_epoch),
             valid_GAUC=np.float64(valid_metrics["GAUC"]),
             valid_nDCG_at_5=np.float64(valid_metrics["nDCG@5"]),
             valid_primary=np.float64(valid_metrics["primary"]),
             n_neg=np.int64(n_neg), method=np.asarray(method),
             dataset=np.asarray(dataset), split_version=np.asarray(split_version),
             git_commit=np.asarray(current_git_commit() or ""))
    return path
