"""Validation-selected listwise bundle imported from the reviewed E: project."""
import os
import numpy as np

MEMBERS = (
    ("outputs/pure/listwise_bundle_validation_member.npz", 0.03830854561368621),
    ("outputs/pure/listwise_bundle_reference_member.npz", 0.13426802012707598),
    ("outputs/pure/listwise_bundle_pointwise_member.npz", 0.1221815089860066),
    ("outputs/pure/listwise_bundle_bpr_member.npz", 0.7052419252732313),
)

def predict_ensemble(root, model_class, dimension, features):
    """Return weighted logits from the imported, validation-selected bundle."""
    if not np.isclose(sum(weight for _, weight in MEMBERS), 1.0):
        raise ValueError("listwise bundle weights must sum to one")
    scores = np.zeros(len(features), dtype=np.float64)
    for relative_path, weight in MEMBERS:
        path = os.path.join(root, relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as saved:
            if saved["V"].shape[0] != dimension or saved["W"].shape != (dimension,):
                raise ValueError(f"incompatible listwise bundle checkpoint: {relative_path}")
            model = model_class(dimension, k=saved["V"].shape[1], seed=0)
            model.V[...] = saved["V"]
            model.W[...] = saved["W"]
            model.b = np.float32(saved["b"])
            scores += weight * model.predict(features)
    return scores
