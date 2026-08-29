"""Frozen prediction adapter for the manifest-selected best model.

The adapter is deliberately separate from the agent loop: it reads the
submission manifest, validates its safety contract, loads only the listed
checkpoint members, and returns one score per encoded input row.
"""
import json
import os

import numpy as np

from modules.listwise_ensemble import predict_ensemble


def load_frozen_manifest(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "prepared_not_submitted":
        raise ValueError("manifest is not a frozen, not-submitted package")
    if manifest.get("model") != "listwise_ensemble":
        raise ValueError("submission adapter only supports listwise_ensemble")
    if manifest.get("validation_only") is not True or manifest.get("test_used") is not False:
        raise ValueError("unsafe manifest: validation-only/test-used contract failed")
    members = manifest.get("ensemble_members")
    if not isinstance(members, list) or not members:
        raise ValueError("manifest has no ensemble members")
    weights = []
    root = os.path.abspath(os.path.join(os.path.dirname(path), os.pardir))
    normalized = []
    for member in members:
        relative = member.get("path")
        weight = float(member.get("weight"))
        if not isinstance(relative, str) or os.path.isabs(relative):
            raise ValueError("manifest checkpoint paths must be relative")
        checkpoint = os.path.abspath(os.path.join(root, relative))
        if os.path.commonpath([root, checkpoint]) != root:
            raise ValueError("manifest checkpoint escapes project root")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("manifest weights must be finite and non-negative")
        normalized.append((os.path.relpath(checkpoint, root), weight))
        weights.append(weight)
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("manifest ensemble weights must sum to one")
    return root, tuple(normalized), manifest


def predict_from_manifest(manifest_path, model_class, dimension, features):
    root, members, _ = load_frozen_manifest(manifest_path)
    # Keep the reviewed ensemble implementation as the only checkpoint loader.
    # The temporary member list is scoped to this call and restored afterwards.
    from modules import listwise_ensemble
    original = listwise_ensemble.MEMBERS
    try:
        listwise_ensemble.MEMBERS = members
        return predict_ensemble(root, model_class, dimension, features)
    finally:
        listwise_ensemble.MEMBERS = original
