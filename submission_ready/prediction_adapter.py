"""Frozen prediction adapter for the manifest-selected best model.

The adapter is deliberately separate from the agent loop: it reads the
submission manifest, validates its safety contract, loads only the listed
checkpoint members, and returns one score per encoded input row.
"""
import json
import os

import numpy as np

from modules.composition import predict_composition
from modules.context_composition import predict_from_composition_checkpoint
from modules.context_composition import predict_from_context_checkpoint


def load_frozen_manifest(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "prepared_not_submitted":
        raise ValueError("manifest is not a frozen, not-submitted package")
    if manifest.get("model") != "composition_fm":
        raise ValueError("submission adapter only supports the current composition model")
    if manifest.get("validation_only") is not True or manifest.get("test_used") is not False:
        raise ValueError("unsafe manifest: validation-only/test-used contract failed")
    if manifest.get("model") == "composition_fm":
        recipe = manifest.get("composition", {})
        members = manifest.get("checkpoint_members")
        ids = recipe.get("component_ids")
        if recipe.get("composition_code") != "11111":
            raise ValueError("composition manifest must use all five families with code 11111")
        if recipe.get("components") != ["bpr_fm", "listwise_fm", "history_fm", "multitask_fm", "cwm_fm"]:
            raise ValueError("composition manifest must contain all five registered families")
        if ids is not None:
            mapping = manifest.get("family_id_map", {})
            if [mapping.get(str(x)) for x in ids] != recipe.get("components"):
                raise ValueError("composition numeric IDs do not match family mapping")
        if (not isinstance(members, list) or len(members) != 5
                or recipe.get("components") != [x.get("family") for x in members]):
            raise ValueError("composition manifest has no valid checkpoint recipe")
        composition_checkpoint = manifest.get("composition_checkpoint")
        if not isinstance(composition_checkpoint, str) or os.path.isabs(composition_checkpoint):
            raise ValueError("composition manifest has no relative composition checkpoint")
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
    checkpoint = os.path.abspath(os.path.join(root, manifest["composition_checkpoint"]))
    if os.path.commonpath([root, checkpoint]) != root or not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    return root, tuple(normalized), manifest


def predict_from_manifest(manifest_path, model_class, dimension, features, split="valid"):
    root, members, manifest = load_frozen_manifest(manifest_path)
    if manifest.get("test_used") is not False:
        raise ValueError("composition manifest violates test isolation")
    if manifest.get("composition_checkpoint"):
        return predict_from_composition_checkpoint(root, manifest["composition_checkpoint"], split=split)
    # Composition is currently a validation-preview adapter. Final test
    # generation remains a separate manually approved path.
    return predict_composition(root, {**manifest["composition"],
                                      "name": "manifest_composition"})
