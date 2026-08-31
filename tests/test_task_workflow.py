import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "kuairand-starter-kit"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from config_generator import (TASK_BUDGETS, adaptive_composition_candidate,
                              composition_recipe_key, validate_composition_config)
from modules.composition import FAMILY_ORDER, _user_zscore, compose
from modules.context_composition import (_composition_forward, _loss_gradient, _pairs)
from autonomous_agent import hypothesis_status


class TaskWorkflowContractTests(unittest.TestCase):
    def predictions(self, n=6):
        users = np.asarray(["u1", "u1", "u1", "u2", "u2", "u2"], dtype=object)[:n]
        return {family: np.arange(n, dtype=float) + index for index, family in enumerate(FAMILY_ORDER)}, users

    def test_candidate_always_uses_all_five_and_fixed_loss(self):
        config = adaptive_composition_candidate(0, "weight_learning")
        self.assertEqual(config["composition_code"], "11111")
        self.assertEqual(config["components"], list(FAMILY_ORDER))
        self.assertEqual(config["component_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(config["composition_loss"], "0.6_listwise_0.4_bpr")
        self.assertTrue(validate_composition_config(config))

    def test_retained_weight_recipe_still_advances_reviewed_axis(self):
        previous = adaptive_composition_candidate(0, "weight_learning")
        next_config = adaptive_composition_candidate(1, "weight_learning", previous)
        self.assertNotEqual(composition_recipe_key(previous), composition_recipe_key(next_config))
        self.assertEqual(next_config["weights"], [0.3, 0.15, 0.15, 0.2, 0.2])
        validate_composition_config(next_config)

    def test_task1_accepts_finite_raw_weight_warm_start(self):
        config = adaptive_composition_candidate(0, "weight_learning")
        config["initial_raw_weights"] = [-.1, .0, .1, .2, -.2]
        self.assertTrue(validate_composition_config(config))

    def test_downstream_prediction_input_weights_are_explicit(self):
        config = adaptive_composition_candidate(0, "dnn_composition", {
            "selected_features": ["tab"],
            "weights": [.1, .2, .3, .2, .2],
        })
        config["prediction_input_weights"] = [.1, .2, .3, .2, .2]
        self.assertTrue(validate_composition_config(config))
        self.assertEqual(config["feature_set"], ["tab"])

    def test_four_tasks_total_fifty_and_unique_recipe(self):
        self.assertEqual(TASK_BUDGETS, {"weight_learning": 12, "additive_interaction": 16,
                                        "dnn_composition": 13, "multi_seed_confirmation": 9})
        recipes = []
        for task in TASK_BUDGETS:
            config = adaptive_composition_candidate(0, task)
            validate_composition_config(config)
            recipes.append(composition_recipe_key(config))
        self.assertEqual(len(set(recipes)), 4)

    def test_task2_adds_one_allowlisted_feature_from_task1_weights(self):
        previous = {"weights": [.1, .4, .1, .3, .1], "selected_features": []}
        config = adaptive_composition_candidate(0, "additive_interaction", previous)
        self.assertEqual(config["selected_features"], ["tab"])
        self.assertEqual(config["feature_set"], ["tab"])
        validate_composition_config(config)

        next_config = adaptive_composition_candidate(1, "additive_interaction",
                                                     {**previous, "selected_features": ["tab"]})
        self.assertEqual(len(next_config["selected_features"]), 2)
        self.assertNotEqual(next_config["selected_features"], config["selected_features"])

    def test_hypothesis_status_is_independent_of_promotion_gate(self):
        self.assertEqual(hypothesis_status(0.003), "strongly_supported")
        self.assertEqual(hypothesis_status(0.001), "supported")
        self.assertEqual(hypothesis_status(0.0), "unsupported")
        self.assertEqual(hypothesis_status(-0.001), "rejected")

    def test_bitmask_cannot_remove_family(self):
        predictions, users = self.predictions()
        with self.assertRaises(ValueError):
            compose(predictions, users, "10000", [.2] * 5)

    def test_all_five_are_row_aligned_and_influence_score(self):
        predictions, users = self.predictions()
        base, meta = compose(predictions, users, "11111", [.2] * 5)
        self.assertEqual(meta["enabled_families"], list(FAMILY_ORDER))
        changed = dict(predictions)
        changed["cwm_fm"] = changed["cwm_fm"] + np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
        new, _ = compose(changed, users, "11111", [.2] * 5)
        self.assertFalse(np.allclose(base, new))

    def test_zero_variance_user_is_finite(self):
        values = _user_zscore(np.asarray([2.0, 2.0, 1.0]), np.asarray(["u", "u", "v"], dtype=object))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[:2].tolist(), [0.0, 0.0])

    def test_bpr_pairs_are_within_user(self):
        users = np.asarray(["u1", "u1", "u2", "u2"], dtype=object)
        labels = np.asarray([1, 0, 1, 0])
        pairs = _pairs(users, labels, 0)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(all(users[p] == users[n] for p, n in pairs))

    def test_composition_loss_uses_finite_long_view_gradient(self):
        users = np.asarray(["u1", "u1", "u2", "u2"], dtype=object)
        labels = np.asarray([1, 0, 1, 0], dtype=float)
        pairs = _pairs(users, labels, 0)
        gradient, loss = _loss_gradient(np.asarray([.4, .1, .2, .0]), users, labels, pairs)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.isfinite(loss))

    def test_small_dnn_and_gated_composition_return_finite_scores(self):
        predictions = np.arange(30, dtype=float).reshape(6, 5) / 10.0
        features = np.arange(12, dtype=float).reshape(6, 2) / 10.0
        for model in ("gated_linear", "small_mlp"):
            if model == "gated_linear":
                state = {"base_logits": np.zeros(5), "bias": 0.0,
                         "gate_matrix": np.zeros((2, 5))}
            else:
                state = {"base_logits": np.zeros(5), "bias": 0.0,
                         "mlp_w1": np.zeros((7, 64)), "mlp_b1": np.zeros(64),
                         "mlp_w2": np.zeros((64, 32)), "mlp_b2": np.zeros(32),
                         "mlp_w3": np.zeros((32, 1)), "mlp_b3": np.zeros(1)}
            scores, _ = _composition_forward(predictions, features, state, model)
            self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
