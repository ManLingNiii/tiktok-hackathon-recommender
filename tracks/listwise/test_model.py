"""Focused tests for Listwise exposure grouping and loss."""

import unittest

import numpy as np

from model import (
    approx_ndcg_gradient,
    group_user_exposures,
    listwise_softmax_gradient,
    position_discounted_listnet_gradient,
)


class ExposureGroupingTest(unittest.TestCase):
    def test_complete_user_lists_preserve_row_order(self) -> None:
        users = ["a", "b", "a", "c", "b", "c"]
        labels = np.asarray([1, 0, 0, 1, 1, 1], dtype=np.float32)
        groups = group_user_exposures(users, labels)
        self.assertEqual([group.user_id for group in groups], ["a", "b", "c"])
        self.assertEqual(
            [group.row_indices.tolist() for group in groups],
            [[0, 2], [1, 4], [3, 5]],
        )
        self.assertEqual([group.positives for group in groups], [1, 1, 2])


class ListwiseLossTest(unittest.TestCase):
    def test_gradient_matches_finite_difference(self) -> None:
        logits = np.asarray([0.2, -0.3, 1.1, 0.4], dtype=np.float32)
        labels = np.asarray([1, 0, 0, 1], dtype=np.float32)
        _, analytic = listwise_softmax_gradient(logits, labels, [2, 2])
        epsilon = 1e-3
        numeric = []
        for index in range(len(logits)):
            upper = logits.copy()
            lower = logits.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            upper_loss, _ = listwise_softmax_gradient(upper, labels, [2, 2])
            lower_loss, _ = listwise_softmax_gradient(lower, labels, [2, 2])
            numeric.append((upper_loss - lower_loss) / (2 * epsilon))
        np.testing.assert_allclose(analytic, numeric, atol=2e-4)
        self.assertAlmostEqual(float(analytic[:2].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(analytic[2:].sum()), 0.0, places=6)

    def test_rejects_constant_label_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "discriminative"):
            listwise_softmax_gradient(
                np.asarray([0.1, 0.2]), np.asarray([0, 0]), [2]
            )

    def test_anchor_only_target_has_zero_gradient_at_baseline(self) -> None:
        baseline = np.asarray([0.7, -0.2, 0.1], dtype=np.float32)
        labels = np.asarray([0, 1, 0], dtype=np.float32)
        _, gradient = listwise_softmax_gradient(
            baseline,
            labels,
            [3],
            anchor_logits=baseline,
            anchor_mix=0.0,
        )
        np.testing.assert_allclose(gradient, 0.0, atol=1e-7)

    def test_approx_ndcg_gradient_matches_finite_difference(self) -> None:
        logits = np.asarray([0.4, -0.2, 0.8, 0.1], dtype=np.float32)
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        _, analytic = approx_ndcg_gradient(
            logits, labels, [4], rank_temperature=0.7, cutoff_temperature=0.6
        )
        epsilon = 1e-3
        numeric = []
        for index in range(len(logits)):
            upper = logits.copy()
            lower = logits.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            upper_loss, _ = approx_ndcg_gradient(
                upper, labels, [4], rank_temperature=0.7, cutoff_temperature=0.6
            )
            lower_loss, _ = approx_ndcg_gradient(
                lower, labels, [4], rank_temperature=0.7, cutoff_temperature=0.6
            )
            numeric.append((upper_loss - lower_loss) / (2 * epsilon))
        np.testing.assert_allclose(analytic, numeric, atol=2e-4)
        self.assertAlmostEqual(float(analytic.sum()), 0.0, places=6)

    def test_position_discounted_gradient_matches_finite_difference(self) -> None:
        logits = np.asarray([0.7, -0.4, 0.2, 1.1], dtype=np.float32)
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        _, analytic = position_discounted_listnet_gradient(
            logits, labels, [4], sort_temperature=0.8
        )
        epsilon = 1e-3
        numeric = []
        for index in range(len(logits)):
            upper = logits.copy()
            lower = logits.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            upper_loss, _ = position_discounted_listnet_gradient(
                upper, labels, [4], sort_temperature=0.8
            )
            lower_loss, _ = position_discounted_listnet_gradient(
                lower, labels, [4], sort_temperature=0.8
            )
            numeric.append((upper_loss - lower_loss) / (2 * epsilon))
        np.testing.assert_allclose(analytic, numeric, atol=3e-4)
        self.assertAlmostEqual(float(analytic.sum()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
