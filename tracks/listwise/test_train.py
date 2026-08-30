"""Tests for same-user hard list construction."""

import unittest

import numpy as np

from model import ExposureGroup
from train import build_hard_user_lists, build_multi_slates, build_top5_boundary_lists


class HardUserListTest(unittest.TestCase):
    def test_keeps_all_positives_and_highest_scored_same_user_negatives(self) -> None:
        labels = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
        scores = np.asarray([0.0, 0.1, 0.9, 0.5, 0.0, 0.2], dtype=np.float32)
        groups = [ExposureGroup("u", np.arange(6), 2)]
        selected = build_hard_user_lists(groups, labels, scores, cap=4)
        self.assertEqual(selected[0].user_id, "u")
        self.assertEqual(selected[0].positives, 2)
        self.assertEqual(selected[0].row_indices.tolist(), [0, 2, 3, 4])

    def test_never_drops_the_only_negative_from_a_large_positive_list(self) -> None:
        labels = np.asarray([1, 1, 1, 1, 0], dtype=np.float32)
        scores = np.arange(5, dtype=np.float32)
        groups = [ExposureGroup("u", np.arange(5), 4)]
        selected = build_hard_user_lists(groups, labels, scores, cap=3)
        self.assertEqual(selected[0].row_indices.tolist(), [0, 1, 2, 3, 4])

    def test_mixed_slate_is_seeded_and_keeps_hard_negatives(self) -> None:
        labels = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        scores = np.asarray([0.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        groups = [ExposureGroup("u", np.arange(6), 1)]
        selected = build_hard_user_lists(
            groups,
            labels,
            scores,
            cap=3,
            hard_fraction=0.5,
            rng=np.random.default_rng(0),
        )
        rows = selected[0].row_indices.tolist()
        self.assertIn(0, rows)
        self.assertIn(1, rows)
        self.assertEqual(len(rows), 3)

    def test_top5_boundary_keeps_near_cutoff_and_random_tail(self) -> None:
        labels = np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, -2.0], dtype=np.float32)
        groups = [ExposureGroup("u", np.arange(7), 1)]
        selected = build_top5_boundary_lists(
            groups, labels, scores, cap=4, rng=np.random.default_rng(0), boundary_count=2
        )
        rows = selected[0].row_indices.tolist()
        self.assertIn(0, rows)
        self.assertIn(4, rows)
        self.assertTrue({3, 5} & set(rows))
        self.assertEqual(len(rows), 4)

    def test_multi_slate_returns_boundary_and_disjoint_random_lists(self) -> None:
        labels = np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, -2.0], dtype=np.float32)
        groups = [ExposureGroup("u", np.arange(7), 1)]
        slates = build_multi_slates(
            groups, labels, scores, cap=3, rng=np.random.default_rng(0)
        )
        self.assertEqual(len(slates), 2)
        self.assertTrue(all(s.user_id == "u" for s in slates))
        self.assertTrue(all(0 in s.row_indices for s in slates))
        first_negatives = set(slates[0].row_indices.tolist()) - {0}
        second_negatives = set(slates[1].row_indices.tolist()) - {0}
        self.assertFalse(first_negatives & second_negatives)


if __name__ == "__main__":
    unittest.main()
