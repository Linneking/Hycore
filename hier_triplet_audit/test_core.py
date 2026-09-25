"""Small CPU-only checks for the no-training HIER audit."""

import unittest

import numpy as np
import torch

from core import (GridConfig, balanced_batches, draw_triplets, mine_pools,
                  mining_counts, poincare_dist_matrix, proxy_probe,
                  random_proxy_bank)


class AuditCoreTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(5), 8)
        generator = torch.Generator().manual_seed(22)
        self.points = torch.randn(40, 12, generator=generator) * 0.04
        self.distances = poincare_dist_matrix(self.points)

    def test_distance_symmetry_and_diagonal(self):
        self.assertTrue(torch.isfinite(self.distances).all())
        self.assertTrue(torch.allclose(self.distances, self.distances.T, atol=1e-5))
        self.assertTrue(torch.equal(self.distances.diag(), torch.zeros(40)))

    def test_balanced_batches_are_reproducible(self):
        a = balanced_batches(self.labels, 22, 3)
        b = balanced_batches(self.labels, 22, 3)
        for first, second in zip(a, b):
            self.assertTrue(np.array_equal(first, second))
            _, counts = np.unique(self.labels[first], return_counts=True)
            self.assertEqual(counts.tolist(), [8] * 5)

    def test_bonus_one_k8_makes_all_same_class_mutual(self):
        config = GridConfig("test", "hier", 8, 8, 1)
        pos, neg = mine_pools(self.distances, self.labels, config)
        for anchor in range(40):
            self.assertEqual(len(pos[anchor]), 7)
            self.assertTrue(np.all(self.labels[pos[anchor]] == self.labels[anchor]))
            self.assertFalse(np.any((self.labels[neg[anchor]] == self.labels[anchor]) &
                                    (neg[anchor] != anchor)))
            self.assertIn(anchor, neg[anchor])
        triplets = draw_triplets(pos, neg, 22, per_anchor=50)
        counts = mining_counts(self.distances, self.labels, pos, neg, triplets)
        self.assertEqual(counts["triplets"], 2000)
        self.assertEqual(counts["k_same_other"], 0)

    def test_class_conditioned_triplets_do_not_cross_class(self):
        config = GridConfig("within", "within", 4, 20)
        pos, neg = mine_pools(self.distances, self.labels, config)
        i, j, k = draw_triplets(pos, neg, 42)
        self.assertGreater(len(i), 0)
        self.assertTrue(np.all(self.labels[i] == self.labels[j]))
        self.assertTrue(np.all(self.labels[i] == self.labels[k]))
        self.assertTrue(np.all(i != k))

    def test_proxy_probe_is_finite(self):
        config = GridConfig("test", "hier", 8, 8, 1)
        pos, neg = mine_pools(self.distances, self.labels, config)
        triples = draw_triplets(pos, neg, 22, per_anchor=3)
        bank = random_proxy_bank(32, 12, 11, torch.device("cpu"))
        distance = poincare_dist_matrix(self.points, bank)
        result = proxy_probe(distance, triples, 123, chunk_size=64)
        self.assertEqual(result["triplets"], 120)
        self.assertLessEqual(result["effective"], result["pre_active"])
        self.assertLessEqual(result["collisions"] + result["effective"], 120)
        self.assertTrue(np.isfinite(result["masked_loss_sum"]))


if __name__ == "__main__":
    unittest.main()
