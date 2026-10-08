"""Mathematical checks for frozen original HIER source attribution."""
import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch

from hier_original_activation_probe_v1 import (ROLE_KEYS, block_distance, frozen_indices,
                                              gradient_probe, instrumented_hhc,
                                              official_loss_object, official_runtime,
                                              record_counts)


REFERENCE = Path(os.environ.get('HIER_REFERENCE', 'D:/Hycore/.codex-local/references/HIER-CVPR23'))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.hier_cls, cls.hypnn, cls.pmath, cls.sampler_cls = official_runtime(REFERENCE)

    def test_collision_mask_and_roles(self):
        cp = torch.zeros(3, 1, requires_grad=True)
        indices = (np.array([0, 1]), np.array([1, 2]), np.array([0, 0]))
        loss, rec = instrumented_hhc(cp, indices)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(rec['collision_count'], 2)
        self.assertEqual(rec['self_k'], 1)
        self.assertEqual(rec['positive_hinge'].sum(), 0)
        counts = {k: np.zeros(3, np.int64) for k in ROLE_KEYS}
        record_counts(counts, rec, 'proxy')
        self.assertEqual(counts['proxy_pair_selected'].sum(), 2)
        self.assertEqual(counts['proxy_pair_positive_hinge'].sum(), 0)
        self.assertEqual(counts['proxy_endpoint_k_selected'][0], 2)

    def test_exact_official_loss_gradient_and_additivity(self):
        torch.manual_seed(20)
        tangent = torch.randn(24, 12) / math.sqrt(12)
        obj = official_loss_object(self.hier_cls, self.hypnn, self.pmath, tangent)
        with torch.no_grad():
            whole = obj.to_hyperbolic(torch.randn(12, 12) / math.sqrt(12))
        rng = np.random.default_rng(9)
        sample_idx = tuple(rng.integers(0, 12, 80) for _ in range(3))
        proxy_idx = tuple(rng.integers(0, 24, 80) for _ in range(3))
        results, vectors, checks = gradient_probe(obj, self.pmath, whole, sample_idx, proxy_idx,
                                                  seed=8, reference_check=True)
        self.assertLess(checks['gradient_additivity_max_abs_delta'], 1e-6)
        for source in ('sample', 'proxy'):
            self.assertEqual(checks[source]['loss_abs_delta'], 0.)
            self.assertLess(checks[source]['gradient_max_abs_delta'], 1e-7)
            for mode in ('direct', 'full_st'):
                g = results[f'{source}_{mode}']
                np.testing.assert_allclose(g['norm'] ** 2,
                                           g['radial_signed'] ** 2 + g['angular_norm'] ** 2,
                                           atol=1e-15)

    def test_official_global_sampler(self):
        labels = np.repeat(np.arange(100), 5)
        batch = frozen_indices(self.sampler_cls, labels, 2, 18, 44)
        self.assertEqual(len(batch), 180)
        classes, counts = np.unique(labels[batch], return_counts=True)
        self.assertEqual(len(classes), 90)
        self.assertTrue(np.all(counts == 2))
        np.testing.assert_array_equal(batch, frozen_indices(self.sampler_cls, labels, 2, 18, 44))

    def test_chunked_distance_equals_official(self):
        torch.manual_seed(34)
        x = torch.randn(12, 16) * .2
        y = torch.randn(24, 16) * .2
        expected = self.pmath.dist_matrix(x, y, c=.1)
        actual = block_distance(x, y, self.pmath, .1, block=3)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_complete_official_forward_source_order(self):
        torch.manual_seed(340)
        tangent = torch.randn(24, 12) / math.sqrt(12)
        obj = official_loss_object(self.hier_cls, self.hypnn, self.pmath, tangent)
        with torch.no_grad():
            whole = obj.to_hyperbolic(torch.randn(16, 12) / math.sqrt(12))
        labels = torch.repeat_interleave(torch.arange(8), 2)
        np.random.seed(81)
        torch.manual_seed(91)
        reference = obj(whole, labels, topk=5)
        reference_gradient, = torch.autograd.grad(reference, obj.lcas)
        np.random.seed(81)
        torch.manual_seed(91)
        proxy = obj.to_hyperbolic(obj.lcas)
        distances = obj.dist_f(torch.cat([whole, proxy]), torch.cat([whole, proxy]))
        sample_sim = torch.exp(-distances[:16, :16]).detach()
        sample_sim[labels[:, None] == labels[None, :]] += 1
        sample_indices = obj.get_reciprocal_triplets(sample_sim, topk=5, t_per_anchor=50)
        sample_loss, _ = instrumented_hhc(distances[:16, 16:], sample_indices)
        proxy_sim = torch.exp(-distances[16:, 16:]).detach()
        proxy_indices = obj.get_reciprocal_triplets(proxy_sim, topk=5, t_per_anchor=50)
        proxy_loss, _ = instrumented_hhc(distances[16:, 16:], proxy_indices)
        loss = sample_loss + proxy_loss
        gradient, = torch.autograd.grad(loss, obj.lcas)
        torch.testing.assert_close(loss, reference, atol=0., rtol=0.)
        torch.testing.assert_close(gradient, reference_gradient, atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
