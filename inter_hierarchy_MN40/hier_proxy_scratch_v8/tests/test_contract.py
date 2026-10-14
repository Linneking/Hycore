from __future__ import annotations
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest
import geoopt
import torch
from torch import nn
from ..geometry import (C, to_ball, depth, InverseMetric, official_distance,
                        configure_backbone, assert_output_geometry)
from ..hier_loss import HIERLoss, ancestor_summary_c
from ..base_protocol import intra_losses
from ..telemetry import EpochTelemetry
from ...hier_proxy_scratch_v5.sampler import SourceClassBatchSampler


class GeometryContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        torch.set_num_threads(2)

    def test_cap_units_and_curve_isolation(self):
        for c in (.1, 1.):
            v = torch.tensor([[2.07, 0.], [10., 0.]], dtype=torch.float64)
            out = to_ball(v, c=c, hook=False)
            constructed_c = float(torch.as_tensor(c))
            self.assertAlmostEqual(float(depth(out, c=constructed_c)[0]), 4.14, places=10)
            self.assertLessEqual(float(depth(out, c=constructed_c).max()), 4.6)
        x = torch.tensor([[.3, .2]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[.2, .1]], dtype=torch.float64, requires_grad=True)
        loss = InverseMetric.apply(x, .1).sum() + InverseMetric.apply(y, 1.).sum()
        loss.backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x) * (1 - .1 * x.detach().square().sum()).square() / 4)
        torch.testing.assert_close(y.grad, torch.ones_like(y) * (1 - y.detach().square().sum()).square() / 4)

    def test_intra_margins_and_gradients(self):
        mu = to_ball(torch.tensor([[2.3, 0.], [-2.3, 0.]], dtype=torch.float64), hook=False).requires_grad_()
        nu = to_ball(torch.tensor([[1.05, 0.], [-1.05, 0.]], dtype=torch.float64), hook=False).requires_grad_()
        values = intra_losses(mu, nu, 200)
        self.assertLess(float(values['intra_radial'].detach()), 3e-5)
        self.assertEqual(float(values['intra_contrastive'].detach()), 0.)
        for n in (200, 400, 600):
            radial = intra_losses(mu, mu, n)['intra_radial']
            self.assertAlmostEqual(float(radial.detach()), 500 / n, places=9)
        loss = sum(intra_losses(mu, nu, 200)[k] for k in ('intra_radial', 'intra_contrastive'))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(mu.grad).all() and torch.isfinite(nu.grad).all()))

    def test_final_shared_cap_all_manifolds_and_one_hook(self):
        class Ball(geoopt.PoincareBall):
            def __init__(self, dim, c=1.):
                super().__init__(c=c)
                self.dim = dim
        class Mobius(nn.Module):
            def __init__(self, manifold, n, m):
                super().__init__()
                self.manifold = manifold
                self.weight = nn.Parameter(torch.randn(m, n) * .2)
                self.last_input = None
            def forward(self, x):
                self.last_input = x
                return self.manifold.mobius_matvec(self.weight, x)
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.manifold, self.manifold2 = Ball(512), Ball(256)
                self.emb = Mobius(self.manifold, 512, 256)
                self.classifier = Mobius(self.manifold2, 256, 40)
            def forward(self, x):
                mu = self.emb(self.manifold.expmap0(x))
                return mu, self.classifier(mu)
        model = configure_backbone(Backbone()).double()
        mu, logits = model(torch.randn(3, 512, dtype=torch.float64) * 2)
        self.assertIs(mu, model.classifier.last_input)
        assert_output_geometry(mu)
        self.assertEqual(len(model.emb._forward_hooks), 1)
        self.assertEqual(len(model.manifold2._forward_hooks), 0)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        with self.assertRaises(RuntimeError):
            configure_backbone(model)
        logits.sum().backward()
        self.assertTrue(bool(torch.isfinite(model.emb.weight.grad).all()))

    def test_hier_finite_roles_and_no_parameter_projection(self):
        proxy = HIERLoss(num_proxies=32, dim=8).double()
        mu = to_ball(torch.randn(64, 8, dtype=torch.float64)).requires_grad_()
        labels = torch.arange(32).repeat_interleave(2)
        loss, stats = proxy(mu, labels, topk=6, t_per_anchor=3, return_components=True,
                            generator=torch.Generator().manual_seed(22))
        self.assertEqual(stats['sample']['self_k_triplets'], 0)
        self.assertEqual(stats['proxy']['self_k_triplets'], 0)
        self.assertLessEqual(stats['geometry']['proxy']['depth_max'], 4.60001)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(proxy.tangent_proxies.grad).all()))
        self.assertTrue(bool(torch.isfinite(mu.grad).all()))
        with torch.no_grad():
            proxy.tangent_proxies[0] *= 10
        before = proxy.tangent_proxies.detach().clone()
        proxy.project_after_step(None)
        self.assertTrue(torch.equal(before, proxy.tangent_proxies.detach()))
        self.assertGreater(float(before[0].norm()), 2.3)

    def test_rank_disjoint_sampler(self):
        labels = torch.arange(40).repeat_interleave(4).tolist()
        a = SourceClassBatchSampler(labels, rank=0, steps=10)
        b = SourceClassBatchSampler(labels, rank=1, steps=10)
        for ia, ib in zip(a, b):
            la, lb = [labels[i] for i in ia], [labels[i] for i in ib]
            self.assertEqual(len(set(la)), 16)
            self.assertEqual(len(set(lb)), 16)
            self.assertFalse(set(la) & set(lb))
            self.assertTrue(all(la.count(c) == 2 for c in set(la)))

    def test_queue_rejects_compute_process_even_when_utilization_is_zero(self):
        from ..queue import idle_gpus
        rows = [{'index': 0, 'compute_busy': True, 'utilization': 0, 'free_MiB': 32000},
                {'index': 1, 'compute_busy': False, 'utilization': 0, 'free_MiB': 32000},
                {'index': 2, 'compute_busy': False, 'utilization': 10, 'free_MiB': 32000},
                {'index': 3, 'compute_busy': False, 'utilization': 0, 'free_MiB': 29000}]
        self.assertEqual([row['index'] for row in idle_gpus(rows)], [1])

    def test_telemetry_physical_depth_units(self):
        monitor = EpochTelemetry([1, 2], [0, 1], num_classes=40)
        points = to_ball(torch.tensor([[2., 0.], [2.3, 0.]], dtype=torch.float64), hook=False)
        monitor._observe_geometry('whole', points)
        result = monitor.summary()['geometry']['whole']
        self.assertAlmostEqual(result['depth']['mean'], 4.3, delta=3e-5)
        self.assertEqual(result['c'], .1)
        self.assertGreater(result['radius']['mean'], 1.)

    @unittest.skipUnless(os.environ.get('V8_OFFICIAL_HYPTORCH'), 'official source optional')
    def test_exact_official_forward_and_backward(self):
        root = Path(os.environ['V8_OFFICIAL_HYPTORCH'])
        pkg = types.ModuleType('hyptorch')
        pkg.__path__ = [str(root)]
        sys.modules['hyptorch'] = pkg
        spec = importlib.util.spec_from_file_location('hyptorch.pmath', root / 'pmath.py')
        pmath = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pmath
        spec.loader.exec_module(pmath)
        pkg.pmath = pmath
        spec2 = importlib.util.spec_from_file_location('hyptorch.nn', root / 'nn.py')
        original_nn = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(original_nn)
        for dtype, atol, rtol in ((torch.float64, 2e-12, 1e-10), (torch.float32, 3e-5, 3e-4)):
            a = torch.randn(17, 8, dtype=dtype, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            ref = original_nn.ToPoincare(c=.1, riemannian=True, clip_r=2.3)(a)
            actual = to_ball(b)
            torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)
            ref.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(b.grad, a.grad, atol=atol, rtol=rtol)
            x = ref.detach().clone().requires_grad_()
            y = actual.detach().clone().requires_grad_()
            dr = pmath.dist_matrix(x, x, c=.1)
            da = official_distance(y)
            torch.testing.assert_close(da, dr, atol=atol, rtol=rtol)
            mask = 1 - torch.eye(len(x), dtype=dtype)
            (dr * mask).sum().backward()
            (da * mask).sum().backward()
            torch.testing.assert_close(y.grad, x.grad, atol=atol * 10, rtol=rtol * 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
