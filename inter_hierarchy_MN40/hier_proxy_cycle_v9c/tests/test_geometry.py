"""CPU equivalence and one-GS-per-branch contracts."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import geoopt
import torch
from torch import nn
import torch.nn.functional as F

from .. import geometry as g


class SourceBall(geoopt.PoincareBall):
    def __init__(self, *, c, dim):
        super().__init__(c=c)
        self.dim = dim


class BallLayer(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.eye(3) * 1.7)

    def forward(self, points):
        return self.manifold.expmap0(F.linear(points, self.weight))


class Classifier(nn.Linear):
    def __init__(self, manifold):
        super().__init__(3, 40)
        self.manifold = manifold

    def forward(self, mu):
        return super().forward(self.manifold.logmap0(mu))


class ProbeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.manifold = SourceBall(c=1., dim=512)
        self.manifold2 = SourceBall(c=1., dim=256)
        self.emb = BallLayer(self.manifold)
        self.classifier = Classifier(self.manifold2)
        self.bn = nn.BatchNorm1d(3)
        self.calls = []

    def forward(self, points, emb=False):
        feature = self.bn(points.mean(-1))
        mu = self.emb(self.manifold.expmap0(feature))
        self.calls.append((bool(emb), points.detach().clone()))
        return mu, mu if emb else self.classifier(mu)


def gs_nodes(tensor):
    pending, seen, result = [tensor.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if 'InverseMetricBackward' in type(node).__name__:
            result.append(node)
        pending.extend(value for value, _ in node.next_functions)
    return len(result)


class GeometryContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4780)
        torch.set_num_threads(2)

    def test_configuration_matches_old_original_weights_rng_forward_and_gradient(self):
        from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020.train import configure_model
        original = ProbeBackbone().double().eval()
        candidate = copy.deepcopy(original)
        rng = torch.get_rng_state().clone()
        configure_model(original, True)
        g.configure_backbone(candidate)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()), set(candidate.state_dict()))
        for key in original.state_dict():
            torch.testing.assert_close(original.state_dict()[key], candidate.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(5, 3, 23, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = original(x), candidate(y)
        for aa, bb in zip(a, b):
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)
        loss_a = a[0].square().sum() + a[1].square().sum()
        loss_b = b[0].square().sum() + b[1].square().sum()
        loss_a.backward()
        loss_b.backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-14)
        for (ka, pa), (kb, pb) in zip(original.named_parameters(), candidate.named_parameters()):
            self.assertEqual(ka, kb)
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-12, atol=1e-14)
        self.assertIsNone(candidate.v9c_geometry.pre_gs)
        self.assertFalse(candidate.v9c_geometry.capture_enabled)

    def test_fixed_shell_depth_preserves_direction_ignores_source_radius(self):
        direction = F.normalize(torch.randn(8, 13, dtype=torch.float64), dim=-1)
        a = direction * torch.linspace(.02, 1.96, 8, dtype=torch.float64)[:, None]
        b = direction * torch.linspace(1.91, .03, 8, dtype=torch.float64)[:, None]
        aa, bb = g.whole_to_hier(a), g.whole_to_hier(b)
        torch.testing.assert_close(aa, bb, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(F.normalize(aa, dim=-1), direction, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(g.hier_depth(aa), torch.full((8,), 4.6, dtype=torch.float64))
        g.assert_hier_shell(aa)
        self.assertAlmostEqual(g.HIER_RADIUS, 1.965119614299547, places=10)

    def test_shell_exact_angular_jacobian_one_gs_and_zero_radial_component(self):
        x = torch.tensor([[.2, -.35, .4], [.1, .2, -.6]], dtype=torch.float64, requires_grad=True)
        probe = torch.tensor([[.8, -.1, .4], [-.5, .9, .2]], dtype=torch.float64)
        y = g.whole_to_hier(x)
        gradient, = torch.autograd.grad((y * probe).sum(), x)
        norm = x.detach().norm(dim=-1, keepdim=True)
        direction = x.detach() / norm
        expected = g.HIER_SHELL_GS * g.HIER_RADIUS / norm * (probe - direction * (direction * probe).sum(-1, keepdim=True))
        torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((gradient * direction).sum(-1), torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-13)
        self.assertEqual(gs_nodes(y), 1)
        self.assertGreater(float(gradient.norm()), .01)

    def test_zero_direction_nonfinite_wrong_dtype_and_domain_rejected(self):
        for bad in (torch.zeros(1, 3), torch.tensor([[1e-9, 0., 0.]]), torch.tensor([[3.2, 0., 0.]]),
                    torch.tensor([[float('nan'), .1, .2]]), torch.empty(0, 3), torch.ones(3)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                g.whole_to_hier(bad)
        with self.assertRaises(TypeError):
            g.whole_to_hier(torch.ones(2, 3, dtype=torch.float16))

    def test_repeated_configuration_or_existing_hooks_rejected(self):
        model = g.configure_backbone(ProbeBackbone())
        with self.assertRaises(RuntimeError):
            g.configure_backbone(model)
        model = ProbeBackbone()
        model.emb.register_forward_hook(lambda module, inputs, value: value)
        with self.assertRaises(RuntimeError):
            g.configure_backbone(model)

    def test_c01_geometry_reports_caps_and_branch_domain(self):
        p = F.normalize(torch.randn(5, 256), dim=-1) * .7
        native = g.native_geometry_stats(p)
        shell = g.hier_geometry_stats(g.whole_to_hier(p))
        self.assertEqual(native['c'], .1)
        self.assertTrue(native['inverse_metric_hook'])
        self.assertTrue(native['extra_cap'])
        self.assertAlmostEqual(shell['depth_mean'], 4.6, places=5)
        with self.assertRaises(RuntimeError):
            g.assert_hier_shell(p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
