"""CPU contracts for V9's native and HIER-only geometry separation."""
from __future__ import annotations

import copy
import math
import unittest

import geoopt
import torch
from torch import nn
import torch.nn.functional as F

from ..geometry import (
    NATIVE_C, HIER_VIEW_C, HIER_DEPTH, HIER_RADIUS, HIER_SHELL_GS,
    MIN_DIRECTION_NORM, InverseMetric, official_distance,
    configure_backbone, freeze_curvature, whole_to_hier,
    native_depth, hier_depth, assert_hier_shell,
    native_geometry_stats, hier_geometry_stats,
)


class ShellGeometryContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        torch.set_num_threads(2)

    def test_shell_depth_direction_and_native_input_unchanged(self):
        direction = F.normalize(torch.randn(8, 13, dtype=torch.float64), dim=-1)
        radius = torch.linspace(.02, .99, 8, dtype=torch.float64)[:, None]
        native = (direction * radius).requires_grad_()
        before = native.detach().clone()
        result = whole_to_hier(native)
        self.assertTrue(torch.equal(before, native.detach()))
        torch.testing.assert_close(F.normalize(result, dim=-1), direction, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(result.norm(dim=-1), torch.full((8,), HIER_RADIUS, dtype=torch.float64))
        torch.testing.assert_close(hier_depth(result), torch.full((8,), HIER_DEPTH, dtype=torch.float64))
        assert_hier_shell(result)
        self.assertLess(float(native_depth(native)[0]), 4.6)
        self.assertGreater(float(native_depth(native)[-1]), 4.6)
        self.assertTrue(result.requires_grad)

    def test_shell_ignores_native_radius_and_proxy_ranking_is_angular(self):
        angles = torch.tensor([-.92, -.61, -.17, .09, .47, .83, 1.31], dtype=torch.float64)
        directions = torch.stack((angles.cos(), angles.sin(), .1 * angles), dim=-1)
        directions = F.normalize(directions, dim=-1)
        a = whole_to_hier(directions * torch.linspace(.05, .95, 7, dtype=torch.float64)[:, None])
        b = whole_to_hier(directions * torch.linspace(.91, .07, 7, dtype=torch.float64)[:, None])
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)
        proxy_direction = F.normalize(torch.tensor([[1., .2, .03]], dtype=torch.float64), dim=-1)
        proxies = proxy_direction * torch.tensor([[.3], [1.2], [1.95]], dtype=torch.float64)
        distances = official_distance(a, proxies, c=HIER_VIEW_C)
        order = distances.argsort(dim=0)
        cosine_order = (directions @ proxy_direction.T).squeeze(-1).argsort(descending=True)
        for col in range(3):
            self.assertTrue(torch.equal(order[:, col], cosine_order))
        torch.testing.assert_close(official_distance(a, proxies, c=HIER_VIEW_C),
                                   official_distance(b, proxies, c=HIER_VIEW_C), rtol=1e-12, atol=1e-12)

    def test_gradient_is_angular_has_exact_jacobian_and_one_gs(self):
        x = torch.tensor([[.2, -.35, .4], [.1, .2, -.6]], dtype=torch.float64, requires_grad=True)
        probe = torch.tensor([[.8, -.1, .4], [-.5, .9, .2]], dtype=torch.float64)
        result = whole_to_hier(x)
        gradient, = torch.autograd.grad((result * probe).sum(), x)
        radius = x.detach().norm(dim=-1, keepdim=True)
        direction = x.detach() / radius
        expected = HIER_SHELL_GS * HIER_RADIUS / radius * (
            probe - direction * (direction * probe).sum(-1, keepdim=True))
        torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((gradient * direction).sum(-1), torch.zeros(2, dtype=torch.float64),
                                   rtol=0, atol=1e-13)
        self.assertGreater(float(gradient.norm()), 1e-3)

    def test_hier_radial_function_has_no_native_radial_gradient(self):
        x = torch.tensor([[.2, .1, .3]], dtype=torch.float64, requires_grad=True)
        loss = whole_to_hier(x).square().sum()
        gradient, = torch.autograd.grad(loss, x)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-13)

    def test_shell_gs_factor_is_constant(self):
        self.assertAlmostEqual(HIER_SHELL_GS, .094197, places=6)
        self.assertAlmostEqual(HIER_RADIUS, 1.9651, places=4)
        points = whole_to_hier(torch.randn(11, 9, dtype=torch.float64) * .03).detach().requires_grad_()
        gradient, = torch.autograd.grad(InverseMetric.apply(points, HIER_VIEW_C).sum(), points)
        torch.testing.assert_close(gradient, torch.full_like(points, HIER_SHELL_GS), rtol=1e-13, atol=1e-13)

    def test_invalid_dtype_nonfinite_origin_and_wrong_native_domain_rejected(self):
        for dtype in (torch.float16, torch.bfloat16, torch.int64):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                whole_to_hier(torch.ones(2, 3, dtype=dtype))
        for bad in (torch.zeros(1, 3), torch.tensor([[MIN_DIRECTION_NORM, 0., 0.]]),
                    torch.tensor([[1e-10, 0., 0.]]), torch.tensor([[float('nan'), .1, .2]]),
                    torch.tensor([[float('inf'), .1, .2]]), torch.tensor([[1.1, 0., 0.]])):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                whole_to_hier(bad)
        with self.assertRaises(ValueError):
            whole_to_hier(torch.empty(0, 3))
        with self.assertRaises(ValueError):
            whole_to_hier(torch.ones(3))

    def test_fp32_shell_and_explicit_geometry_statistics(self):
        source = F.normalize(torch.randn(5, 256), dim=-1) * .7
        result = whole_to_hier(source)
        assert_hier_shell(result)
        torch.testing.assert_close(hier_depth(result), torch.full((5,), HIER_DEPTH), rtol=1e-6, atol=1e-6)
        native = native_geometry_stats(source)
        view = hier_geometry_stats(result)
        self.assertEqual(native['c'], NATIVE_C)
        self.assertFalse(native['extra_cap'])
        self.assertFalse(native['inverse_metric_hook'])
        self.assertEqual(view['c'], HIER_VIEW_C)
        self.assertTrue(view['inverse_metric_hook'])
        self.assertAlmostEqual(view['depth_mean'], HIER_DEPTH, places=5)
        with self.assertRaises(RuntimeError):
            assert_hier_shell(result * .9)


class NativeBackboneGeometryContract(unittest.TestCase):
    @staticmethod
    def model():
        class NativeLayer(nn.Module):
            def __init__(self, manifold, weight):
                super().__init__()
                self.manifold = manifold
                self.weight = nn.Parameter(weight)

            def forward(self, points):
                return self.manifold.expmap0(F.linear(points, self.weight))

        class NativeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.manifold = geoopt.PoincareBall(c=1.)
                self.manifold2 = geoopt.PoincareBall(c=1.)
                self.emb = NativeLayer(self.manifold, torch.eye(3) * 3)
                self.classifier = NativeLayer(self.manifold2, torch.eye(3) * .5)

            def forward(self, points):
                mu = self.emb(points)
                return mu, self.classifier(mu)

        return NativeBackbone().double()

    def test_native_setup_preserves_forward_weights_and_base_gradients(self):
        original = self.model()
        candidate = copy.deepcopy(original)
        state = {key: value.clone() for key, value in candidate.state_dict().items()}
        self.assertIs(configure_backbone(candidate), candidate)
        x = torch.tensor([[.8, .2, .4], [-.5, .6, .4]], dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        mu0, logits0 = original(x)
        mu1, logits1 = candidate(y)
        torch.testing.assert_close(mu0, mu1, rtol=0, atol=0)
        torch.testing.assert_close(logits0, logits1, rtol=0, atol=0)
        self.assertGreater(float(native_depth(mu1).max()), HIER_DEPTH)
        loss0 = logits0.square().sum() + mu0.square().sum()
        loss1 = logits1.square().sum() + mu1.square().sum()
        loss0.backward()
        loss1.backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-14)
        for (name0, parameter0), (name1, parameter1) in zip(original.named_parameters(), candidate.named_parameters()):
            self.assertEqual(name0, name1)
            if parameter0.grad is not None:
                torch.testing.assert_close(parameter0.grad, parameter1.grad, rtol=0, atol=0)
        for key, value in candidate.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        self.assertEqual(len(candidate.emb._forward_hooks), 0)
        self.assertEqual(len(candidate.classifier._forward_hooks), 0)
        self.assertFalse(candidate.manifold.isp_c.requires_grad)
        self.assertFalse(candidate.manifold2.isp_c.requires_grad)

    def test_wrong_curvature_or_existing_output_hook_rejected(self):
        wrong = self.model()
        wrong.manifold2 = geoopt.PoincareBall(c=.1)
        with self.assertRaises(RuntimeError):
            configure_backbone(wrong)
        wrong = self.model()
        wrong.emb.register_forward_hook(lambda module, inputs, output: output)
        with self.assertRaises(RuntimeError):
            configure_backbone(wrong)
        wrong = self.model()
        wrong.v8_geometry = nn.Identity()
        with self.assertRaises(RuntimeError):
            configure_backbone(wrong)


if __name__ == '__main__':
    unittest.main()