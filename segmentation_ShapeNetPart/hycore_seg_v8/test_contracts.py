"""Focused CPU contracts for Experiment I's H-compatible two-forward design."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v2.losses import choose_parts
from segmentation_ShapeNetPart.hycore_seg_v2.native import MobiusLayer
from segmentation_ShapeNetPart.hycore_seg_v3.monitoring import GeometryMonitor
from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2

from .losses import HyCoReSegLoss
from .model import NativeI


def fresh(cls=NativeI):
    torch.manual_seed(4780)
    return NativeI(k=4) if cls is NativeI else NativeTail2("h", k=4)


def fixture():
    generator = torch.Generator().manual_seed(811)
    points = torch.randn(3, 3, 24, generator=generator) * .15
    labels = torch.eye(16)[:3]
    categories = torch.arange(3)
    targets = torch.tensor([0, 4, 6]).repeat_interleave(24)
    return points, labels, categories, targets


def bn_counts(model):
    return {name: int(module.num_batches_tracked) for name, module in model.named_modules()
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))}


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)

    def test_h_initial_state_rng_and_topology_unchanged(self):
        h = fresh(NativeTail2)
        h_rng = torch.get_rng_state().clone()
        model = fresh()
        torch.testing.assert_close(torch.get_rng_state(), h_rng, rtol=0, atol=0)
        self.assert_exact_state(model.state_dict(), h.state_dict())
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 1460283)
        self.assertIsInstance(model.dp1, nn.Identity)
        self.assertEqual(sum(isinstance(module, MobiusLayer) for module in model.modules()), 2)
        self.assertEqual((model.emb2.in_features, model.emb2.out_features), (256, 128))
        self.assertEqual((model.mlr.in_features, model.mlr.out_features), (128, 50))
        self.assertEqual(model.feature_dim, 128)
        self.assertEqual(model.euclidean_feature_dim, 256)

    def test_whole_main_forward_is_exactly_h(self):
        points, labels, _, _ = fixture()
        h = fresh(NativeTail2).train()
        model = fresh().train()
        torch.manual_seed(113)
        h_logits, h_features = h(points, labels, return_features=True)
        h_rng = torch.get_rng_state().clone()
        torch.manual_seed(113)
        logits, features = model(points, labels, return_features=True)
        torch.testing.assert_close(logits, h_logits, rtol=0, atol=0)
        torch.testing.assert_close(features, h_features, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), h_rng, rtol=0, atol=0)
        self.assert_exact_state(model.state_dict(), h.state_dict())
        self.assertEqual(len(bn_counts(model)), 14)
        self.assertEqual(set(bn_counts(model).values()), {1})

    def test_part_selection_matches_h_and_two_train_forwards_update_bn_twice(self):
        points, labels, _, _ = fixture()
        model = fresh().train()
        criterion = HyCoReSegLoss(part_min=4, part_max=8)
        _, features = model(points, labels, return_features=True)
        state = criterion.generator.get_state().clone()
        indices, centers, size = criterion.select_parts(features)
        reference_generator = torch.Generator().set_state(state)
        _, expected_indices, expected_centers, expected_size = choose_parts(
            features, reference_generator, 4, 8)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(centers, expected_centers, rtol=0, atol=0)
        self.assertEqual(size, expected_size)
        part_xyz = points.transpose(1, 2).gather(
            1, indices[:, :, None].expand(-1, -1, 3)).transpose(1, 2).contiguous()
        part_features = model.encode_part(part_xyz, labels)
        self.assertEqual(part_features.shape, (3, size, 256))
        self.assertEqual(set(bn_counts(model).values()), {2})
        self.assertTrue(model.training)
        self.assertTrue(all(module.training for module in model.modules()
                            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))))

    def test_regularizers_reach_shared_first_m_but_not_final_mlr(self):
        points, labels, categories, targets = fixture()
        model = fresh().train()
        monitor = GeometryMonitor().begin(True)
        model.monitor = monitor
        criterion = HyCoReSegLoss(part_min=4, part_max=8)
        logits, full = model(points, labels, return_features=True)
        indices, centers, size = criterion.select_parts(full)
        part_xyz = points.transpose(1, 2).gather(
            1, indices[:, :, None].expand(-1, -1, 3)).transpose(1, 2).contiguous()
        part = model.encode_part(part_xyz, labels)
        whole_mu = model.regularizer_embedding(full, "whole")
        part_mu = model.regularizer_embedding(part, "part")
        reg, diagnostics = criterion(whole_mu, part_mu, categories, size, indices, centers, full, part)
        self.assertEqual(whole_mu.shape, (3, 128))
        self.assertEqual(part_mu.shape, (3, 128))
        self.assertEqual(diagnostics["negative_legal_rate"], 1.)
        self.assertEqual(diagnostics["part_points"], size)
        self.assertEqual(diagnostics["feature_dim"], 128)
        self.assertAlmostEqual(diagnostics["gamma"] * size, 1000.)
        self.assertEqual(set(bn_counts(model).values()), {2})
        self.assertTrue({"whole_post_conv9", "part_post_conv9", "whole_mu", "part_mu"}.issubset(monitor.nodes))
        parameters = [model.conv1[0].weight, model.conv9[0].weight,
                      model.emb2._weight, model.emb2._bias,
                      model.mlr._weight, model.mlr._bias]
        for value in (reg, *criterion.last_components.values()):
            grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in grads[:4]))
            self.assertTrue(all(grad is None for grad in grads[4:]))
        contrast_grads = torch.autograd.grad(
            criterion.last_components["contr"], parameters[:4], retain_graph=True)
        self.assertTrue(all(float(grad.norm()) > 0. for grad in contrast_grads))
        ce = F.cross_entropy(logits.transpose(1, 2).reshape(-1, 50), targets)
        ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
        self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in ce_grads))
        (ce + reg).backward()
        self.assertEqual(set(bn_counts(model).values()), {2})

    def test_eval_uses_only_full_segmentation_forward(self):
        points, labels, _, _ = fixture()
        model = fresh().eval()
        before = bn_counts(model)
        with torch.no_grad():
            logits = model(points, labels)
        self.assertEqual(logits.shape, (3, 50, 24))
        self.assertEqual(before, bn_counts(model))

    def test_invalid_regularizer_width_and_coefficients_rejected(self):
        for settings in ({"feature_dim": 256}, {"alpha": 0.}, {"beta": 0.},
                         {"alpha": .02}, {"part_min": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                HyCoReSegLoss(**settings)


if __name__ == "__main__":
    unittest.main()
