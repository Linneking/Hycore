"""CPU contracts for the retained Conv9 / two-Mobius-tail experiment.

Installed: python -m segmentation_ShapeNetPart.hycore_seg_v4.test_contracts
Staging: PYTHONPATH=<repository> python <staging>/test_contracts.py
No real point clouds, checkpoints, or CUDA are accessed.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v1.vendor.dgcnn_model import DGCNN_partseg
from segmentation_ShapeNetPart.hycore_seg_v2.native import MobiusLayer
from segmentation_ShapeNetPart.hycore_seg_v3.model import NativePseg3
from segmentation_ShapeNetPart.hycore_seg_v3.losses import HyCoReSegLoss
from segmentation_ShapeNetPart.hycore_seg_v3.monitoring import GeometryMonitor

if __package__:
    from .model import NativeTail2
else:
    spec = importlib.util.spec_from_file_location(
        "pseg4_staged_model", Path(__file__).with_name("model.py"))
    staged_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged_model)
    NativeTail2 = staged_model.NativeTail2


def fresh(seed=4780):
    torch.manual_seed(seed)
    return NativeTail2("d", k=4)


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_actual_topology_keeps_conv9_and_dp1_but_only_two_m_maps(self):
        model = fresh()
        self.assertEqual(model.arm, "d")
        self.assertEqual(model.architecture, "pseg3_tail2_v4_d")
        self.assertEqual(model.feature_dim, 256)
        self.assertTrue(model.uses_category)
        self.assertFalse(model.has_m1)
        self.assertTrue(model.has_euclidean_conv9)
        self.assertEqual(model.regularizer_feature_stage, "after_conv9_bn9_leaky")
        self.assertIsInstance(model.dp1, nn.Dropout)
        self.assertEqual(model.dp1.p, .5)
        self.assertTrue(model.dropout_active)
        self.assertEqual((model.conv9[0].in_channels, model.conv9[0].out_channels), (256, 256))
        self.assertIsNone(model.conv9[0].bias)
        self.assertIs(model.conv9[1], model.bn9)
        self.assertIsInstance(model.conv9[2], nn.LeakyReLU)
        self.assertEqual(model.conv9[2].negative_slope, .2)
        self.assertEqual(sum(isinstance(m, MobiusLayer) for m in model.modules()), 2)
        self.assertEqual((model.emb2.in_features, model.emb2.out_features), (256, 128))
        self.assertEqual((model.mlr.in_features, model.mlr.out_features), (128, 50))
        for name in ("emb", "conv10", "conv11", "bn10", "dp2"):
            self.assertFalse(hasattr(model, name), name)
            self.assertFalse(any(k.startswith(name + ".") for k in model.state_dict()), name)
        head_ids = {id(p) for p in model.head_parameters()}
        self.assertEqual(head_ids, {id(p) for layer in (model.emb2, model.mlr) for p in layer.parameters()
                                  if p.requires_grad})
        self.assertNotIn(id(model.conv9[0].weight), head_ids)

    def test_original_conv9_and_all_surviving_b_initial_values_are_exact(self):
        model = fresh()
        torch.manual_seed(4780)
        baseline = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50)
        torch.manual_seed(4780)
        b = NativePseg3("b", k=4)
        ds, bs, originals = model.state_dict(), b.state_dict(), baseline.state_dict()
        self.assertEqual(set(ds) - set(bs), {k for k in ds if k.startswith(("conv9.", "bn9."))})
        self.assertTrue(any(k.startswith("emb2.") for k in bs))
        self.assertTrue(any(k.startswith("mlr.") for k in bs))
        for name, value in bs.items():
            self.assert_exact(ds[name], value)
        for name, value in originals.items():
            if name in ds:
                self.assert_exact(ds[name], value)
        self.assertEqual(count(model) - count(b), 66048)
        self.assertEqual(count(baseline) - count(model), 78)
        self.assertEqual(count(model), 1460283)

    def test_fresh_constructor_is_reproducible_without_shifting_cpu_rng(self):
        first = fresh()
        after = torch.rand(17)
        second = fresh()
        self.assert_exact(torch.rand(17), after)
        for name, value in first.state_dict().items():
            self.assert_exact(second.state_dict()[name], value)
        torch.manual_seed(4780)
        NativePseg3("b", k=4)
        self.assert_exact(torch.rand(17), after)

    def test_frontend_through_conv9_matches_original_in_train_and_eval(self):
        generator = torch.Generator().manual_seed(811)
        points = torch.randn(3, 3, 24, generator=generator)
        labels = torch.eye(16)[:3]
        for training in (False, True):
            with self.subTest(training=training):
                model = fresh().train(training)
                torch.manual_seed(4780)
                reference = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50).train(training)
                captured = []
                handle = reference.conv9.register_forward_hook(lambda m, a, out: captured.append(out.detach().clone()))
                torch.manual_seed(113)
                with torch.no_grad():
                    reference(points, labels)
                handle.remove()
                torch.manual_seed(113)
                with torch.no_grad():
                    actual = model.forward_features(points, labels)
                self.assertEqual(actual.shape, (3, 24, 256))
                self.assert_exact(actual.transpose(1, 2), captured[0])

    def test_training_dropout_is_before_retained_conv9_and_h_is_after_it(self):
        model = fresh().train()
        monitor = GeometryMonitor()
        monitor.begin(True, epoch=1, step=1)
        model.monitor = monitor
        dp_results, conv9_inputs, conv9_results = [], [], []
        handles = [
            model.dp1.register_forward_hook(lambda m, a, out: dp_results.append(out)),
            model.conv9.register_forward_pre_hook(lambda m, a: conv9_inputs.append(a[0])),
            model.conv9.register_forward_hook(lambda m, a, out: conv9_results.append(out)),
        ]
        points, labels = torch.randn(3, 3, 24), torch.eye(16)[:3]
        _, h = model(points, labels, return_features=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(dp_results), 1)
        self.assertGreater(float((dp_results[0] == 0).float().mean()), .35)
        self.assert_exact(conv9_inputs[0], dp_results[0])
        self.assert_exact(monitor.nodes["pre_conv9"].transpose(1, 2), dp_results[0])
        self.assert_exact(h.transpose(1, 2), conv9_results[0])
        self.assertIs(monitor.nodes["h"], h)
        self.assertNotIn("m1", monitor.nodes)
        self.assertTrue({"m2", "m3", "point_exp"}.issubset(monitor.nodes))

    def test_one_forward_updates_all_fourteen_bn_counters_once(self):
        model = fresh().train()
        logits, h = model(torch.randn(3, 3, 24), torch.eye(16)[:3], return_features=True)
        self.assertEqual(logits.shape, (3, 50, 24))
        self.assertEqual(h.shape, (3, 24, 256))
        self.assertTrue(torch.isfinite(logits).all())
        counters = [int(m.num_batches_tracked) for m in model.modules()
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        self.assertEqual(len(counters), 14)
        self.assertEqual(counters, [1] * 14)

    def test_ce_and_regularizer_route_through_conv9_but_regularizer_bypasses_m(self):
        model = fresh().train()
        monitor = GeometryMonitor()
        monitor.begin(True, epoch=1, step=1)
        model.monitor = monitor
        criterion = HyCoReSegLoss(part_min=4, part_max=8, seed=4780, feature_dim=256)
        criterion.monitor = monitor
        categories = torch.tensor([0, 1, 2])
        logits, h = model(torch.randn(3, 3, 24), F.one_hot(categories, 16).float(), return_features=True)
        regularizer, diagnostics = criterion(h, categories)
        ce = F.cross_entropy(logits.transpose(1, 2).reshape(-1, 50),
                             torch.tensor([0, 4, 6]).repeat_interleave(24))
        self.assert_exact(monitor.nodes["whole_tangent"], h.mean(dim=1))
        self.assertEqual(diagnostics["negative_legal_rate"], 1.)
        shared = [model.conv9[0].weight, model.conv8[0].weight]
        heads = [p for p in model.head_parameters()]
        gradients = torch.autograd.grad(regularizer, (*shared, *heads), retain_graph=True, allow_unused=True)
        for gradient in gradients[:len(shared)]:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.)
        self.assertTrue(all(g is None for g in gradients[len(shared):]))
        ce_gradients = torch.autograd.grad(ce, (*shared, *heads), retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in ce_gradients))
        self.assertTrue(all(float(g.norm()) > 0. for g in ce_gradients))
        # A pure radial gradient may legitimately be tiny after ball projection.
        radial_grad, = torch.autograd.grad(criterion.last_components["hier"],
                                           (model.conv9[0].weight,), retain_graph=True)
        self.assertTrue(torch.isfinite(radial_grad).all())
        before = [m.num_batches_tracked.clone() for m in model.modules()
                  if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        (ce + regularizer).backward()
        after = [m.num_batches_tracked for m in model.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        for actual, expected in zip(after, before):
            self.assert_exact(actual, expected)

    def test_head_forward_is_the_exact_surviving_official_two_map_composition(self):
        model = fresh().eval()
        h = torch.randn(3, 11, 256) * .02
        with torch.no_grad():
            expected = model.mlr(model.emb2(model.manifold.expmap0(h)))
            actual = model.forward_head(h)
        self.assert_exact(actual, expected)
        self.assertLessEqual(float(actual.norm(dim=-1).max()), .996001)

    def test_invalid_arm_is_rejected_before_constructing_modules(self):
        torch.manual_seed(4780)
        before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            NativeTail2("b", k=4)
        self.assert_exact(torch.get_rng_state(), before)

    def test_cpu_contracts_do_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
