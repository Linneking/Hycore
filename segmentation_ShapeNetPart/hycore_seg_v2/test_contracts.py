"""CPU contracts for native component reuse and segmentation gradient routing."""

import ast
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geoopt
import torch
from torch.nn import functional as F

from ..hycore_seg_v1.utils import restore_rng, rng_state
from ..hycore_seg_v1.vendor.dgcnn_model import DGCNN_partseg
from .losses import (HyCoReSegLoss, choose_parts, flip_negative_indices,
                     relation_losses)
from .model import NativePartSeg
from .native import (MobiusLayer, PoincareBall, cal_loss, get_graph_feature,
                     hype_triplet_losses, verify_provenance)


def original_classification_reference(args):
    """Test-only exact original class, bypassing unrelated package imports.

    Production uses ordinary classes in model/native. The fixture compiles the
    unmodified original DGCNN class and injects only its graph-device adapter.
    """
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "classification_ModelNet40").is_dir())
    path = root / "classification_ModelNet40/models/dgcnn_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DGCNN")
    fixture = ast.Module(body=[definition], type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn, "F": F, "get_graph_feature": get_graph_feature}
    exec(compile(fixture, str(path), "exec"), namespace)
    return namespace["DGCNN"](args, 40)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_original_source_provenance(self):
        manifest = verify_provenance()
        self.assertGreaterEqual(len(manifest["files"]), 5)

    def test_graph_extraction_stays_inside_each_cloud(self):
        points = torch.tensor([[[0., 1., 2.], [0., 0., 0.]],
                               [[100., 101., 102.], [100., 100., 100.]]])
        graph = get_graph_feature(points, k=2)
        self.assertEqual(graph.shape, (2, 4, 3, 2))
        # Translation alters only the repeated centers, not neighbor offsets.
        torch.testing.assert_close(graph[0, :2], graph[1, :2], rtol=0, atol=0)
        torch.testing.assert_close(graph[1, 2:] - graph[0, 2:],
                                   torch.full_like(graph[0, 2:], 100.), rtol=0, atol=0)

    def test_classification_encoder_reuses_original_numerics(self):
        torch.manual_seed(4780)
        reference = original_classification_reference(SimpleNamespace(k=4, emb_dims=1024, dropout=.5)).eval()
        model = NativePartSeg("p0", k=4).eval()
        subset = {name: value for name, value in reference.state_dict().items()
                  if name.startswith(("conv1.", "conv2.", "conv3.", "conv4.", "conv5.",
                                      "bn1.", "bn2.", "bn3.", "bn4.", "bn5."))}
        model.load_state_dict(subset, strict=False)
        captured = []
        handle = reference.conv5.register_forward_hook(lambda module, args, result: captured.append(result))
        points = torch.randn(3, 3, 24)
        with torch.no_grad():
            reference(points)
            actual = model.forward_encoder(points)
        handle.remove()
        torch.testing.assert_close(actual, captured[0], rtol=0, atol=0)

    def test_third_arm_preserves_category_broadcast_and_original_frontend(self):
        torch.manual_seed(4780)
        reference = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50).eval()
        torch.manual_seed(4780)
        model = NativePartSeg("pseg3", k=4).eval()
        for name, value in model.state_dict().items():
            if name in reference.state_dict():
                torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        captured = []
        handle = reference.dp1.register_forward_hook(lambda module, args, result: captured.append(result))
        points, labels = torch.randn(3, 3, 24), torch.eye(16)[:3]
        with torch.no_grad():
            reference(points, labels)
            features = model.forward_features(points, labels)
        handle.remove()
        self.assertEqual(model.conv8[0].in_channels, 1280)
        self.assertEqual(model.conv7[0].in_channels, 16)
        self.assertEqual(model.conv7[0].out_channels, 64)
        torch.testing.assert_close(features.transpose(1, 2), captured[0], rtol=0, atol=0)

    def test_three_arm_shapes_bn_updates_and_category_policy(self):
        for architecture, width in (("p0", 512), ("pglobal", 512), ("pseg3", 256)):
            with self.subTest(architecture=architecture):
                torch.manual_seed(4780)
                model = NativePartSeg(architecture, k=4).train()
                points, labels = torch.randn(3, 3, 24), torch.eye(16)[:3]
                logits, features = model(points, labels, return_features=True)
                self.assertEqual(logits.shape, (3, 50, 24))
                self.assertEqual(features.shape, (3, 24, width))
                counters = [int(m.num_batches_tracked) for m in model.modules()
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
                self.assertTrue(counters and all(n == 1 for n in counters))
                if architecture != "pseg3":
                    self.assertFalse(model.uses_category)
                    model.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(model(points, labels), model(points, labels.flip(0)), rtol=0, atol=0)
                else:
                    self.assertTrue(model.uses_category)

    def test_projection_head_widths_parameter_counts_and_seed(self):
        counts = {}
        for architecture in ("p0", "pglobal", "pseg3"):
            torch.manual_seed(4780)
            first = NativePartSeg(architecture, k=4)
            torch.manual_seed(4780)
            second = NativePartSeg(architecture, k=4)
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
            head_count = sum(p.numel() for p in first.head_parameters())
            self.assertEqual(head_count, 105138 if architecture == "pseg3" else 144178)
            if architecture == "p0":
                self.assertEqual(first.linear1.in_features, 1024)
            elif architecture == "pglobal":
                self.assertEqual(first.linear1.in_features, 2048)
            counts[architecture] = sum(p.numel() for p in first.parameters())
        self.assertEqual(counts["pglobal"] - counts["p0"], 524288)

    def test_complete_mobius_parameters_forward_and_gradients(self):
        torch.manual_seed(4780)
        ball = PoincareBall(c=1., dim=8).double()
        layer = MobiusLayer(8, 5, ball).double()
        tangent = (torch.randn(2, 7, 8, dtype=torch.float64) * .03).requires_grad_()
        value = ball.expmap0(tangent)
        expected_bias = ball.expmap0(layer._weight * layer._bias)
        expected_weight = ball.transp0(expected_bias, layer._weight)
        expected = ball.mobius_matvec(expected_weight, value)
        actual = layer(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.reshape(-1, 5), layer(value.reshape(-1, 8)), rtol=1e-12, atol=1e-12)
        actual.square().sum().backward()
        for gradient in (tangent.grad, layer._weight.grad, layer._bias.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.)
        self.assertLessEqual(float(layer._bias.detach().abs().max()), 4 / math.sqrt(8))

    def test_head_pointwise_flattening_equivalence(self):
        for architecture in ("p0", "pglobal", "pseg3"):
            with self.subTest(architecture=architecture):
                model = NativePartSeg(architecture, k=4).eval()
                features = torch.randn(2, 7, model.feature_dim) * .02
                with torch.no_grad():
                    expected = model.forward_head(features.reshape(-1, model.feature_dim)).reshape(2, 7, 50)
                    actual = model.forward_head(features)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_original_geometry_matches_analytic_ball_distance(self):
        ball = PoincareBall(c=1., dim=2).double()
        x = torch.tensor([[.1, .2], [-.2, .3]], dtype=torch.float64)
        y = torch.tensor([[.25, -.1], [.1, .2]], dtype=torch.float64)
        # Geoopt stores inverse-softplus(c). A FP32-created ball cast to FP64
        # can have an actual c slightly different from exactly 1; use that c
        # in the independent analytic formula instead of changing tolerance.
        curvature = ball.c.detach()
        sqrt_c = curvature.sqrt()
        expected = torch.acosh(1 + 2 * curvature * (x - y).square().sum(-1) /
                               ((1 - curvature * x.square().sum(-1)) *
                                (1 - curvature * y.square().sum(-1)))) / sqrt_c
        torch.testing.assert_close(ball.dist(x, y), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ball.dist0(x), 2 * torch.atanh(sqrt_c * x.norm(dim=-1)) / sqrt_c,
                                   rtol=1e-12, atol=1e-12)

    def test_feature_neighborhood_pooling_gradients_and_ties(self):
        values = torch.tensor([[[0.], [1.], [4.], [10.], [11.]],
                               [[-3.], [-1.], [2.], [3.], [8.]]], requires_grad=True)
        original = values.detach().clone()
        pooled, indices, centers, size = choose_parts(values, torch.Generator().manual_seed(9), 2, 2)
        for row in range(2):
            distances = (values[row, :, 0].detach() - values[row, centers[row], 0].detach()).square()
            distances[centers[row]] = -1
            torch.testing.assert_close(indices[row], distances.argsort(stable=True)[:2])
        pooled.sum().backward()
        expected = torch.zeros_like(values)
        expected.scatter_(1, indices[:, :, None], .5)
        torch.testing.assert_close(values.grad, expected)
        torch.testing.assert_close(values, original)
        self.assertEqual(size, 2)
        _, tied, centers, _ = choose_parts(torch.zeros(3, 8, 2), torch.Generator().manual_seed(4780), 3, 3)
        self.assertTrue((tied[:, 0] == centers).all())

    def test_batch_flip_keeps_same_class_pairs_and_reports_self_pairs(self):
        labels = torch.tensor([0, 1, 1, 0])
        self.assertEqual(flip_negative_indices(labels).tolist(), [3, 2, 1, 0])
        criterion = HyCoReSegLoss(part_min=2, part_max=2)
        _, info = criterion(torch.randn(4, 8, 3) * .05, labels)
        self.assertEqual(info["valid_negative"], 4)
        self.assertEqual(info["same_class_pair_fraction"], 1.)
        self.assertEqual(info["self_pair_fraction"], 0.)
        _, odd = criterion(torch.randn(3, 8, 3) * .05, torch.tensor([0, 1, 2]))
        self.assertAlmostEqual(odd["self_pair_fraction"], 1 / 3, places=6)

    def test_original_loss_kernel_scalar_and_input_gradient_regression(self):
        # Match the source kernel's freshly instantiated FP32 curvature
        # parameter while retaining independent FP64 input leaves below.
        ball = PoincareBall(c=1., dim=3)
        whole = torch.tensor([[.1, .02, .01], [.03, -.15, .02], [-.06, .04, .02]],
                             dtype=torch.float64, requires_grad=True)
        part = torch.tensor([[.2, .03, .02], [-.03, -.1, .03], [-.08, .1, .01]],
                            dtype=torch.float64, requires_grad=True)
        gamma, delta = .17, .31
        values = hype_triplet_losses(whole, part, gamma, delta, ball_dim=3, opposite_hier=True)
        radial_values = ball.dist0(whole) - ball.dist0(part) + gamma
        contrast_values = ball.dist(whole, part) - ball.dist(whole, part.flip(0)) + delta
        expected = .01 * torch.maximum(radial_values, torch.zeros_like(radial_values)).mean()
        expected = expected + .01 * torch.maximum(contrast_values, torch.zeros_like(contrast_values)).mean()
        actual = .01 * values[4] + .01 * values[5]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual_grads = torch.autograd.grad(actual, (whole, part), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (whole, part))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_segmentation_radial_direction_whole_shallow_part_deep(self):
        ball = PoincareBall(c=1., dim=2).double()
        whole = torch.tensor([[.2, 0.]], dtype=torch.float64, requires_grad=True)
        part = torch.tensor([[.1, 0.]], dtype=torch.float64, requires_grad=True)
        radial, _, _ = relation_losses(ball, whole, part, gamma=2., delta=4.)
        radial.backward()
        self.assertGreater(float((whole.grad * whole.detach()).sum()), 0.)
        self.assertLess(float((part.grad * part.detach()).sum()), 0.)

    def test_per_object_mean_before_exp_and_only_two_regularizers(self):
        torch.manual_seed(4780)
        features = torch.randn(4, 10, 5) * .07
        criterion = HyCoReSegLoss(alpha=.03, beta=.07, part_min=3, part_max=3)
        initial_generator = criterion.generator.get_state()
        with patch(__package__ + ".losses.hype_triplet_losses", wraps=hype_triplet_losses) as kernel:
            value, info = criterion(features, torch.tensor([0, 1, 1, 0]))
        criterion.generator.set_state(initial_generator)
        pooled, _, _, _ = choose_parts(features, criterion.generator, 3, 3)
        whole, part = kernel.call_args.args
        self.assertEqual(whole.shape, (4, 5))
        self.assertEqual(part.shape, (4, 5))
        torch.testing.assert_close(whole, criterion.ball.expmap0(features.mean(1)), rtol=0, atol=0)
        torch.testing.assert_close(part, criterion.ball.expmap0(pooled), rtol=0, atol=0)
        self.assertAlmostEqual(float(value), .03 * info["contr"] + .07 * info["hier"], places=5)
        self.assertAlmostEqual(info["gamma"] * info["part_points"], 1000.)
        self.assertEqual(info["delta"], 4.)
        self.assertEqual(kernel.call_args.kwargs["ball_dim"], 5)
        self.assertTrue(kernel.call_args.kwargs["opposite_hier"])

    def test_independent_part_rng_and_exact_generator_restore(self):
        features = torch.randn(4, 16, 8) * .03
        criterion = HyCoReSegLoss(part_min=3, part_max=8)
        before_global = torch.random.get_rng_state().clone()
        before_part = criterion.generator.get_state().clone()
        first, first_info = criterion(features, torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(before_global, torch.random.get_rng_state(), rtol=0, atol=0)
        criterion.generator.set_state(before_part)
        again, again_info = criterion(features, torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        self.assertEqual(first_info, again_info)

    def test_disabled_regularizer_does_not_sample(self):
        criterion = HyCoReSegLoss(alpha=0, beta=0)
        before = criterion.generator.get_state().clone()
        with patch(__package__ + ".losses.choose_parts", side_effect=AssertionError("disabled loss sampled")):
            loss, info = criterion(torch.randn(2, 24, 8), torch.tensor([0, 1]))
        self.assertEqual(float(loss), 0.)
        self.assertTrue(info["disabled"])
        torch.testing.assert_close(before, criterion.generator.get_state(), rtol=0, atol=0)

    def test_regularizer_reaches_backbone_but_not_prediction_head(self):
        for architecture in ("p0", "pglobal", "pseg3"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(4780)
                model = NativePartSeg(architecture, k=4).eval()
                points, labels = torch.randn(4, 3, 24), torch.eye(16)[:4]
                logits, features = model(points, labels, return_features=True)
                criterion = HyCoReSegLoss(part_min=6, part_max=6, feature_dim=model.feature_dim)
                regularizer, _ = criterion(features, torch.arange(4))
                shared = model.conv1[0].weight
                head = tuple(model.head_parameters())
                gradients = torch.autograd.grad(regularizer, (shared,) + head,
                                                retain_graph=True, allow_unused=True)
                self.assertTrue(torch.isfinite(gradients[0]).all())
                self.assertGreater(float(gradients[0].norm()), 0.)
                self.assertTrue(all(g is None for g in gradients[1:]))
                targets = torch.randint(50, (4, 24))
                ce = cal_loss(logits.transpose(1, 2).reshape(-1, 50), targets.reshape(-1), smoothing=False)
                expected_ce = F.cross_entropy(logits, targets)
                torch.testing.assert_close(ce, expected_ce, rtol=1e-6, atol=1e-6)
                ce_gradients = torch.autograd.grad(ce, (shared,) + head, retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() for g in ce_gradients))
                self.assertTrue(all(float(g.norm()) > 0 for g in ce_gradients))
                (ce + regularizer).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_rsgd_resume_restores_model_optimizer_and_both_rng_streams(self):
        torch.manual_seed(4780)
        model = NativePartSeg("pseg3", k=4).train()
        loss_module = HyCoReSegLoss(part_min=4, part_max=8, feature_dim=256)
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=.0001)
        points, labels, targets = torch.randn(4, 3, 24), torch.eye(16)[:4], torch.randint(50, (4, 24))

        def step(net, criterion, opt):
            opt.zero_grad(set_to_none=True)
            logits, features = net(points, labels, return_features=True)
            regularizer, _ = criterion(features, torch.arange(4))
            (F.cross_entropy(logits, targets) + regularizer).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()

        step(model, loss_module, optimizer)
        scheduler.step()
        saved = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "rng": rng_state(),
                               "part_rng": loss_module.generator.get_state()})
        step(model, loss_module, optimizer)
        replica = NativePartSeg("pseg3", k=4).train()
        replica.load_state_dict(saved["model"])
        resumed_loss = HyCoReSegLoss(part_min=4, part_max=8, feature_dim=256)
        resumed_loss.generator.set_state(saved["part_rng"])
        opt = geoopt.optim.RiemannianSGD(replica.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
        opt.load_state_dict(saved["optimizer"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, eta_min=.0001)
        sch.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        step(replica, resumed_loss, opt)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, replica.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
