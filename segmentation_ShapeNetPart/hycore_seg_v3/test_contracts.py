"""CPU contracts for the three approved Pseg3 v3 arms and passive diagnostics.

Run after installation with
``python -m segmentation_ShapeNetPart.hycore_seg_v3.test_contracts``.
Tiny point clouds exercise the real model without reading a dataset or using CUDA.
"""

import ast
import copy
import importlib
import json
import random
import unittest
from pathlib import Path

import geoopt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v2.losses import choose_parts
from segmentation_ShapeNetPart.hycore_seg_v2.native import MobiusLayer, PoincareBall

from .losses import HyCoReSegLoss, relation_losses, sample_negative_indices
from .model import NativePseg3
from .monitoring import GeometryMonitor, gradient_tensor_summary, tensor_summary


def _trainable_count(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def _fresh(arm, seed=4780):
    torch.manual_seed(seed)
    return NativePseg3(arm, k=4)


def _rng_snapshot():
    # CPU-only: do not query or capture any CUDA generator.
    return {"torch": torch.get_rng_state().clone(),
            "python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state())}


def _restore_cpu_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_rng_equal(self, actual, expected):
        self.assert_exact(actual["torch"], expected["torch"])
        self.assertEqual(actual["python"], expected["python"])
        self.assertEqual(actual["numpy"][0], expected["numpy"][0])
        np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
        self.assertEqual(actual["numpy"][2:], expected["numpy"][2:])

    def test_fresh_arms_preserve_every_surviving_parameter_and_buffer(self):
        models = {arm: _fresh(arm) for arm in ("a", "b", "c")}
        states = {arm: model.state_dict() for arm, model in models.items()}
        self.assertEqual(set(states["a"]), set(states["c"]))
        self.assertTrue(any(name.startswith("transform_net.") for name in states["c"]))
        self.assertTrue(any(name.startswith("conv8.") for name in states["c"]))
        self.assertTrue(any(name.startswith("conv7.") for name in states["c"]))
        self.assertTrue(any(name.startswith("emb2.") for name in states["b"]))
        self.assertTrue(any(name.startswith("mlr.") for name in states["b"]))
        self.assertFalse(any(name.startswith("emb.") for name in states["b"]))
        for name, value in states["c"].items():
            self.assert_exact(states["a"][name], value)
            if name in states["b"]:
                self.assert_exact(states["b"][name], value)
        self.assertEqual(set(states["c"]) - set(states["b"]),
                         {name for name in states["c"] if name.startswith("emb.")})
        for arm in ("a", "b", "c"):
            repeated = _fresh(arm).state_dict()
            self.assertEqual(set(repeated), set(states[arm]))
            for name, value in states[arm].items():
                self.assert_exact(repeated[name], value)

    def test_b_removes_real_m1_parameters_and_executes_only_two_head_layers(self):
        a, b, c = (_fresh(arm) for arm in ("a", "b", "c"))
        self.assertEqual(_trainable_count(a), _trainable_count(c))
        self.assertEqual(_trainable_count(a) - _trainable_count(b), 65792)
        self.assertIsInstance(a.dp1, nn.Identity)
        self.assertIsInstance(b.dp1, nn.Identity)
        self.assertIsInstance(c.dp1, nn.Dropout)
        self.assertEqual(c.dp1.p, .5)
        self.assertFalse(hasattr(b, "emb"))
        self.assertEqual(a.conv8[0].in_channels, 1280)
        self.assertEqual(a.conv7[0].in_channels, 16)
        self.assertEqual(a.conv7[0].out_channels, 64)
        features = torch.randn(2, 7, 256) * .02
        for arm, model, expected_layers, expected_parameters in (
                ("a", a, 3, 105138), ("b", b, 2, 39346), ("c", c, 3, 105138)):
            with self.subTest(arm=arm):
                calls, handles = [], []
                for name, layer in model.named_modules():
                    if isinstance(layer, MobiusLayer):
                        handles.append(layer.register_forward_hook(
                            lambda module, args, result, key=name: calls.append(key)))
                try:
                    output = model.forward_head(features)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(output.shape, (2, 7, 50))
                self.assertEqual(len(calls), expected_layers)
                self.assertEqual(calls, ["emb", "emb2", "mlr"] if arm != "b" else ["emb2", "mlr"])
                self.assertEqual(sum(parameter.numel() for parameter in model.head_parameters()),
                                 expected_parameters)

    def test_a_c_eval_outputs_match_and_train_dropout_changes_actual_features(self):
        a, c = _fresh("a"), _fresh("c")
        points, categories = torch.randn(4, 3, 24), F.one_hot(torch.arange(4), 16).float()
        a.eval()
        c.eval()
        with torch.no_grad():
            self.assert_exact(a(points, categories), c(points, categories))
        a.train()
        c.train()
        before = torch.get_rng_state().clone()
        features_a = a.forward_features(points, categories)
        self.assert_exact(torch.get_rng_state(), before)
        features_c = c.forward_features(points, categories)
        self.assertFalse(torch.equal(torch.get_rng_state(), before))
        self.assertFalse(torch.equal(features_a, features_c))
        kept = features_c != 0
        self.assertTrue(kept.any())
        self.assertTrue((~kept).any())
        self.assert_exact(features_c[kept], 2 * features_a[kept])
        for model in (a, c):
            counters = [int(module.num_batches_tracked) for module in model.modules()
                        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
            self.assertTrue(counters and all(count == 1 for count in counters))

    def test_negative_sampler_guarantees_different_class_and_allows_reuse(self):
        labels = torch.tensor([0, 0, 0, 1])
        indices, counts, valid = sample_negative_indices(labels, torch.Generator().manual_seed(4780))
        self.assertEqual(counts.tolist(), [1, 1, 1, 3])
        self.assertEqual(indices[:3].tolist(), [3, 3, 3])
        self.assertTrue(valid.all())
        self.assertTrue((labels != labels[indices]).all())
        self.assertTrue((torch.arange(4) != indices).all())
        self.assertTrue(all(index in (0, 1, 2) for index in indices[3:].tolist()))
        indices, counts, valid = sample_negative_indices(torch.zeros(4, dtype=torch.long),
                                                         torch.Generator().manual_seed(4780))
        self.assertEqual(indices.tolist(), [-1] * 4)
        self.assertEqual(counts.tolist(), [0] * 4)
        self.assertFalse(valid.any())

    def test_negative_sampler_is_uniform_over_samples_not_categories(self):
        # Anchor zero has two class-1 candidates and one class-2 candidate.
        # Uniform classes would give probabilities 1/4, 1/4, 1/2 and fail this test.
        labels = torch.tensor([0, 1, 1, 2])
        generator = torch.Generator().manual_seed(4780)
        hits = torch.zeros(4, dtype=torch.long)
        draws = 4096
        for _ in range(draws):
            indices, counts, valid = sample_negative_indices(labels, generator)
            self.assertEqual(counts.tolist(), [3, 2, 2, 3])
            self.assertTrue(valid.all())
            hits[indices[0]] += 1
        self.assertEqual(int(hits[0]), 0)
        for count in hits[1:]:
            self.assertLess(abs(float(count) / draws - 1 / 3), .04)

    def test_negative_rng_is_independent_and_restorable(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        generator = torch.Generator().manual_seed(4780)
        before, local = _rng_snapshot(), generator.get_state().clone()
        first = sample_negative_indices(labels, generator)
        self.assert_rng_equal(_rng_snapshot(), before)
        generator.set_state(local)
        repeated = sample_negative_indices(labels, generator)
        for actual, expected in zip(repeated, first):
            self.assert_exact(actual, expected)

    def test_real_contrast_uses_gathered_negative_and_preserves_its_gradient(self):
        ball = PoincareBall(c=1., dim=2).double()
        whole = torch.tensor([[.05, .01], [-.03, .02], [.02, -.04]],
                             dtype=torch.float64, requires_grad=True)
        part = torch.tensor([[.01, .09], [-.06, .03], [.07, .08]],
                            dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([1, -1, -1])
        valid = torch.tensor([True, False, False])
        _, contrast, info = relation_losses(ball, whole, part, .2, 4., indices, valid)
        expression = ball.dist(whole[:1], part[:1]) - ball.dist(whole[:1], part[1:2]) + 4.
        expected = torch.maximum(expression, torch.zeros_like(expression)).mean()
        self.assert_exact(contrast, expected)
        actual_grads = torch.autograd.grad(contrast, (whole, part), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (whole, part))
        for actual, reference in zip(actual_grads, expected_grads):
            self.assert_exact(actual, reference)
            self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual_grads[1][1].norm()), 0.)
        self.assert_exact(actual_grads[1][2], torch.zeros(2, dtype=torch.float64))
        self.assertEqual(info["valid_negative"], 1)

    def test_no_candidates_gives_differentiable_zero_contrast_without_fake_satisfaction(self):
        ball = PoincareBall(c=1., dim=2).double()
        whole = torch.tensor([[.05, .01], [-.03, .02]], dtype=torch.float64, requires_grad=True)
        part = torch.tensor([[.01, .09], [-.06, .03]], dtype=torch.float64, requires_grad=True)
        radial, contrast, info = relation_losses(ball, whole, part, .2, 4.,
                                                 torch.tensor([-1, -1]))
        self.assertTrue(contrast.requires_grad)
        self.assertEqual(float(contrast), 0.)
        gradients = torch.autograd.grad(contrast, (whole, part), retain_graph=True)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assert_exact(gradient, torch.zeros_like(gradient))
        self.assertEqual(info["valid_negative"], 0)
        self.assertTrue(info["no_negative_candidates"])
        self.assertIsNone(info["contr_satisfied"])
        self.assertIsNone(info["negative_distance"])
        (radial + contrast).backward()
        self.assertTrue(torch.isfinite(whole.grad).all())
        self.assertTrue(torch.isfinite(part.grad).all())

    def test_actual_feature_regularizer_matches_independent_mean_then_exp_formula(self):
        torch.manual_seed(4780)
        features = (torch.randn(4, 11, 5, dtype=torch.float64) * .03).requires_grad_()
        labels = torch.tensor([0, 0, 0, 1])
        criterion = HyCoReSegLoss(alpha=.03, beta=.07, gamma_base=.6, delta=4.,
                                  part_min=3, part_max=3, seed=4780, feature_dim=5)
        part_state, negative_state = (criterion.generator.get_state().clone(),
                                      criterion.negative_generator.get_state().clone())
        actual, info = criterion(features, labels)
        part_generator = torch.Generator().set_state(part_state)
        negative_generator = torch.Generator().set_state(negative_state)
        pooled, _, _, count = choose_parts(features, part_generator, 3, 3)
        indices, _, valid = sample_negative_indices(labels, negative_generator)
        whole, part = criterion.ball.expmap0(features.mean(1)), criterion.ball.expmap0(pooled)
        raw_hier = criterion.ball.dist0(whole) - criterion.ball.dist0(part) + .6 / count
        raw_contr = criterion.ball.dist(whole[valid], part[valid])
        raw_contr = raw_contr - criterion.ball.dist(whole[valid], part[indices[valid]]) + 4.
        expected = .07 * torch.maximum(raw_hier, torch.zeros_like(raw_hier)).mean()
        expected = expected + .03 * torch.maximum(raw_contr, torch.zeros_like(raw_contr)).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assert_exact(actual, criterion.last_components["contr"] + criterion.last_components["hier"])
        self.assertEqual(info["candidate_counts"], [1, 1, 1, 3])
        self.assertEqual(info["negative_legal_rate"], 1.)
        self.assertEqual(info["negative_max_reuse"], 3)
        self.assertEqual(info["same_class_pair_fraction"], 0.)
        self.assertAlmostEqual(info["gamma"] * info["part_points"], .6)
        actual_gradient = torch.autograd.grad(actual, features, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, features)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    def test_both_local_rng_streams_restore_complete_loss_and_diagnostics(self):
        features = (torch.randn(4, 19, 8) * .03).requires_grad_()
        labels = torch.tensor([0, 1, 1, 2])
        criterion = HyCoReSegLoss(part_min=3, part_max=9, feature_dim=8)
        states = {"part": criterion.generator.get_state().clone(),
                  "negative": criterion.negative_generator.get_state().clone()}
        before_global = _rng_snapshot()
        first, first_info = criterion(features, labels)
        self.assert_rng_equal(_rng_snapshot(), before_global)
        # Recreating a criterion cannot substitute for saving two consumed streams.
        criterion(features, labels)
        resumed = HyCoReSegLoss(part_min=3, part_max=9, feature_dim=8)
        resumed.generator.set_state(states["part"])
        resumed.negative_generator.set_state(states["negative"])
        again, again_info = resumed(features, labels)
        self.assert_exact(first, again)
        self.assertEqual(first_info, again_info)
        first_grad = torch.autograd.grad(first, features, retain_graph=True)[0]
        again_grad = torch.autograd.grad(again, features)[0]
        self.assert_exact(first_grad, again_grad)

    def test_all_same_class_feature_batch_keeps_radial_and_ce_backpropagation(self):
        features = (torch.randn(4, 13, 6) * .04).requires_grad_()
        criterion = HyCoReSegLoss(part_min=3, part_max=5, feature_dim=6)
        regularizer, info = criterion(features, torch.zeros(4, dtype=torch.long))
        self.assertEqual(info["no_candidate_anchors"], 4)
        self.assertEqual(info["negative_indices"], [-1] * 4)
        self.assertEqual(info["contr"], 0.)
        self.assertIsNone(info["negative_legal_rate"])
        self.assertEqual(float(criterion.last_components["contr"]), 0.)
        ce = features.square().mean()
        (ce + regularizer).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.norm()), 0.)

    def test_pooled_exp_boundary_does_not_imply_all_point_exp_on_boundary(self):
        ball = PoincareBall(c=1., dim=2)
        features = torch.tensor([[[10., 0.], [10., 0.], [0., 0.], [0., 0.]]])
        point_exp = ball.expmap0(features)
        whole_tangent = features.mean(1)
        whole_exp = ball.expmap0(whole_tangent)
        point_stats = tensor_summary("point_exp", point_exp, tangent_input=features)
        whole_stats = tensor_summary("whole_exp", whole_exp, tangent_input=whole_tangent)
        self.assertEqual(point_stats["near_boundary_fraction"], .5)
        self.assertEqual(whole_stats["near_boundary_fraction"], 1.)
        self.assertEqual(point_stats["point_norm"]["p10"], 0.)
        self.assertGreater(whole_stats["norm"]["p10"], .99)
        self.assertEqual(point_stats["analytic_projection_rate"], .5)
        self.assertEqual(whole_stats["analytic_projection_rate"], 1.)
        opposing = torch.tensor([[[10., 0.], [-10., 0.]]])
        self.assertEqual(tensor_summary("point_exp", ball.expmap0(opposing))["near_boundary_fraction"], 1.)
        self.assertEqual(tensor_summary("whole_exp", ball.expmap0(opposing.mean(1)))["near_boundary_fraction"], 0.)

    def test_gradient_none_zero_and_point_vs_pooled_norms_stay_distinct(self):
        self.assertEqual(gradient_tensor_summary(None)["status"], "none")
        self.assertIsNone(gradient_tensor_summary(None)["l2"])
        zero = gradient_tensor_summary(torch.zeros(1, 2, 2))
        self.assertEqual(zero["status"], "zero")
        self.assertEqual(zero["l2"], 0.)
        gradients = torch.tensor([[[1., 0.], [-1., 0.]]])
        summary = gradient_tensor_summary(gradients, torch.ones_like(gradients))
        self.assertEqual(summary["point_gradient_norm"]["mean"], 1.)
        self.assertEqual(summary["object_mean_gradient_norm"]["mean"], 0.)
        monitor = GeometryMonitor().begin(True)
        features = torch.randn(2, 3, 4, requires_grad=True)
        unused = torch.randn(2, 3, 4, requires_grad=True)
        monitor.capture("h", features)
        monitor.capture("m1", unused)
        report = monitor.gradient_summary({"hier": features.square().sum(), "zero": (features * 0).sum()})
        self.assertEqual(report["gradients"]["hier"]["nodes"]["m1"]["status"], "none")
        self.assertEqual(report["gradients"]["zero"]["nodes"]["h"]["status"], "zero")
        self.assertIsNone(features.grad)
        self.assertIsNone(unused.grad)
        json.dumps(report, allow_nan=False)

    def test_actual_monitor_switch_preserves_outputs_gradients_bn_and_all_cpu_rng(self):
        baseline, observed = _fresh("c"), _fresh("c")
        observed.load_state_dict(baseline.state_dict())
        points = torch.randn(4, 3, 24)
        categories = F.one_hot(torch.arange(4), 16).float()
        targets = torch.randint(50, (4, 24))
        initial_rng = _rng_snapshot()
        runs = []
        for enabled, model in ((False, baseline), (True, observed)):
            _restore_cpu_rng(initial_rng)
            criterion = HyCoReSegLoss(part_min=4, part_max=8, feature_dim=256)
            monitor = GeometryMonitor().begin(enabled, epoch=1, step=1)
            model.monitor = criterion.monitor = monitor
            model.train()
            logits, features = model(points, categories, return_features=True)
            regularizer, info = criterion(features, torch.arange(4))
            ce = F.cross_entropy(logits, targets)
            losses = {"ce": ce, **criterion.last_components}
            before_observation = _rng_snapshot()
            geometry = monitor.summarize()
            derivative = monitor.gradient_summary(losses, parameter_groups={
                "shared": dict(model.conv1[0].named_parameters()),
                "m1": dict(model.emb.named_parameters()),
                "m2": dict(model.emb2.named_parameters()),
                "m3": dict(model.mlr.named_parameters())})
            self.assert_rng_equal(_rng_snapshot(), before_observation)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            if enabled:
                self.assertEqual(set(geometry["geometry"]), {
                    "h", "point_exp", "m1", "m2", "m3", "whole_tangent", "part_tangent", "whole_exp", "part_exp"})
                for loss_name in ("contr", "hier"):
                    for node in ("point_exp", "m1", "m2", "m3"):
                        self.assertEqual(derivative["gradients"][loss_name]["nodes"][node]["status"], "none")
                    for group in ("m1", "m2", "m3"):
                        self.assertEqual(derivative["gradients"][loss_name]["parameter_groups"][group]["status"], "none")
                json.dumps({"geometry": geometry, "derivatives": derivative}, allow_nan=False)
            else:
                self.assertEqual(geometry["geometry"], {})
                self.assertEqual(derivative["gradients"], {})
            (ce + regularizer).backward()
            runs.append({"logits": logits.detach().clone(), "features": features.detach().clone(),
                         "loss": (ce + regularizer).detach().clone(), "info": info,
                         "gradients": {name: None if parameter.grad is None else parameter.grad.detach().clone()
                                       for name, parameter in model.named_parameters()},
                         "state": copy.deepcopy(model.state_dict()), "rng": _rng_snapshot(),
                         "part_rng": criterion.generator.get_state().clone(),
                         "negative_rng": criterion.negative_generator.get_state().clone()})
            monitor.clear()
            self.assertEqual(monitor.nodes, {})
        first, second = runs
        for name in ("logits", "features", "loss", "part_rng", "negative_rng"):
            self.assert_exact(second[name], first[name])
        self.assertEqual(second["info"], first["info"])
        self.assert_rng_equal(second["rng"], first["rng"])
        for name, gradient in first["gradients"].items():
            if gradient is None:
                self.assertIsNone(second["gradients"][name])
            else:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assert_exact(second["gradients"][name], gradient)
        for name, value in first["state"].items():
            self.assert_exact(second["state"][name], value)

    def test_training_norm_observation_does_not_shrink_large_gradients(self):
        trainer = importlib.import_module(__package__ + ".train")
        first = nn.Parameter(torch.tensor([1., -2.]))
        second = nn.Parameter(torch.tensor([3.]))
        first.grad, second.grad = torch.tensor([3., 4.]), torch.tensor([12.])
        before = [first.grad.clone(), second.grad.clone()]
        norm = trainer.global_gradient_norm([first, second])
        self.assertAlmostEqual(float(norm), 13., places=6)
        self.assertGreater(float(norm), 1.)
        self.assert_exact(first.grad, before[0])
        self.assert_exact(second.grad, before[1])
        optimizer = torch.optim.SGD([first, second], lr=.1)
        optimizer.step()
        torch.testing.assert_close(first, torch.tensor([.7, -2.4]), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(second, torch.tensor([1.8]), rtol=1e-6, atol=1e-7)

    def test_actual_trainer_contains_no_gradient_clipping_call(self):
        path = Path(__file__).with_name("train.py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = {"clip_grad_norm_", "clip_grad_value_", "clip_grad_norm", "clip_grad_value"}
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        called_names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                        for node in calls if isinstance(node.func, (ast.Attribute, ast.Name))}
        self.assertFalse(called_names & forbidden)
        self.assertIn("global_gradient_norm", called_names)

    def test_no_cpu_contract_initialized_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
