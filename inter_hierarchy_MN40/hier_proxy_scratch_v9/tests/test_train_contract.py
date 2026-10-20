"""CPU contracts for the actual V9 entry point, geometry and HIER route.

The real PointMLP is instantiated on CPU; only its projection/Mobius tail is
executed, so CUDA-only FPS/grouping kernels are never called.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import sys
import unittest
from unittest import mock

import geoopt
import torch
from torch import nn

from .. import geometry
from ..telemetry import EpochTelemetry


class V9TrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual entry point and its production dependencies. No
        # process group, dataset or CUDA device is initialized by this import.
        from .. import train
        cls.train = train
        torch.set_num_threads(2)

    def production_args(self, *extra):
        argv = ["v9.train", "--data-dir", "unused-training-data", "--run-dir", "unused-new-run", *extra]
        with mock.patch.object(sys, "argv", argv):
            return self.train.arguments()

    def test_real_native_pointmlp_state_and_mobius_tail_are_unchanged(self):
        cuda_was_initialized = torch.cuda.is_initialized()
        torch.manual_seed(22)
        original = self.train.Hype_pointMLP().eval()
        torch.manual_seed(22)
        candidate = self.train.Hype_pointMLP().eval()
        before_keys = tuple(candidate.state_dict())
        before_parameter_ids = tuple(id(parameter) for parameter in candidate.parameters())
        configured = geometry.configure_backbone(candidate)
        self.assertIs(configured, candidate)
        self.assertEqual(tuple(candidate.state_dict()), before_keys)
        self.assertEqual(tuple(id(parameter) for parameter in candidate.parameters()), before_parameter_ids)
        self.assertEqual(tuple(original.state_dict()), before_keys)
        for key, reference in original.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[key], reference, rtol=0, atol=0)
        for manifold in (candidate.manifold, candidate.manifold2,
                         candidate.emb.manifold, candidate.classifier.manifold):
            self.assertAlmostEqual(float(manifold.c), 1., places=6)
            self.assertFalse(any(parameter.requires_grad for parameter in manifold.parameters()))
        self.assertIs(candidate.emb.manifold, candidate.manifold)
        self.assertIs(candidate.classifier.manifold, candidate.manifold2)
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in candidate.modules()))
        self.assertFalse(hasattr(candidate, "v8_geometry") or hasattr(candidate, "v9_geometry"))

        inputs = torch.randn(4, original.proj[0].in_features,
                             generator=torch.Generator().manual_seed(81)) * .03
        a, b = inputs.requires_grad_(), inputs.detach().clone().requires_grad_()

        def tail(model, value):
            mu = model.emb(model.manifold.expmap0(model.proj(value)))
            return mu, model.classifier(mu)

        original_mu, original_logits = tail(original, a)
        candidate_mu, candidate_logits = tail(candidate, b)
        torch.testing.assert_close(candidate_mu, original_mu, rtol=0, atol=0)
        torch.testing.assert_close(candidate_logits, original_logits, rtol=0, atol=0)
        expected = torch.autograd.grad(original_logits.square().sum() + original_mu.square().sum(), a)[0]
        actual = torch.autograd.grad(candidate_logits.square().sum() + candidate_mu.square().sum(), b)[0]
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreater(float(actual.norm()), 0.)
        # FP32 backward can accumulate in a different order when curvature
        # is frozen. States and forward values above still require bitwise equality.
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-8)
        self.assertEqual(torch.cuda.is_initialized(), cuda_was_initialized)

    def test_production_configuration_records_both_domains_and_original_base_margins(self):
        args = self.production_args()
        config = self.train.fixed_configuration(args)
        for key in ("c", "native_c"):
            self.assertEqual(config[key], 1.)
        self.assertEqual(config["hier_c"], .1)
        self.assertIsNone(config["native_extra_cap"])
        self.assertFalse(config["native_GS"])
        self.assertEqual(config["hier_view_depth"], 4.6)
        self.assertEqual(config["proxy_tangent_cap"], 2.3)
        self.assertTrue(config["HIER_backward_hook"])
        self.assertEqual(config["hook_scope"], "HIER whole view and proxies only")
        self.assertEqual((config["radial_gamma"], config["contrastive_margin"]), (1000., 4.))
        self.assertEqual((config["alpha_radial"], config["alpha_contrastive"]), (.01, .01))
        self.assertEqual(config["minimum_direction_norm"], 1e-8)
        self.assertEqual(config["invalid_direction_policy"], "fail; never replace or detach")
        self.assertEqual((config["seed"], config["epochs"], config["steps_per_epoch"]), (22, 300, 200))
        self.assertEqual((config["world_size"], config["global_batch"],
                          config["classes_per_rank"], config["instances_per_class"]), (2, 64, 16, 2))
        self.assertTrue(config["part_overwrites_whole"])
        self.assertEqual(config["warmup_epochs"], 0)
        self.assertEqual(config["lambda_hier_after_warmup"], .5)
        self.assertEqual(config["model_optimizer"]["name"], "RiemannianSGD")
        self.assertEqual(config["proxy_optimizer"]["name"], "AdamW")
        self.assertEqual(config["proxy_optimizer"]["eps"], 1e-8)
        self.assertEqual(config["validation"], "none")
        self.assertIn("test2468 every epoch", config["selection"])
        self.assertIsNone(config["evaluation_batch_limit"])

    def test_cli_disallows_unapproved_production_changes_and_keeps_smoke_training_only(self):
        cases = (("--seed", "4780"), ("--lambda-hier", ".1"),
                 ("--warmup-epochs", "20"), ("--epochs", "2"),
                 ("--steps-per-epoch", "2"),
                 ("--smoke", "--epochs", "2", "--steps-per-epoch", "2"))
        for extra in cases:
            with self.subTest(arguments=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.production_args(*extra)
        smoke = self.production_args("--smoke", "--epochs", "2", "--steps-per-epoch", "2", "--skip-final-test")
        self.assertTrue(smoke.smoke and smoke.skip_final_test)
        self.assertEqual(self.train.fixed_configuration(smoke)["evaluation_batch_limit"], 2)

    def test_native_telemetry_accepts_depth_over4_6_without_reporting_live_cap_or_gs(self):
        whole = torch.tensor([[.99, 0., 0.], [.995, 0., 0.]], dtype=torch.float64)
        part = torch.tensor([[.90, 0., 0.], [.97, 0., 0.]], dtype=torch.float64)
        telemetry = EpochTelemetry([10, 11], [0, 1], num_classes=40)
        telemetry.update(ids=torch.tensor([10, 11]), labels=torch.tensor([0, 1]),
                         logits=torch.zeros(2, 40, dtype=torch.float64), mu=whole, nu=part,
                         scalar_metrics={"lambda_hier": .5})
        summary = telemetry.summary()["geometry"]
        self.assertGreater(summary["whole"]["depth"]["min"], 4.6)
        torch.testing.assert_close(torch.tensor(summary["whole"]["depth"]["mean"], dtype=torch.float64),
                                   geometry.native_depth(whole).mean(), rtol=1e-13, atol=1e-13)
        self.assertEqual(summary["whole"]["shadow_tangent_cap_count"], 2)
        for role in ("whole", "part"):
            report = summary[role]
            self.assertEqual(report["c"], 1.)
            self.assertIsNone(report["live_tangent_cap"])
            self.assertIsNone(report["live_depth_cap"])
            self.assertFalse(report["inverse_metric_hook_applied"])
            self.assertTrue(report["inverse_metric_factor_is_counterfactual_not_applied"])
            self.assertTrue(report["shadow_is_final_mu_logmap_counterfactual_not_live_extra_clip"])
        domain = summary["hier_view_domain"]
        self.assertEqual((domain["c"], domain["target_whole_depth"], domain["hook_applied"]), (.1, 4.6, True))
        native_stats = geometry.native_geometry_stats(whole)
        self.assertFalse(native_stats["extra_cap"] or native_stats["inverse_metric_hook"])
        self.assertGreater(native_stats["depth_min"], 4.6)

    def test_actual_hier_sample_gradient_reaches_encoder_but_proxy_only_does_not(self):
        torch.manual_seed(40)
        encoder = nn.Linear(6, 8, bias=False).double()
        classifier = nn.Linear(8, 40).double()
        inputs = torch.randn(64, 6, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
        native = geoopt.PoincareBall(c=1.).double().expmap0(encoder(inputs) * .3)
        logits = classifier(native)
        labels = torch.arange(32).repeat_interleave(2)
        proxy = self.train.HIERLoss(num_proxies=32, dim=8, seed=22).double()
        native_before = native.detach().clone()
        hier_view = geometry.whole_to_hier(native)
        self.assertTrue(hier_view.requires_grad)
        geometry.assert_hier_shell(hier_view)
        torch.testing.assert_close(native, native_before, rtol=0, atol=0)
        _, stats = proxy(hier_view, labels, topk=6, t_per_anchor=3, seed=81,
                         generator=torch.Generator().manual_seed(81),
                         return_components=True, monitor_structure=False)
        components = stats["_components"]
        native_gradient, encoder_gradient, classifier_gradient, proxy_gradient = torch.autograd.grad(
            components["sample_loss"], (native, encoder.weight, classifier.weight, proxy.tangent_proxies),
            allow_unused=True, retain_graph=True)
        for gradient in (native_gradient, encoder_gradient, proxy_gradient):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.norm()), 1e-12)
        self.assertIsNone(classifier_gradient)
        # The fixed-shell branch discards only the direct radial component.
        radial = (native_gradient * native.detach()).sum(-1)
        torch.testing.assert_close(radial, torch.zeros_like(radial), rtol=0, atol=1e-10)
        native_from_proxy, encoder_from_proxy, proxy_from_proxy = torch.autograd.grad(
            components["proxy_loss"], (native, encoder.weight, proxy.tangent_proxies), allow_unused=True)
        self.assertIsNone(native_from_proxy)
        self.assertIsNone(encoder_from_proxy)
        self.assertTrue(bool(torch.isfinite(proxy_from_proxy).all()))
        self.assertGreater(float(proxy_from_proxy.norm()), 1e-12)
        # The classifier is a native branch and still receives base CE gradients.
        ce_gradient = torch.autograd.grad(self.train.smoothed_ce(logits, labels), classifier.weight)[0]
        self.assertGreater(float(ce_gradient.norm()), 1e-12)

    def test_production_active_hier_route_passes_live_view_not_detached_native(self):
        tree = ast.parse(inspect.getsource(self.train.main))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        view_assignments = [node for node in assignments
                            if any(isinstance(target, ast.Name) and target.id == "hier_mu"
                                   for target in node.targets)]
        self.assertEqual(len(view_assignments), 1)
        expected = ast.parse('hier_mu = whole_to_hier(values["mu"])').body[0].value
        self.assertEqual(ast.dump(view_assignments[0].value), ast.dump(expected))
        active = [node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                  and node.test.id == "active_hier"]
        live_calls = [node for branch in active for statement in branch.body for node in ast.walk(statement)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "proxy"]
        self.assertEqual(len(live_calls), 1)
        self.assertIsInstance(live_calls[0].args[0], ast.Name)
        self.assertEqual(live_calls[0].args[0].id, "hier_mu")
        total_loss = [node for node in assignments
                      if any(isinstance(target, ast.Name) and target.id == "loss" for target in node.targets)]
        expected_loss = ast.parse('loss = values["base"] + lambda_hier * hier_loss').body[0].value
        self.assertTrue(any(ast.dump(node.value) == ast.dump(expected_loss) for node in total_loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)