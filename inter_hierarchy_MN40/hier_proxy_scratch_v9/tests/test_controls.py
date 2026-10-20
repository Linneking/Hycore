"""Contracts for the native-only, matched global64 V9 control.

The reference is the base route of the running V9 main experiment, not an
older V8 geometry or a single-card shuffle protocol. CUDA kernels are not
called: the real PointMLP is only instantiated for initialization identity.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import sys
import unittest
from unittest import mock

from geoopt import PoincareBall
import torch

from .. import controls, train
from ..geometry import configure_backbone


NATIVE_SEED22_SHA256 = "fd7251e0eb6b397ecaf3862e9cc00a6bc2be9b684329fdc31dc05915b388d29d"


def parsed_arguments(module=controls, *extra):
    argv = ["v9.controls", "--data-dir", "unused-train-data", "--run-dir", "unused-new-run", *extra]
    with mock.patch.object(sys, "argv", argv):
        return module.arguments()


def assigned_expressions(function, name):
    tree = ast.parse(inspect.getsource(function))
    return [ast.dump(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


class V9ControlContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_training_configuration_matches_v9_main(self):
        main = train.fixed_configuration(parsed_arguments(train))
        control = controls.fixed_configuration(parsed_arguments())
        keys = ("architecture", "num_classes", "seed", "epochs", "steps_per_epoch",
                "global_batch", "world_size", "classes_per_rank", "instances_per_class", "sampler",
                "c", "base_c", "native_c", "D", "label_smoothing", "alpha_contrastive", "alpha_radial",
                "contrastive_margin", "radial_gamma", "radial_margin", "whole_count_range", "child_count_range",
                "part_overwrites_whole", "FPS_first_stage", "BN", "precision", "native_extra_cap", "native_GS",
                "intra_hinge", "intra_distance", "model_optimizer", "validation", "training_pool", "selection",
                "selection_forward", "train_eval_every", "eval_batch_size", "workers", "audit_epochs",
                "smoke", "evaluation_batch_limit")
        self.assertEqual({key: control[key] for key in keys}, {key: main[key] for key in keys})
        self.assertEqual((control["c"], control["radial_gamma"], control["contrastive_margin"]), (1., 1000., 4.))
        self.assertIsNone(control["native_extra_cap"])
        self.assertFalse(control["native_GS"])
        self.assertEqual(control["canonical_name"], "V9-B64-C1-CAPnone-GS0-M1000_4-TS-S22")

    def test_control_has_no_hier_parameters_optimizer_or_view(self):
        config = controls.fixed_configuration(parsed_arguments())
        self.assertFalse(config["HIER_objective_enabled"])
        self.assertEqual((config["HIER_calls_per_step"], config["P"], config["lambda_hier_after_warmup"]), (0, 0, 0.))
        self.assertFalse(config["HIER_backward_hook"])
        for key in ("proxy_optimizer", "proxy_initialization", "proxy_parameter_projection", "sample_K", "proxy_K",
                    "T", "margin", "tau", "hier_c", "hier_view", "hier_view_depth", "proxy_tangent_cap", "hook_scope"):
            self.assertIsNone(config[key], key)
        self.assertEqual(config["geometry_domains"], {"native_whole_part": 1.})

    def test_cli_production_is_b64_seed22_full300_and_smoke_is_bounded(self):
        args = parsed_arguments()
        self.assertEqual((args.control, args.seed, args.epochs, args.steps_per_epoch), ("B64", 22, 300, 200))
        for extra in (("--control", "B0-32"), ("--seed", "4780"), ("--epochs", "2"),
                      ("--steps-per-epoch", "307"), ("--lambda-hier", ".5"), ("--warmup-epochs", "20"),
                      ("--smoke", "--epochs", "2", "--steps-per-epoch", "2"),
                      ("--smoke", "--epochs", "3", "--steps-per-epoch", "2", "--skip-final-test")):
            with self.subTest(arguments=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parsed_arguments(controls, *extra)
        smoke = parsed_arguments(controls, "--smoke", "--epochs", "2", "--steps-per-epoch", "2", "--skip-final-test")
        self.assertTrue(smoke.smoke and smoke.skip_final_test)
        self.assertEqual(controls.fixed_configuration(smoke)["evaluation_batch_limit"], 2)

    def test_native_e0_identity_matches_running_v9_main_without_extra_hooks(self):
        train.seed_all(22)
        main = configure_backbone(train.Hype_pointMLP())
        controls.seed_all(22)
        control = controls.configure_backbone(controls.Hype_pointMLP())
        self.assertEqual(tuple(main.state_dict()), tuple(control.state_dict()))
        for key, reference in main.state_dict().items():
            torch.testing.assert_close(control.state_dict()[key], reference, rtol=0, atol=0)
        self.assertEqual(controls.state_hash(control), train.state_hash(main))
        self.assertEqual(controls.state_hash(control), NATIVE_SEED22_SHA256)
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in control.modules()))
        for manifold in (control.manifold, control.manifold2, control.emb.manifold, control.classifier.manifold):
            self.assertAlmostEqual(float(manifold.c), 1., places=6)
        self.assertFalse(any("prox" in name.lower() or "hier" in name.lower() for name, _ in control.named_parameters()))

    def test_actual_control_loss_and_gradients_equal_native_v9_base(self):
        generator = torch.Generator().manual_seed(81)
        whole_tangent = torch.randn(64, 8, generator=generator, dtype=torch.float64) * .15
        part_tangent = torch.randn(64, 8, generator=generator, dtype=torch.float64) * .08
        ball = PoincareBall(c=1.).double()
        mu = ball.expmap0(whole_tangent).requires_grad_()
        nu = ball.expmap0(part_tangent).requires_grad_()
        logits = torch.randn(64, 40, generator=generator, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(32).repeat_interleave(2)
        outputs = {"mu": mu, "nu": nu, "logits": logits, "whole_count": 900, "child_count": 400}
        actual = controls.control_base_losses(outputs, labels, "B64")
        expected = train.global_base_losses(outputs, labels)
        for key in ("base", "ce", "intra_contrastive", "intra_radial", "whole_depth", "part_depth"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        actual_gradients = torch.autograd.grad(actual["base"], (mu, nu, logits), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected["base"], (mu, nu, logits))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            self.assertTrue(bool(torch.isfinite(actual_gradient).all()))
            self.assertGreater(float(actual_gradient.norm()), 0.)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)

    def test_real_training_route_never_constructs_hier_or_proxy_and_uses_only_base_loss(self):
        source = ast.parse(inspect.getsource(controls))
        forbidden = {"HIERLoss", "whole_to_hier", "InverseMetric", "AdamW", "proxy_update_stats"}
        imports = {alias.name.rsplit(".", 1)[-1] for node in ast.walk(source)
                   if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        self.assertFalse(imports & forbidden, imports & forbidden)
        calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)]
        called_names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                        for node in calls if isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(called_names & forbidden, called_names & forbidden)
        self.assertNotIn("proxy", called_names)
        expected = ast.dump(ast.parse('values["base"]').body[0].value)
        self.assertIn(expected, assigned_expressions(controls.main, "loss"))

    def test_smoke_test_file_loading_exists_only_in_non_smoke_branch(self):
        tree = ast.parse(inspect.getsource(controls.main))

        def is_test_load(node):
            return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_shards"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "test")

        test_calls = [node for node in ast.walk(tree) if is_test_load(node)]
        self.assertEqual(len(test_calls), 1)
        smoke_branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name)
                          and node.test.value.id == "args" and node.test.attr == "smoke"]
        self.assertTrue(any(test_calls[0] in [descendant for statement in branch.orelse for descendant in ast.walk(statement)]
                            for branch in smoke_branches))
        self.assertFalse(any(is_test_load(descendant) for branch in smoke_branches
                             for statement in branch.body for descendant in ast.walk(statement)))

    def test_sampling_crop_and_step_rng_match_running_v9_reference(self):
        self.assertIs(controls.PointClouds, train.PointClouds)
        self.assertIs(controls.SourceClassBatchSampler, train.SourceClassBatchSampler)
        self.assertIs(controls.shared_crop_counts, train.shared_crop_counts)
        for key in ("epoch_seed", "step_seed"):
            self.assertEqual(assigned_expressions(controls.main, key), assigned_expressions(train.main, key))
        labels = torch.arange(40).repeat_interleave(8).tolist()
        for rank in (0, 1):
            candidate = controls.SourceClassBatchSampler(labels, rank=rank, seed=22, steps=3)
            reference = train.SourceClassBatchSampler(labels, rank=rank, seed=22, steps=3)
            candidate.set_epoch(2)
            reference.set_epoch(2)
            self.assertEqual(list(candidate), list(reference))
        for epoch, step in ((0, 0), (0, 199), (299, 199)):
            self.assertEqual(controls.shared_crop_counts(22, epoch, step), train.shared_crop_counts(22, epoch, step))

    def test_telemetry_reports_only_native_geometry_with_no_fake_hier_view(self):
        whole = torch.tensor([[.99, 0., 0.], [.995, 0., 0.]], dtype=torch.float64)
        part = torch.tensor([[.90, 0., 0.], [.97, 0., 0.]], dtype=torch.float64)
        monitor = controls.EpochTelemetry([10, 11], [0, 1], num_classes=40)
        monitor.update(ids=torch.tensor([10, 11]), labels=torch.tensor([0, 1]), logits=torch.zeros(2, 40),
                       mu=whole, nu=part, scalar_metrics={"lambda_hier": 0.})
        geometry = monitor.summary()["geometry"]
        self.assertGreater(geometry["whole"]["depth"]["min"], 4.6)
        for role in ("whole", "part"):
            self.assertEqual(geometry[role]["c"], 1.)
            self.assertIsNone(geometry[role]["live_tangent_cap"])
            self.assertFalse(geometry[role]["inverse_metric_hook_applied"])
        view = geometry["hier_view_domain"]
        self.assertFalse(view["available"] or view["hook_applied"])
        self.assertIsNone(view["c"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
