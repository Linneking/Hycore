"""Invariant and real-kernel controls for the bounded frozen-backbone probe."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.hier_postrun_audit.backbone_probe import run_backbone_probe

TORCH = importlib.util.find_spec("torch") is not None


class BackboneGuardTests(unittest.TestCase):
    def test_gpu_is_explicit_before_any_checkpoint_or_cuda_access(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "explicit physical GPU"):
                run_backbone_probe({}, Path(folder) / "new", folder, gpu=None)

    def test_existing_output_refused_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing"
            path.mkdir()
            marker = path / "marker.txt"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                run_backbone_probe({}, path, folder, gpu=None)
            self.assertEqual(marker.read_text(), "preserve")

    def test_full_checkpoint_requires_explicit_own_trust(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "own trusted"):
                run_backbone_probe({}, Path(folder) / "new", folder, gpu=1)


@unittest.skipUnless(TORCH, "CPU PyTorch not available in the local runtime")
class BackboneTorchTests(unittest.TestCase):
    def _model(self):
        import torch
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3, 4)
                self.bn = torch.nn.BatchNorm1d(4)
                self.classifier = torch.nn.Linear(4, 4)

            def forward(self, points, emb=False):
                features = self.bn(self.encoder(points.mean(dim=-1)))
                mu = .65 * torch.tanh(features)
                radius = mu.norm(dim=-1, keepdim=True)
                mu = mu * (.8 / radius.clamp_min(1e-12)).clamp(max=1)
                return (mu, mu) if emb else (mu, self.classifier(mu))
        return ToyModel()

    def _data(self):
        rng = np.random.default_rng(251)
        clouds = rng.normal(size=(8, 16, 3)).astype(np.float32)
        return clouds, np.repeat(np.arange(4), 2), np.arange(100, 108)

    def test_encoder_base_gradients_bn_rng_inputs_and_existing_grad_buffers_preserved(self):
        import random
        import torch
        from tools.hier_postrun_audit.backbone_probe import probe_model_batch
        from tools.hier_postrun_audit.extract import state_digest
        torch.manual_seed(121)
        model = self._model()
        model.train()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = state_digest(model.state_dict())
        grad_before = [parameter.grad.clone() for parameter in model.parameters()]
        cpu_rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        clouds, labels, ids = self._data()
        original = clouds.copy()
        report = probe_model_batch(model, clouds, labels, ids, whole_count=16,
                                   child_count=8, microbatch_size=2, seed=22)
        self.assertGreater(report["parameter_groups"]["shared_encoder"]["norms"]["base"], 0)
        self.assertFalse(report["hier_stats"]["hier_available"])
        self.assertEqual(state_digest(model.state_dict()), state_before)
        self.assertTrue(model.training)
        np.testing.assert_array_equal(clouds, original)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for parameter, original_grad in zip(model.parameters(), grad_before):
            self.assertTrue(torch.equal(parameter.grad, original_grad))
        self.assertEqual(report["read_only"]["optimizer_updates"], 0)

    def test_proxy_term_has_zero_backbone_gradient_while_sample_term_reaches_encoder(self):
        import torch
        from tools.hier_postrun_audit.backbone_probe import probe_model_batch
        torch.manual_seed(121)
        model = self._model()
        clouds, labels, ids = self._data()
        tangent = np.random.default_rng(29).normal(size=(8, 4)).astype(np.float32) * .2
        report = probe_model_batch(model, clouds, labels, ids, proxy_tangent=tangent,
                                  config={"lambda_hier": .1, "tau": 1., "margin": .5,
                                          "sample_K": 4, "proxy_K": 3, "t_per_anchor": 20},
                                  whole_count=16, child_count=8, microbatch_size=2)
        norms = report["parameter_groups"]["shared_encoder"]["norms"]
        self.assertEqual(norms["proxy"], 0.)
        self.assertGreater(norms["sample"], 0.)
        self.assertAlmostEqual(norms["hier_weighted"], .1 * norms["hier_unweighted"], places=6)
        self.assertEqual(report["whole_ball_partials"]["proxy"]["gradient_norm"], 0.)
        self.assertGreater(report["whole_ball_partials"]["ce"]["gradient_norm"], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_group_cosines_use_the_same_parameter_coordinates(self):
        import torch
        from tools.hier_postrun_audit.backbone_probe import gradient_comparison
        first = torch.tensor([1., 2., 3.])
        self.assertAlmostEqual(gradient_comparison(first, first)["cosine"], 1., places=6)
        self.assertAlmostEqual(gradient_comparison(first, -first)["cosine"], -1., places=6)
        self.assertIsNone(gradient_comparison(torch.zeros_like(first), first)["cosine"])

    def test_nonreentrant_checkpoint_preserves_all_loss_gradients_and_random_streams(self):
        import copy
        import random
        import torch
        from tools.hier_postrun_audit.backbone_probe import probe_model_batch
        from tools.hier_postrun_audit.extract import state_digest
        torch.manual_seed(71)
        model = self._model()
        # ModelNet FPS consumes random values even under eval mode. Exercise
        # that property with a small genuinely random differentiable forward.
        class StochasticModel(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.encoder = original.encoder
                self.bn = original.bn
                self.classifier = original.classifier

            def forward(self, points, emb=False):
                features = self.bn(self.encoder((points + .03 * torch.rand_like(points)).mean(dim=-1)))
                mu = .65 * torch.tanh(features)
                mu = mu * (.8 / mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
                return (mu, mu) if emb else (mu, self.classifier(mu))
        plain = StochasticModel(model)
        recomputed = copy.deepcopy(plain)
        clouds, labels, ids = self._data()
        tangent = np.random.default_rng(29).normal(size=(8, 4)).astype(np.float32) * .2
        kwargs = {"proxy_tangent": tangent,
                  "config": {"lambda_hier": .1, "tau": 1., "margin": .5,
                             "sample_K": 4, "proxy_K": 3, "t_per_anchor": 5},
                  "whole_count": 16, "child_count": 8, "microbatch_size": 2, "seed": 22}
        cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        original_state = state_digest(plain.state_dict())
        before = probe_model_batch(plain, clouds, labels, ids, activation_checkpoint=False, **kwargs)
        after = probe_model_batch(recomputed, clouds, labels, ids, activation_checkpoint=True, **kwargs)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(state_digest(plain.state_dict()), original_state)
        self.assertEqual(state_digest(recomputed.state_dict()), original_state)
        for name, value in before["objective_values"].items():
            self.assertAlmostEqual(value, after["objective_values"][name], places=6)
        for group, values in before["parameter_groups"].items():
            for name, value in values["norms"].items():
                self.assertAlmostEqual(value, after["parameter_groups"][group]["norms"][name], places=6)
            for comparison in ("hier_vs_base", "sample_vs_base", "total_vs_base"):
                for name, value in values.get(comparison, {}).items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(value, after["parameter_groups"][group][comparison][name], places=6)
        for name, values in before["whole_ball_partials"].items():
            self.assertAlmostEqual(values["gradient_norm"], after["whole_ball_partials"][name]["gradient_norm"], places=6)
            self.assertAlmostEqual(values["signed_radial_mean"], after["whole_ball_partials"][name]["signed_radial_mean"], places=6)
        self.assertTrue(all(parameter.grad is None for parameter in plain.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in recomputed.parameters()))

    def test_invalid_flip_negative_rejected(self):
        from tools.hier_postrun_audit.backbone_probe import probe_model_batch
        clouds, labels, ids = self._data()
        with self.assertRaisesRegex(ValueError, "flip negative"):
            probe_model_batch(self._model(), clouds, np.zeros_like(labels), ids,
                              whole_count=16, child_count=8, microbatch_size=2)


if __name__ == "__main__":
    unittest.main()
