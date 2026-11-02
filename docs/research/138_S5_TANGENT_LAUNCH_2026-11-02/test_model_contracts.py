"""CPU-only contracts for the single approved S5 tangent-readout change."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v9.model import NativeTail2 as Original
from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2 as Tangent
from segmentation_ShapeNetPart.hycore_seg_v10.monitoring import GeometryMonitor


def paired_models():
    torch.manual_seed(4780)
    old = Original("h", k=4)
    state = torch.get_rng_state().clone()
    torch.manual_seed(4780)
    new = Tangent("h", k=4)
    return old, new, state


class TangentModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise RuntimeError("Contracts must run without CUDA initialization")

    def test_constructor_preserves_every_tensor_and_rng(self):
        old, new, rng = paired_models()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        self.assertTrue(torch.equal(old.mlr.weight, new.mlr.weight))
        self.assertEqual(type(old.dp1), type(new.dp1))
        with self.assertRaises(ValueError):
            Tangent("g", k=4)

    def test_readout_formula_and_same_fixed_weight_predictions(self):
        old, new, _ = paired_models()
        old, new = old.double().eval(), new.double().eval()
        generator = torch.Generator().manual_seed(90)
        # Include weak features and features whose input EXP reaches projection.
        features = torch.randn(2, 16, 256, generator=generator, dtype=torch.float64)
        for magnitude in (.002, .2, 10.):
            h = features * magnitude
            with torch.no_grad():
                ball = new.emb2(new.manifold.expmap0(h))
                expected = F.linear(new.manifold.logmap0(ball), new.mlr.weight)
                actual = new.forward_head(h)
                previous = old.forward_head(h)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            mismatch = previous.argmax(-1) != actual.argmax(-1)
            # Only numerical near-ties could legitimately change the ranking.
            margin = actual.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
            self.assertFalse(bool((mismatch & (margin > 1e-10)).any()))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.isfinite(previous).all())
            self.assertTrue(torch.equal(old.mlr._weight, new.mlr._weight))
            self.assertTrue(torch.equal(old.mlr._bias, new.mlr._bias))

    def test_ce_gradient_and_native_parameter_connectivity(self):
        _, model, _ = paired_models()
        h = (torch.randn(2, 8, 256) * .03).requires_grad_()
        targets = torch.arange(16).reshape(2, 8)
        scores = model.forward_head(h)
        loss = F.cross_entropy((5 * scores).reshape(-1, 50), targets.reshape(-1)) / 5
        score_gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
        expected_gradient = ((5 * scores).softmax(-1) - F.one_hot(targets, 50)) / targets.numel()
        torch.testing.assert_close(score_gradient, expected_gradient)
        parameters = (h, model.emb2._weight, model.emb2._bias, model.mlr._weight, model.mlr._bias)
        gradients = torch.autograd.grad(loss, parameters)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.)

    def test_monitor_real_scores_and_probability_statistics_without_side_effects(self):
        _, model, _ = paired_models()
        monitor = GeometryMonitor()
        model.monitor = monitor
        h = (torch.randn(2, 8, 256) * .03).requires_grad_()
        targets = torch.arange(16).reshape(2, 8)
        monitor.begin(True, epoch=1, step=1)
        scores = model.forward_head(h)
        monitor.capture_targets(targets, scale=5.)
        rng = torch.get_rng_state().clone()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        report = monitor.summarize()
        q = report["geometry"]["final_tangent"]
        self.assertNotIn("m3", report["geometry"])
        for key in ("radius", "depth", "near_boundary_fraction", "projection_radius"):
            self.assertNotIn(key, q)
        for key in ("point_exp", "m2"):
            self.assertIn("near_boundary_fraction", report["geometry"][key])
        probabilities = (5 * scores.detach()).softmax(-1)
        self.assertAlmostEqual(q["scaled_max_class_probability"]["mean"],
                               float(probabilities.amax(-1).double().mean()), places=10)
        self.assertAlmostEqual(q["scaled_correct_class_probability"]["mean"],
                               float(probabilities.gather(-1, targets.unsqueeze(-1)).double().mean()), places=10)
        self.assertEqual(q["coordinate_space"], "origin_tangent_real_scores")
        loss = F.cross_entropy((5 * scores).reshape(-1, 50), targets.reshape(-1)) / 5
        gradients = monitor.gradient_summary({"ce": loss})
        self.assertEqual(gradients["gradients"]["ce"]["nodes"]["final_tangent"]["status"], "nonzero")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        monitor.clear()
        self.assertIsNone(monitor._targets)
        self.assertFalse(monitor.nodes)
        monitor.begin(False)
        monitor.capture_targets(targets)
        self.assertIsNone(monitor._targets)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
