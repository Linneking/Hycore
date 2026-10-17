"""CPU checks for optimizer radial accounting, including real PyTorch updates."""

import json
import unittest

import torch

try:
    from .optimizer_math import decompose_gradient, decompose_update, summarize
except ImportError:
    from optimizer_math import decompose_gradient, decompose_update, summarize


class OptimizerMathTests(unittest.TestCase):
    def test_gradient_pythagoras_and_sign(self):
        p = torch.tensor([[3., 4.], [2., 0.]], dtype=torch.float64)
        g = torch.tensor([[4., -3.], [1., 2.]], dtype=torch.float64)
        rec = decompose_gradient(p, g)
        torch.testing.assert_close(rec["gradient_norm"].square(),
                                   rec["signed_radial_gradient"].square()
                                   + rec["angular_gradient_norm"].square(),
                                   rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(float(rec["signed_radial_gradient"][0]), 0., places=14)
        self.assertEqual(float(rec["signed_radial_gradient"][1]), 1.)

    def test_tangential_sgd_finite_step_increases_norm(self):
        p = torch.nn.Parameter(torch.tensor([[3., 4.]], dtype=torch.float64))
        before = p.detach().clone()
        p.grad = torch.tensor([[4., -3.]], dtype=torch.float64)
        torch.optim.SGD([p], lr=.2).step()
        rec = decompose_update(before, p)
        self.assertAlmostEqual(float(rec["signed_radial_step"]), 0., places=14)
        self.assertGreater(float(rec["exact_norm_change"]), 0.)
        self.assertAlmostEqual(float(rec["after_norm"]), 26 ** .5, places=14)
        torch.testing.assert_close(rec["exact_norm_change"], rec["second_order_excess"],
                                   rtol=1e-14, atol=1e-14)

    def test_adamw_preconditioned_tangent_gradient_can_move_outward_first_order(self):
        p = torch.nn.Parameter(torch.tensor([[3., 4.]], dtype=torch.float64))
        before = p.detach().clone()
        p.grad = torch.tensor([[4., -3.]], dtype=torch.float64)
        raw = decompose_gradient(p, p.grad)
        self.assertAlmostEqual(float(raw["signed_radial_gradient"]), 0., places=14)
        # On its first step, coordinatewise Adam rescales (4,-3) to (~1,-1).
        # Gradient descent then moves (~-.1,+.1), outward relative to (3,4).
        torch.optim.AdamW([p], lr=.1, weight_decay=0.).step()
        rec = decompose_update(before, p)
        self.assertGreater(float(rec["signed_radial_step"]), .0199)
        self.assertGreater(float(rec["exact_norm_change"]), .02)
        self.assertGreater(float(rec["second_order_excess"]), 0.)

    def test_gradient_scalar_reduction_is_not_equal_adamw_step_reduction(self):
        start = torch.tensor([[3., 4.]], dtype=torch.float64)
        steps = []
        for scale in (1., .001):
            p = torch.nn.Parameter(start.clone())
            p.grad = torch.tensor([[4., -3.]], dtype=torch.float64) * scale
            torch.optim.AdamW([p], lr=.1, weight_decay=0., eps=1e-12).step()
            steps.append(p.detach() - start)
        torch.testing.assert_close(steps[0], steps[1], rtol=1e-9, atol=1e-10)

    def test_exact_accounting_on_stored_float32_updates(self):
        generator = torch.Generator().manual_seed(193)
        before = torch.randn(512, 256, generator=generator, dtype=torch.float32)
        after = before + torch.randn(512, 256, generator=generator) * 1e-4
        saved_before, saved_after = before.clone(), after.clone()
        rec = decompose_update(before, after)
        torch.testing.assert_close(rec["exact_norm_change"],
                                   rec["signed_radial_step"] + rec["second_order_excess"],
                                   rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(rec["squared_norm_change"],
                                   rec["squared_first_order_radial"] + rec["squared_step"],
                                   rtol=1e-8, atol=2e-12)
        self.assertGreaterEqual(float(rec["second_order_excess"].min()), -1e-14)
        torch.testing.assert_close(before, saved_before, rtol=0, atol=0)
        torch.testing.assert_close(after, saved_after, rtol=0, atol=0)
        self.assertFalse(any(value.requires_grad for value in rec.values()))

    def test_real_weight_decay_is_included(self):
        p = torch.nn.Parameter(torch.tensor([[3., 4.]], dtype=torch.float64))
        before = p.detach().clone()
        p.grad = torch.zeros_like(p)
        torch.optim.AdamW([p], lr=.1, weight_decay=.2).step()
        rec = decompose_update(before, p)
        self.assertAlmostEqual(float(rec["signed_radial_step"]), -.1, places=14)
        self.assertAlmostEqual(float(rec["exact_norm_change"]), -.1, places=14)
        self.assertAlmostEqual(float(rec["second_order_excess"]), 0., places=14)

    def test_zero_vectors_are_flagged_and_finite(self):
        zero = torch.zeros(1, 2)
        rec = decompose_gradient(zero, torch.tensor([[3., 4.]]))
        self.assertFalse(bool(rec["direction_defined"][0]))
        self.assertEqual(float(rec["angular_gradient_norm"][0]), 5.)
        update = decompose_update(zero, zero)
        self.assertEqual(float(update["signed_radial_step_fraction"][0]), 0.)
        self.assertTrue(all(bool(torch.isfinite(item).all()) for item in update.values()))

    def test_summary_population_std_and_nonfinites(self):
        rec = summarize([1., 2., 3., float("nan"), float("inf")], quantiles=(0, .5, 1))
        self.assertEqual(rec["count"], 5)
        self.assertEqual(rec["finite_count"], 3)
        self.assertAlmostEqual(rec["std"], (2 / 3) ** .5)
        self.assertEqual(rec["quantiles"], {"q0": 1., "q0.5": 2., "q1": 3.})
        json.dumps(rec, allow_nan=False)
        empty = summarize([])
        self.assertIsNone(empty["mean"])
        self.assertEqual(empty["count"], 0)
        json.dumps(empty, allow_nan=False)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            decompose_update(torch.zeros(2, 3), torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            decompose_gradient(torch.tensor([[float("nan")]]), torch.ones(1, 1))
        with self.assertRaises(TypeError):
            decompose_gradient(torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2))
        with self.assertRaises(ValueError):
            summarize([1], quantiles=(1.1,))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
