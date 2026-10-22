"""CPU counterfactual tests against actual AdamW state and anonymous schema."""
import copy
import json
import unittest

import torch

from counterfactuals import analyze_snapshot, one_step, selected_steps, summarize_update
from optimizer_step_audit import before_step, after_step


class CounterfactualTests(unittest.TestCase):
    def test_replay_actual_step_and_preserve_saved_moments(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(83)
        p = torch.nn.Parameter(torch.randn(32, 8, generator=generator) * .8)
        optimizer = torch.optim.AdamW([p], lr=.01, eps=1e-8, weight_decay=.01)
        for _ in range(7):
            p.grad = torch.randn(32, 8, generator=generator) * 1e-5
            optimizer.step()
        p.grad = torch.randn(32, 8, generator=generator) * 1e-5
        before = before_step(p, optimizer)
        optimizer.step()
        _, snapshot = after_step(before, p, optimizer, include_snapshot=True)
        original = copy.deepcopy(snapshot)
        result = analyze_snapshot({"epoch": 1, "step": 8, "actual_optimizer_step": snapshot})
        self.assertEqual(result["baseline_vs_recorded_actual"]["max_coordinate_error"], 0.)
        self.assertEqual(len(result["variants"]), 4)
        self.assertIn("history_m_zero_keep_v_and_step", result["variants"])
        # All output data are aggregates with no parameter or ID arrays.
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("exp_avg", serialized)
        self.assertNotIn("parameter_after", serialized)
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(snapshot[key], value, rtol=0, atol=0)
        for key in result["variants"]:
            self.assertEqual(result["variants"][key]["all"]["count"], 32)

    def test_pure_historical_radial_m_removal_turns_outward_into_decay(self):
        p = torch.tensor([[3., 4.]], dtype=torch.float64)
        snapshot = {"parameter": p, "gradient": torch.zeros_like(p),
                    "exp_avg": -.01 * p, "exp_avg_sq": torch.ones_like(p),
                    "step_before": 3, "lr": .1, "betas": (.9, .999),
                    "eps": 1e-8, "weight_decay": .0001}
        original_m = snapshot["exp_avg"].clone()
        baseline = summarize_update(p, one_step(snapshot, "native"))
        tangent = summarize_update(p, one_step(snapshot, "history_m_radial_removed"))
        zero = summarize_update(p, one_step(snapshot, "history_m_zero_keep_v_and_step"))
        self.assertGreater(baseline["all"]["mean_exact_norm_change"], 0.)
        self.assertLess(tangent["all"]["mean_exact_norm_change"], 0.)
        self.assertAlmostEqual(tangent["all"]["mean_exact_norm_change"],
                               zero["all"]["mean_exact_norm_change"], places=14)
        torch.testing.assert_close(snapshot["exp_avg"], original_m, rtol=0, atol=0)

    def test_reject_mismatch_and_parse_defaults(self):
        self.assertEqual(selected_steps("e1s1,e005_s200"), [(1, 1), (5, 200)])
        with self.assertRaises(ValueError):
            selected_steps("e1s1,e001_s001")
        with self.assertRaises(ValueError):
            selected_steps("../private")


if __name__ == "__main__":
    unittest.main()
