"""CPU invariants for the read-only real-step observer."""
import copy
import unittest

import torch

from optimizer_step_audit import before_step, after_step


class AdamStepAuditTests(unittest.TestCase):
    def test_real_adamw_reconstruction_and_readonly_across_steps(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(22)
        parameter = torch.nn.Parameter(torch.randn(512, 256, generator=generator) * .13)
        optimizer = torch.optim.AdamW([parameter], lr=.01, eps=1e-8, weight_decay=.01)
        worst = 0.
        for iteration in range(24):
            parameter.grad = torch.randn(512, 256, generator=generator) * 10 ** (-iteration % 7 - 7)
            before = before_step(parameter, optimizer)
            p_before = parameter.detach().clone()
            g_before = parameter.grad.clone()
            # The observer has not initialized or mutated real optimizer state.
            self.assertEqual(int(before["step_before"]), iteration)
            torch.testing.assert_close(parameter, p_before, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, g_before, rtol=0, atol=0)
            optimizer.step()
            saved_p = parameter.detach().clone()
            saved_state = copy.deepcopy(optimizer.state_dict())
            report, private = after_step(before, parameter, optimizer, include_snapshot=True)
            worst = max(worst, report["all"]["prediction_error_max_coordinate"]["max"])
            self.assertEqual(report["state_invariants"]["step_after"], iteration + 1)
            self.assertLess(report["state_invariants"]["squared_norm_identity_max_error"], 1e-12)
            self.assertIn("exp_avg_after", private)
            self.assertIn("_per_proxy", report)
            torch.testing.assert_close(parameter, saved_p, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state_dict()["state"][0][key],
                                           saved_state["state"][0][key], rtol=0, atol=0)
        print("float32 worst predicted-vs-real max coordinate error:", worst)
        self.assertLess(worst, 2e-7)

    def test_zero_raw_radial_gradient_has_outward_adam_update(self):
        parameter = torch.nn.Parameter(torch.tensor([[3., 4.]], dtype=torch.float64))
        parameter.grad = torch.tensor([[4., -3.]], dtype=torch.float64)
        optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=0., eps=1e-8)
        before = before_step(parameter, optimizer)
        optimizer.step()
        report, snapshot = after_step(before, parameter, optimizer)
        self.assertIsNone(snapshot)
        self.assertAlmostEqual(report["all"]["signed_radial_gradient"]["mean"], 0.)
        self.assertGreater(report["all"]["current_tangent_g_step"]["mean"], .0199)
        self.assertGreater(report["all"]["exact_norm_change"]["mean"], .02)
        self.assertLess(report["all"]["prediction_error_norm"]["max"], 1e-15)


if __name__ == "__main__":
    unittest.main()
