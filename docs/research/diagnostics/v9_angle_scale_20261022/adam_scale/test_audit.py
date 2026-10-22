"""Synthetic checks for fixed-state scaling semantics; no experimental data."""
import unittest
import torch
from audit import analyze_snapshot, clone_step


def state(step=0):
    p = torch.tensor([[2.1, .2], [2.2, -.1]], dtype=torch.float64)
    g = torch.tensor([[-1e-5, 2e-5], [-2e-5, 1e-5]], dtype=torch.float64)
    m = torch.full_like(p, -2e-5) if step else torch.zeros_like(p)
    v = torch.full_like(p, 2e-10) if step else torch.zeros_like(p)
    return dict(parameter=p, gradient=g, exp_avg=m, exp_avg_sq=v, step_before=step,
                lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)


class TestAudit(unittest.TestCase):
    def test_whole_history_scaling_equivalent_to_inverse_eps_scaling(self):
        s = state(20)
        a = .2
        actual = clone_step(s, current_scale=a, history_scale=a)
        equivalent = clone_step(s, eps=s["eps"] / a)
        torch.testing.assert_close(actual, equivalent, rtol=0, atol=1e-14)

    def test_first_step_current_and_whole_history_match(self):
        s = state(0)
        torch.testing.assert_close(clone_step(s, current_scale=.2),
                                   clone_step(s, current_scale=.2, history_scale=.2))

    def test_current_only_is_different_after_history(self):
        s = state(20)
        self.assertGreater(float((clone_step(s, current_scale=.2) -
                                  clone_step(s, current_scale=.2, history_scale=.2)).abs().max()), 1e-5)

    def test_larger_eps_reduces_adaptive_step(self):
        s = state(0)
        decay_only = s["parameter"] * (1 - s["lr"] * s["weight_decay"])
        native = (clone_step(s) - decay_only).norm()
        changed = (clone_step(s, eps=1e-4) - decay_only).norm()
        self.assertLess(float(changed), float(native) * .2)

    def test_native_replay_and_no_mutation(self):
        s = state(20)
        originals = {key: value.clone() for key, value in s.items() if torch.is_tensor(value)}
        s["parameter_after"] = clone_step(s)
        summary = analyze_snapshot(dict(actual_optimizer_step=s, epoch=1, step=21))
        self.assertEqual(summary["native_replay_max_coordinate_error"], 0.)
        self.assertEqual(summary["variants"]["native"]["all"]["count"], 2)
        for key, original in originals.items():
            torch.testing.assert_close(s[key], original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
