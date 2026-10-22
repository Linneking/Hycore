"""Small frozen-graph checks; no GPU allocation or training is required."""
import copy
import unittest

import torch

import replay_routing as replay


class ExactRoutingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, component):
        rng = torch.Generator().manual_seed(119)
        p = (torch.randn(8, 4, generator=rng) * .8).requires_grad_(True)
        mu = replay.to_ball(torch.randn(6, 4, generator=rng), hook=False).detach()
        q = replay.to_ball(p)
        distances = replay.official_distance(mu if component == "sample" else q, q)
        n = 6 if component == "sample" else 8
        triplets = torch.tensor([[i, (i + 1) % n, (i + 3) % n] for i in range(n)] * 8)
        state = rng.get_state().clone()
        loss, stats, details = replay.ghhc_loss(distances, triplets, generator=rng, return_details=True)
        gradient, = torch.autograd.grad(.5 * loss, p)
        recorded = {"generator_before": state, "distances": distances.detach(),
            "triplets": triplets, "tau": .1, "margin": .1, "loss": loss.detach(), "stats": stats,
            "details": {key: value.detach() for key, value in details.items()}}
        return p.detach().clone().requires_grad_(True), mu if component == "sample" else None, recorded, gradient

    def test_exact_replay_both_endpoints_and_routing_identity(self):
        for name in ("sample", "proxy"):
            p, mu, recorded, observed = self.fixture(name)
            result, gradients = replay.replay_component(p, mu, recorded, observed, .5, torch.device("cpu"))
            self.assertTrue(result["validation_passed"], result["failures"])
            self.assertTrue(result["full_gradient_vs_observed"]["bitwise_equal"])
            self.assertTrue(result["distances"]["bitwise_equal"])
            self.assertEqual(result["decomposition_max_abs_error"], 0.)
            self.assertTrue(all(v == 0 for v in result["exact_choice_checks"].values()))
            torch.testing.assert_close(gradients["full_st"], gradients["fixed_choice"] + gradients["routing"])

    def test_mismatched_hard_choices_fail_closed(self):
        p, mu, recorded, observed = self.fixture("sample")
        changed = copy.deepcopy(recorded)
        changed["details"]["pair_proxy_idx"][0] = (changed["details"]["pair_proxy_idx"][0] + 1) % len(p)
        result, _ = replay.replay_component(p, mu, changed, observed, .5, torch.device("cpu"))
        self.assertFalse(result["validation_passed"])
        self.assertIn("pair_proxy_idx_mismatches", result["failures"])

    def test_observed_gradient_mismatch_fails_and_reports_relative_error(self):
        p, mu, recorded, observed = self.fixture("proxy")
        result, _ = replay.replay_component(p, mu, recorded, observed + .01, .5, torch.device("cpu"))
        self.assertFalse(result["validation_passed"])
        self.assertIn("full_gradient_absolute_error", result["failures"])
        self.assertGreater(result["full_gradient_vs_observed"]["relative_l2_error"], .1)

    def test_reviewed_source_guard(self):
        self.assertEqual(set(replay.source_guard()), set(replay.EXPECTED_SOURCE))


if __name__ == "__main__":
    unittest.main()
