"""Full train pool controls and opportunistic allocation contracts."""
from argparse import Namespace
import unittest
import torch
from ..controls import ShuffleBatchSampler, fixed_configuration, control_base_losses
from ..train import fixed_configuration as main_configuration
from ..controls_queue import plan_allocations
from ...hier_proxy_scratch_v8.controls import control_base_losses as prior_base_losses


def args(control):
    return Namespace(control=control, seed=22, epochs=300,
                     steps_per_epoch=200 if control == "B64" else 307,
                     workers=2, eval_batch_size=32, train_eval_every=10,
                     audit_epochs=(), smoke=False, smoke_eval_batches=1,
                     warmup_epochs=0, lambda_hier=0.)


class FullControls(unittest.TestCase):
    def test_protocol_geometry_matches_fulltrain_main_and_has_no_hier(self):
        main = main_configuration(args("B64"))
        keys = ("c", "D", "clip_r", "final_depth_cap", "HIER_backward_hook", "hook_scope",
                "radial_gamma", "contrastive_margin", "model_optimizer", "BN",
                "part_overwrites_whole", "validation", "selection", "training_pool")
        for c in ("B64", "B0-32"):
            config = fixed_configuration(args(c))
            self.assertEqual({k: config[k] for k in keys}, {k: main[k] for k in keys})
            self.assertFalse(config["HIER_objective_enabled"])
            self.assertEqual(config["HIER_calls_per_step"], 0)
            self.assertEqual(config["P"], 0)
            self.assertIsNone(config["proxy_optimizer"])
            self.assertIn("FULL9840-TESTSEL", config["canonical_name"])

    def test_full_shuffle_epoch_visits9824_distinct_objects_and_drops16(self):
        sampler = ShuffleBatchSampler(9840, 22)
        flat = sum(list(sampler), [])
        self.assertEqual(len(sampler), 307)
        self.assertEqual(len(flat), 9824)
        self.assertEqual(len(set(flat)), 9824)
        self.assertEqual(len(set(range(9840)) - set(flat)), 16)
        sampler.set_epoch(1)
        self.assertNotEqual(flat, sum(list(sampler), []))

    def test_preserved_shuffle_loss_and_gradient_matches_prior_v8(self):
        torch.manual_seed(22)
        mu = (torch.randn(32, 8, dtype=torch.float64) * .2).requires_grad_()
        nu = (torch.randn(32, 8, dtype=torch.float64) * .2).requires_grad_()
        logits = torch.randn(32, 40, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(32) % 5
        outputs = dict(mu=mu, nu=nu, logits=logits, child_count=400)
        new = control_base_losses(outputs, labels, "B0-32")["base"]
        old = prior_base_losses(outputs, labels, "B0-32")["base"]
        torch.testing.assert_close(new, old, atol=0, rtol=0)
        for a, b in zip(torch.autograd.grad(new, (mu, nu, logits), retain_graph=True),
                        torch.autograd.grad(old, (mu, nu, logits))):
            # Parallel gradient accumulation may differ in its final floating-point bits.
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)

    def test_one_free_gpu_runs_shuffle_without_waiting_for_b64(self):
        free = [{"uuid": "a"}]
        self.assertEqual(plan_allocations(["B64", "B0-32"], free), [("B0-32", free)])
        self.assertEqual(plan_allocations(["B64"], free), [])

    def test_two_free_gpus_start_b64_first_and_three_allow_both(self):
        free = [{"uuid": k} for k in ("a", "b", "c")]
        self.assertEqual(plan_allocations(["B0-32", "B64"], free[:2]), [("B64", free[:2])])
        self.assertEqual(plan_allocations(["B64", "B0-32"], free),
                         [("B64", free[:2]), ("B0-32", free[2:])])

    def test_occupied_or_reserved_devices_are_not_allocated(self):
        self.assertEqual(plan_allocations(["B64", "B0-32"], []), [])
        self.assertEqual(plan_allocations([], [{"uuid": "a"}]), [])


if __name__ == "__main__":
    unittest.main()
