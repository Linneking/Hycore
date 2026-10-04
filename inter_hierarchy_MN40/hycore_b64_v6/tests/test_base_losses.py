"""CPU fidelity gates for the real B64 entry point and released criterion."""

import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "Torch is unavailable locally; run this CPU gate on the server")
class SourceBaseLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual entry point, including its production dependencies.
        # No model is constructed, process group initialized or CUDA call made.
        from inter_hierarchy_MN40.hycore_b64_v6 import train
        from classification_ModelNet40 import hutil
        cls.train, cls.source = train, hutil

    def actual_loss(self, mu, nu, logits, gold):
        outputs = {"mu": mu, "nu": nu, "logits": logits,
                   "whole_count": 900, "child_count": 300}
        with mock.patch.object(self.train, "gather_with_grad", side_effect=lambda value: value), \
                mock.patch.object(self.train, "gather_without_grad", side_effect=lambda value: value):
            return self.train.global_source_losses(outputs, gold)

    def source_loss(self, mu, nu, logits, gold):
        components = self.source.hype_triplet_losses(
            mu, nu, hier_margin=1000. / 300, contr_margin=4., ball_dim=256)
        ce = self.source.cal_loss(logits, gold)
        return ce + .01 * components[4] + .01 * components[5], ce, components

    def test_global64_source_values_gradients_and_same_class_flip(self):
        rng = torch.Generator(device="cpu").manual_seed(22)
        mu = torch.randn(64, 256, generator=rng) * .002
        nu = torch.randn(64, 256, generator=rng) * .002
        # Distinct rank halves make global flip observably different from
        # concatenating two local flips, while keeping every point in c1.
        mu[:32, 0] += .25
        mu[32:, 0] -= .25
        nu[:32, 0] += .15
        nu[32:, 0] -= .15
        mu.requires_grad_()
        nu.requires_grad_()
        logits = torch.randn(64, 40, generator=rng, requires_grad=True)
        gold = torch.zeros(64, dtype=torch.long)  # All same-class negatives are allowed.
        actual = self.actual_loss(mu, nu, logits, gold)
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in (mu, nu, logits))
        expected, expected_ce, components = self.source_loss(*reference_inputs, gold)
        for key, reference in zip(
                ("whole_radius", "part_radius", "positive_distance", "negative_distance",
                 "intra_contrastive", "intra_radial"), components):
            torch.testing.assert_close(actual[key], reference, rtol=0, atol=0)
        torch.testing.assert_close(actual["ce"], expected_ce, rtol=0, atol=0)
        torch.testing.assert_close(actual["base"], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual["gold"], gold))
        actual_gradients = torch.autograd.grad(actual["base"], (mu, nu, logits))
        expected_gradients = torch.autograd.grad(expected, reference_inputs)
        for name, gradient, reference in zip(("mu", "nu", "logits"), actual_gradients, expected_gradients):
            with self.subTest(gradient=name):
                self.assertEqual(gradient.device.type, "cpu")
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 0.)
                torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        ball = self.source.PoincareBall(c=1., dim=256)
        global_negative = ball.dist(mu, nu.flip(0)).mean()
        local_flip = torch.cat((nu[:32].flip(0), nu[32:].flip(0)))
        local_negative = ball.dist(mu, local_flip).mean()
        torch.testing.assert_close(actual["negative_distance"], global_negative, rtol=0, atol=0)
        self.assertGreater(float((global_negative - local_negative).abs()), .1)

    def test_zero_hinge_tie_preserves_source_half_gradient(self):
        class TieBall:
            """Controlled distances isolate max's exact-zero gradient convention."""
            def __init__(self, **_kwargs):
                pass

            def dist(self, x, y):
                return x[:, 0] + y[:, 0]

            def dist0(self, x):
                return x[:, 1]

        mu = torch.zeros(64, 256, dtype=torch.float64)
        nu = torch.zeros_like(mu)
        mu[:, 1] = 1000. / 300  # Every radial hinge is exactly zero.
        nu[:32, 0], nu[32:, 0] = -2., 2.  # First 32 contrastive hinges are exactly zero.
        mu.requires_grad_()
        nu.requires_grad_()
        logits = torch.zeros(64, 40, dtype=torch.float64, requires_grad=True)
        gold = torch.zeros(64, dtype=torch.long)
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in (mu, nu, logits))
        with mock.patch.dict(self.train.hype_triplet_losses.__globals__, {"PoincareBall": TieBall}), \
                mock.patch.object(self.train, "PoincareBall", TieBall), \
                mock.patch.object(self.source, "PoincareBall", TieBall):
            actual = self.actual_loss(mu, nu, logits, gold)
            expected, _ce, components = self.source_loss(*reference_inputs, gold)
            self.assertEqual(float(actual["intra_radial"]), 0.)
            self.assertEqual(float(actual["intra_contrastive"]), 4.)
            torch.testing.assert_close(actual["base"], expected, rtol=0, atol=0)
            gradients = torch.autograd.grad(actual["base"], (mu, nu, logits))
            references = torch.autograd.grad(expected, reference_inputs)
        for gradient, reference in zip(gradients, references):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        half_weight = .01 / (2 * 64)
        torch.testing.assert_close(gradients[0][:, 1], torch.full((64,), -half_weight, dtype=mu.dtype),
                                   rtol=0, atol=0)
        torch.testing.assert_close(gradients[1][:, 1], torch.full((64,), half_weight, dtype=nu.dtype),
                                   rtol=0, atol=0)
        torch.testing.assert_close(gradients[1][:32, 0], torch.full((32,), -half_weight, dtype=nu.dtype),
                                   rtol=0, atol=0)
        torch.testing.assert_close(gradients[1][32:, 0], torch.full((32,), half_weight, dtype=nu.dtype),
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
