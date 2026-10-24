"""Real scripted Geoopt projection regression for non-reentrant checkpointing."""
from __future__ import annotations

import copy
import unittest

import geoopt
import torch
from torch import nn

from inter_hierarchy_MN40.parthier_v1.base_protocol import HyCoReTrainingForward


class ScriptedGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=1.)
        self.weight = nn.Parameter(torch.eye(4) * .7)
        self.classifier = nn.Parameter(torch.randn(3, 4) * .1)

    def forward(self, points, emb=False):
        mu = self.ball.mobius_matvec(self.weight, points)
        return (mu, mu) if emb else (mu, self.ball.mobius_matvec(self.classifier, mu))


class FourPassScriptedGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=1.)
        self.encoder = nn.Linear(3, 6)
        self.bn = nn.BatchNorm1d(6)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Parameter(torch.randn(5, 6) * .1)
        self.started = 0
        self.completed = 0

    def forward(self, points, emb=False):
        self.started += 1
        tangent = self.dropout(torch.tanh(self.bn(self.encoder(points.mean(-1))))) * .1
        mu = self.ball.expmap0(tangent)
        logits = mu if emb else self.ball.mobius_matvec(self.classifier, mu)
        self.completed += 1
        return mu, logits


class ScriptedGeooptCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def compare_gradients(self, normal, checked):
        for (name, parameter), (other_name, other) in zip(normal.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(other.grad)
                self.assertTrue(bool(torch.isfinite(other.grad).all()))
                torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0)

    def test_real_geoopt_mobius_projection_backward_supports_no_grad_inputs(self):
        # On affected Torch/Geoopt versions, a direct non-reentrant checkpoint
        # with default early_stop=True raises a blank TorchScript RuntimeError
        # during _project recomputation. The production helper must complete it.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(4780)
                normal = ScriptedGeometry().to(dtype=dtype)
                checked = copy.deepcopy(normal)
                points = torch.randn(8, 4, dtype=dtype) * .03
                self.assertFalse(points.requires_grad)
                expected = HyCoReTrainingForward(normal)._backbone_forward(points)
                actual = HyCoReTrainingForward(checked, activation_checkpoint=True)._backbone_forward(points)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                sum(value.square().sum() for value in expected).backward()
                sum(value.square().sum() for value in actual).backward()
                self.compare_gradients(normal, checked)

    def test_four_complete_scripted_passes_preserve_gradients_bn_and_rng(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(4780)
                normal = FourPassScriptedGeometry().to(dtype=dtype).train()
                checked = copy.deepcopy(normal)
                points = torch.randn(4, 820, 3, dtype=dtype)
                crops = {"centers": [799, 403, 17, 0], "whole_centers": [0, 17, 300, 819]}
                start_rng = torch.get_rng_state().clone()
                expected = HyCoReTrainingForward(normal)(points, 800, **crops)
                sum(expected[key].square().sum() for key in ("mu", "parts", "logits")).backward()
                final_rng = torch.get_rng_state().clone()
                torch.set_rng_state(start_rng)
                actual = HyCoReTrainingForward(checked, activation_checkpoint=True)(points, 800, **crops)
                sum(actual[key].square().sum() for key in ("mu", "parts", "logits")).backward()
                for key in ("mu", "parts", "logits"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.compare_gradients(normal, checked)
                for (name, value), (other_name, other) in zip(normal.named_buffers(), checked.named_buffers()):
                    self.assertEqual(name, other_name)
                    torch.testing.assert_close(other, value, rtol=0, atol=0)
                self.assertEqual(int(checked.bn.num_batches_tracked), 4)
                self.assertEqual((normal.started, normal.completed), (4, 4))
                self.assertEqual((checked.started, checked.completed), (8, 8))
                self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
