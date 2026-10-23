"""Actual four-pass CPU equivalence for checkpoint/BN/Dropout and live weights."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from inter_hierarchy_MN40.parthier_v1.base_protocol import HyCoReTrainingForward, _PreserveBatchNormBuffers


class CheckpointBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8, 1)
        self.bn1 = nn.BatchNorm1d(8, momentum=.1)
        self.conv2 = nn.Conv1d(8, 12, 1)
        self.bn2 = nn.BatchNorm1d(12, momentum=.3)
        self.drop1 = nn.Dropout(.35)
        self.linear1 = nn.Linear(12, 7)
        self.bn3 = nn.BatchNorm1d(7, momentum=None)
        self.drop2 = nn.Dropout(.45)
        self.classifier = nn.Linear(7, 5)
        self.calls = 0

    def forward(self, points, emb=False):
        self.calls += 1
        x = torch.relu(self.bn1(self.conv1(points)))
        x = self.drop1(torch.nn.functional.silu(self.bn2(self.conv2(x))))
        mu = self.drop2(torch.relu(self.bn3(self.linear1(x.mean(-1)))))
        return (mu, mu) if emb else (mu, self.classifier(mu))


class ActivationCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def buffers(model):
        return {name: value.detach().clone() for name, value in model.named_buffers()}

    def fixture(self, dtype=torch.float32, seed=4780):
        torch.manual_seed(seed)
        original = CheckpointBackbone().to(dtype=dtype).train()
        checked = copy.deepcopy(original)
        points = torch.randn(4, 820, 3, dtype=dtype)
        probes = {"mu": torch.randn(4, 7, dtype=dtype),
                  "parts": torch.randn(4, 3, 7, dtype=dtype),
                  "logits": torch.randn(4, 5, dtype=dtype)}
        crops = {"whole_centers": [0, 17, 300, 819], "centers": [799, 403, 17, 0]}
        self.assertFalse(points.requires_grad)
        return original, checked, points, probes, crops

    @staticmethod
    def objective(values, probes):
        return sum((values[key] * probe).sum() + .07 * values[key].square().mean()
                   for key, probe in probes.items())

    def compare_buffers(self, reference, actual):
        self.assertEqual(reference.keys(), actual.keys())
        for name in reference:
            torch.testing.assert_close(reference[name], actual[name], rtol=0, atol=0)
            if name.endswith("num_batches_tracked"):
                self.assertEqual(int(actual[name]), 4)

    def test_four_pass_outputs_gradients_bn_and_rng_match_without_input_grad(self):
        for dtype in (torch.float32, torch.float64):
            for seed in (4780, 22):
                with self.subTest(dtype=dtype, seed=seed):
                    normal, checked, points, probes, crops = self.fixture(dtype, seed)
                    normal_wrapper = HyCoReTrainingForward(normal)
                    checkpoint_wrapper = HyCoReTrainingForward(checked, activation_checkpoint=True)
                    initial_rng = torch.get_rng_state().clone()
                    expected = normal_wrapper(points, 800, **crops)
                    normal_buffers = self.buffers(normal)
                    self.objective(expected, probes).backward()
                    final_rng = torch.get_rng_state().clone()

                    torch.set_rng_state(initial_rng)
                    actual = checkpoint_wrapper(points, 800, **crops)
                    for key in probes:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    self.compare_buffers(normal_buffers, self.buffers(checked))
                    self.objective(actual, probes).backward()
                    self.compare_buffers(normal_buffers, self.buffers(checked))
                    self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
                    for parameter, counterpart in zip(normal.parameters(), checked.parameters()):
                        self.assertIsNotNone(parameter.grad)
                        self.assertIsNotNone(counterpart.grad)
                        torch.testing.assert_close(counterpart.grad, parameter.grad, rtol=0, atol=0)
                    self.assertEqual(normal.calls, 4)
                    self.assertEqual(checked.calls, 8)
                    self.assertTrue(actual["input_stats"]["activation_checkpoint"])

    def test_same_graph_multiple_autograd_grad_reenters_context_and_restores_bn(self):
        normal, checked, points, probes, crops = self.fixture()
        initial_rng = torch.get_rng_state().clone()
        expected = HyCoReTrainingForward(normal)(points, 800, **crops)
        torch.set_rng_state(initial_rng)
        actual = HyCoReTrainingForward(checked, activation_checkpoint=True)(points, 800, **crops)
        normal_buffers = self.buffers(normal)
        normal_loss, actual_loss = self.objective(expected, probes), self.objective(actual, probes)
        final_rng = torch.get_rng_state().clone()
        for _ in range(2):
            reference = torch.autograd.grad(normal_loss, tuple(normal.parameters()), retain_graph=True)
            gradients = torch.autograd.grad(actual_loss, tuple(checked.parameters()), retain_graph=True)
            for a, b in zip(gradients, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.compare_buffers(normal_buffers, self.buffers(checked))
            self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
        self.assertEqual(checked.calls, 12)

    def test_recompute_context_restores_original_buffer_objects_on_exception_and_reuse(self):
        model = CheckpointBackbone()
        context = _PreserveBatchNormBuffers(model)
        originals = dict(model.named_buffers())
        values = self.buffers(model)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with context:
                    with torch.no_grad():
                        for value in model.buffers():
                            value.add_(1)
                    for name, value in originals.items():
                        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
                    outer_buffers = dict(model.named_buffers())
                    outer_values = self.buffers(model)
                    with context:
                        with torch.no_grad():
                            for value in model.buffers():
                                value.add_(2)
                    for name, value in model.named_buffers():
                        self.assertIs(value, outer_buffers[name])
                        torch.testing.assert_close(value, outer_values[name], rtol=0, atol=0)
                    raise RuntimeError("probe")
            for name, value in model.named_buffers():
                self.assertIs(value, originals[name])
                torch.testing.assert_close(value, values[name], rtol=0, atol=0)
        self.assertEqual(context._snapshots, [])

    def test_cpu_offload_and_checkpoint_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            HyCoReTrainingForward(CheckpointBackbone(), save_on_cpu=True, activation_checkpoint=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
