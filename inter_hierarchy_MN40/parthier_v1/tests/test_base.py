"""CPU contracts for nested source identity, BN and all three live losses."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

HAS_GEOOPT = importlib.util.find_spec("geoopt") is not None


@unittest.skipIf(torch is None, "CPU Torch is required")
class CropAndForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inter_hierarchy_MN40.parthier_v1 import base_protocol
        cls.base = base_protocol
        torch.set_num_threads(2)

    def fixture(self):
        return torch.randn(4, 820, 3, generator=torch.Generator().manual_seed(4780))

    def test_one_frozen_600_selection_has_exact_nested_sources_and_one_whole_overwrite(self):
        points = self.fixture()
        before = points.clone()
        whole, parts, stats = self.base.sample_nested_parts(
            points, 800, whole_centers=[0, 17, 302, 819], centers=[799, 403, 17, 0])
        frozen = points.transpose(1, 2).gather(2, stats["whole_indices"][:, None].expand(-1, 3, -1))
        expected600 = frozen.gather(2, stats["part_indices"][:, None].expand(-1, 3, -1))
        torch.testing.assert_close(points, before, rtol=0, atol=0)
        torch.testing.assert_close(whole[:, :, :600], expected600, rtol=0, atol=0)
        torch.testing.assert_close(whole[:, :, 600:], frozen[:, :, 600:], rtol=0, atol=0)
        for count, part in zip(self.base.PART_COUNTS, parts):
            torch.testing.assert_close(part, expected600[:, :, :count], rtol=0, atol=0)
            recovered = points.transpose(1, 2).gather(
                2, stats["part_source_indices"][:, None, :count].expand(-1, 3, -1))
            torch.testing.assert_close(part, recovered, rtol=0, atol=0)
            self.assertTrue(bool((stats["part_indices"][:, :count] ==
                                  stats["part_centers"][:, None]).any(dim=1).all()))
        self.assertEqual([part.shape[-1] for part in parts], [200, 400, 600])
        self.assertTrue(stats["parts_nested"])
        self.assertTrue(stats["whole_suffix_unchanged"])
        self.assertTrue(stats["part600_matches_whole_prefix"])
        self.assertFalse(stats["parts_alias_whole"])
        self.assertEqual(stats["single_part_overwrite_count"], 1)
        # Different scales are independent snapshots, not mutable aliases.
        old400, oldwhole = parts[1].clone(), whole.clone()
        parts[0][0, 0, 0] += 10
        torch.testing.assert_close(parts[1], old400, rtol=0, atol=0)
        torch.testing.assert_close(whole, oldwhole, rtol=0, atol=0)

    def test_invalid_sources_counts_and_centers_are_rejected(self):
        points = self.fixture()
        for count in (599, 821, 700.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.base.sample_nested_parts(points, count)
        for centers in ([800] * 4, [-1] * 4, [0, 1]):
            with self.subTest(centers=centers), self.assertRaises(ValueError):
                self.base.sample_nested_parts(points, 800, centers=centers)
        with self.assertRaises(TypeError):
            self.base.sample_nested_parts(points, 800, centers=[0.0] * 4)

    def test_four_bn_calls_small_to_large_then_whole_without_cap_or_recompute(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)
                self.embedding = nn.Parameter(torch.tensor([.995, 0., 0.]))
                self.classifier = nn.Linear(3, 40)
                self.calls = []

            def forward(self, points, emb=False):
                batch = self.bn(points.mean(-1))
                mu = self.embedding[None].expand(len(points), -1) + batch * 0
                self.calls.append((bool(emb), points.detach().clone()))
                return mu, self.classifier(mu)

        for offload in (False, True):
            with self.subTest(save_on_cpu=offload):
                backbone = Probe().train()
                model = self.base.HyCoReTrainingForward(backbone, save_on_cpu=offload)
                result = model(self.fixture(), 800)
                self.assertEqual(result["parts"].shape, (4, 3, 3))
                self.assertEqual([call[0] for call in backbone.calls], [True, True, True, False])
                self.assertEqual([call[1].shape[-1] for call in backbone.calls], [200, 400, 600, 800])
                self.assertEqual(int(backbone.bn.num_batches_tracked), 4)
                for index, count in enumerate(self.base.PART_COUNTS):
                    torch.testing.assert_close(backbone.calls[index][1], backbone.calls[-1][1][:, :, :count],
                                               rtol=0, atol=0)
                (result["mu"].sum() + result["parts"].sum()).backward()
                torch.testing.assert_close(backbone.embedding.grad, torch.full((3,), 16.), rtol=0, atol=0)
                self.assertEqual(int(backbone.bn.num_batches_tracked), 4)
                self.assertEqual(len(backbone.calls), 4)
                self.assertGreater(float(2 * torch.atanh(result["mu"].detach().norm(dim=-1)).min()), 4.6)
                self.assertFalse(any(module._forward_hooks for module in model.modules()))


@unittest.skipIf(torch is None or not HAS_GEOOPT, "CPU Torch/Geoopt is required for native intra")
class GlobalBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inter_hierarchy_MN40.parthier_v1 import base_protocol
        cls.base = base_protocol
        torch.set_num_threads(2)

    def fixture(self):
        generator = torch.Generator().manual_seed(4780)
        mu = (torch.randn(64, 9, generator=generator) * .03).requires_grad_()
        parts = (torch.randn(64, 3, 9, generator=generator) * .02).requires_grad_()
        logits = torch.randn(64, 40, generator=generator, requires_grad=True)
        return (mu, parts, logits), torch.arange(32).repeat_interleave(2)

    @staticmethod
    def outputs(mu, parts, logits):
        return {"mu": mu, "parts": parts, "logits": logits, "whole_count": 900,
                "input_stats": {"audit": True}}

    def test_three_original_intra_terms_sum_at_full_weight_and_have_live_gradients(self):
        inputs, labels = self.fixture()
        actual = self.base.global_base_losses(self.outputs(*inputs), labels)
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        mu, parts, logits = reference_inputs
        expected_scales = {str(count): self.base.intra_losses(mu, parts[:, index], count)
                           for index, count in enumerate(self.base.PART_COUNTS)}
        contrastive = sum(item["intra_contrastive"] for item in expected_scales.values())
        radial = sum(item["intra_radial"] for item in expected_scales.values())
        expected = self.base.smoothed_ce(logits, labels) + .01 * contrastive + .01 * radial
        for scale, values in expected_scales.items():
            for key, value in values.items():
                torch.testing.assert_close(actual["per_scale"][scale][key], value, rtol=0, atol=0)
        torch.testing.assert_close(actual["base"], expected, rtol=0, atol=0)
        gradients = torch.autograd.grad(actual["base"], inputs)
        references = torch.autograd.grad(expected, reference_inputs)
        for gradient, reference in zip(gradients, references):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        for scale in range(3):
            self.assertGreater(float(gradients[1][:, scale].norm()), 0.)
        self.assertEqual(actual["global_batch_size"], 64)
        self.assertTrue(actual["input_stats"]["audit"])

    def test_global_gather_order_and_full_global_flip_are_preserved(self):
        inputs, labels = self.fixture()
        expected = self.base.global_base_losses(self.outputs(*inputs), labels)
        with mock.patch.object(self.base, "gather_with_grad", side_effect=inputs), \
                mock.patch.object(self.base, "gather_without_grad", side_effect=(
                    labels, torch.tensor([[900, 200, 400, 600], [900, 200, 400, 600]]))):
            actual = self.base.global_base_losses(self.outputs(*(value[:32] for value in inputs)), labels[:32])
        torch.testing.assert_close(actual["base"], expected["base"], rtol=0, atol=0)
        self.assertIs(actual["parts"], inputs[1])
        with self.assertRaisesRegex(RuntimeError, "shares the anchor class"):
            self.base.global_base_losses(self.outputs(*inputs), torch.zeros_like(labels))

    def test_global_count_and_rank_count_mismatch_fail(self):
        inputs, labels = self.fixture()
        with self.assertRaisesRegex(ValueError, "expected global64"):
            self.base.global_base_losses(self.outputs(*(value[:32] for value in inputs)), labels[:32])
        with mock.patch.object(self.base, "gather_without_grad", side_effect=(
                labels, torch.tensor([[900, 200, 400, 600], [800, 200, 400, 600]]))):
            with self.assertRaisesRegex(RuntimeError, "counts must be shared"):
                self.base.global_base_losses(self.outputs(*inputs), labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
