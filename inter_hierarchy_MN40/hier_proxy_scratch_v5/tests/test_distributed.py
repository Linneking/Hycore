"""Meaningful CPU gates for restored aliasing and distributed gradients."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from inter_hierarchy_MN40.hier_proxy_scratch_v5.base_protocol import (
    HyCoReTrainingForward, get_children_alias, global_base_losses, smoothed_ce,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.check_distributed import run_check
from inter_hierarchy_MN40.hier_proxy_scratch_v5.distributed import shared_crop_counts


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.calls = []

    def forward(self, x, emb=False):
        self.calls.append((x.shape[-1], emb, x.detach().clone()))
        value = self.bn(x).mean(-1).tanh() * 0.05
        return value, value


class BaseProtocolTests(unittest.TestCase):
    def test_child_overwrites_whole_prefix_through_original_view(self):
        source = torch.arange(30, dtype=torch.float64).reshape(1, 3, 10)
        original = source.clone()
        _, whole, _ = get_children_alias(source, 8, centers=[7])
        before_child = whole.clone()
        _, child, _ = get_children_alias(whole, 4, centers=[6])
        self.assertEqual(source.untyped_storage().data_ptr(), whole.untyped_storage().data_ptr())
        self.assertEqual(whole.untyped_storage().data_ptr(), child.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(whole[:, :, :4], child))
        self.assertFalse(torch.equal(whole, before_child))
        self.assertFalse(torch.equal(source, original))

    def test_child_then_whole_updates_same_bn_twice(self):
        torch.manual_seed(21)
        random.seed(21)
        backbone = RecordingBackbone().train()
        wrapper = HyCoReTrainingForward(backbone)
        result = wrapper(torch.randn(4, 16, 3), 12, 6)
        self.assertEqual(int(backbone.bn.num_batches_tracked), 2)
        self.assertEqual([(count, emb) for count, emb, _ in backbone.calls], [(6, True), (12, False)])
        self.assertTrue(torch.equal(backbone.calls[0][2], backbone.calls[1][2][:, :, :6]))
        self.assertTrue(result["input_stats"]["child_aliases_whole"])
        self.assertTrue(result["input_stats"]["whole_changed_by_child"])

    def test_crop_counts_are_shared_reproducibly_without_global_rng_side_effect(self):
        random.seed(124)
        before = random.getstate()
        counts = shared_crop_counts(22, 0, 9)
        self.assertEqual(counts, shared_crop_counts(22, 0, 9))
        self.assertEqual(before, random.getstate())
        self.assertTrue(800 <= counts[0] <= 1024 and 200 <= counts[1] <= 600)

    @unittest.skipUnless(importlib.util.find_spec("geoopt"), "Geoopt required for original c1 distances")
    def test_base_uses_global_flip_and_original_ce_smoothing(self):
        torch.manual_seed(23)
        mu = (torch.randn(64, 4, dtype=torch.float64) * 0.02).requires_grad_()
        nu = (torch.randn(64, 4, dtype=torch.float64) * 0.01).requires_grad_()
        logits = torch.randn(64, 40, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(32).repeat_interleave(2)
        outputs = {"mu": mu, "nu": nu, "logits": logits, "child_count": 350}
        result = global_base_losses(outputs, labels)
        from geoopt import PoincareBall
        ball = PoincareBall(c=1).double()
        expected_contrastive = (ball.dist(mu, nu) - ball.dist(mu, nu.flip(0)) + 4).relu().mean()
        torch.testing.assert_close(result["intra_contrastive"], expected_contrastive)
        target = torch.full_like(logits, 0.2 / 39)
        target.scatter_(1, labels[:, None], 0.8)
        expected_ce = -(target * logits.log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(result["ce"], expected_ce)
        result["base"].backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in (mu, nu, logits)))

    def test_smoothing_is_not_pytorch_uniform_over_all_classes(self):
        logits = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float64)
        labels = torch.tensor([0])
        expected = -(torch.tensor([[0.8, 0.1, 0.1]], dtype=logits.dtype) * logits.log_softmax(-1)).sum()
        torch.testing.assert_close(smoothed_ce(logits, labels), expected)


class DistributedGradientTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         "CPU Gloo required")
    def test_two_rank_global_objective_matches_single_rank_parameter_gradient(self):
        self.assertTrue(run_check()["passed"])


if __name__ == "__main__":
    unittest.main()
