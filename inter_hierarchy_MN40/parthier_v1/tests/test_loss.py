import json
import unittest
from unittest.mock import patch

import torch

from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss
from inter_hierarchy_MN40.parthier_v1.loss import PartHierarchyLoss, _selection_losses


MODULE = "inter_hierarchy_MN40.parthier_v1.loss"


def native_features(batch=24, dimension=5):
    rng = torch.Generator().manual_seed(309)
    whole = torch.randn((batch, dimension), generator=rng, dtype=torch.float64)
    whole = (whole / whole.norm(dim=-1, keepdim=True) * .55).requires_grad_()
    part = torch.randn((batch, 3, dimension), generator=rng, dtype=torch.float64)
    radii = torch.tensor([.35, .25, .15], dtype=torch.float64)[None, :, None]
    part = (part / part.norm(dim=-1, keepdim=True) * radii).requires_grad_()
    return whole, part, torch.arange(batch) // 2, torch.arange(batch, dtype=torch.long)


class LossTests(unittest.TestCase):
    def test_sample_hier_and_st_gradient_are_exactly_v6_reference(self):
        rng = torch.Generator().manual_seed(817)
        original = torch.rand((6, 9), generator=rng, dtype=torch.float64) * 2
        first, second = original.clone().requires_grad_(), original.clone().requires_grad_()
        triplets = torch.tensor([[0, 1, 2], [1, 2, 4], [2, 3, 5], [4, 0, 3]])
        expected, reference_stats, reference = ghhc_loss(first, triplets,
            generator=torch.Generator().manual_seed(79), return_details=True)
        actual, _, detail = _selection_losses(second, triplets,
            torch.linspace(.1, 1., 6, dtype=torch.float64),
            torch.linspace(.1, 1., 9, dtype=torch.float64),
            generator=torch.Generator().manual_seed(79))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(detail["pair_idx"], reference["pair_proxy_idx"], rtol=0, atol=0)
        torch.testing.assert_close(detail["triple_idx"], reference["triple_proxy_idx"], rtol=0, atol=0)
        self.assertEqual(int((~detail["valid"]).sum()), reference_stats["collisions"])
        expected.backward()
        actual.backward()
        torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)

    def test_collision_masks_both_terms_with_all_draw_denominator(self):
        distances = torch.tensor([[1., 3.], [2., 4.], [5., 1.]], dtype=torch.float64, requires_grad=True)
        triplets = torch.tensor([[0, 1, 2], [0, 1, 2]])
        pair = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
        triple = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        with patch(MODULE + "._hard_gumbel", side_effect=[pair, triple]):
            hier, order, details = _selection_losses(distances, triplets,
                torch.tensor([.4, .5, .1], dtype=torch.float64),
                torch.tensor([.7, .9], dtype=torch.float64))
        self.assertEqual(details["valid"].tolist(), [False, True])
        torch.testing.assert_close(hier, details["hier_raw"][1] / 2)
        torch.testing.assert_close(order, details["order_hinges"][1].sum() / 2)
        self.assertEqual(details["order_hinges"].shape[1], 2)
        self.assertEqual(float(details["order_per_draw"][0]), 0.)

    def test_order_depth_uses_live_st_selection_and_direct_part_gradient(self):
        distances = torch.tensor([[.1, .2, .4], [.3, .2, .1], [.4, .3, .2]],
                                 dtype=torch.float64, requires_grad=True)
        depths = torch.tensor([1.5, 1.8, 2.], dtype=torch.float64, requires_grad=True)
        whole_depth = torch.tensor([.2, .3, .4], dtype=torch.float64, requires_grad=True)
        _, order, details = _selection_losses(distances, torch.tensor([[0, 1, 2]]),
            whole_depth, depths, generator=torch.Generator().manual_seed(7))
        self.assertTrue(bool(details["valid"].all()))
        order.backward()
        self.assertGreater(float(distances.grad.abs().sum()), 0.)
        self.assertGreater(float(depths.grad.abs().sum()), 0.)
        # Third descendant depth is diagnostic only, preserving the proposal.
        self.assertEqual(float(whole_depth.grad[2]), 0.)

    def test_empty_draws_have_finite_differentiable_zero(self):
        distances = torch.ones((3, 2), requires_grad=True)
        whole_depth = torch.ones(3, requires_grad=True)
        candidate_depth = torch.ones(2, requires_grad=True)
        hier, order, details = _selection_losses(distances, torch.empty((0, 3), dtype=torch.long),
            whole_depth, candidate_depth)
        (hier + order).backward()
        self.assertEqual(float(hier + order), 0.)
        self.assertTrue(bool(torch.isfinite(distances.grad).all()))
        self.assertEqual(details["order_hinges"].shape, (0, 2))

    def test_empty_then_populated_queue_forward_live_scales_and_json_stats(self):
        whole, parts, labels, ids = native_features()
        loss = PartHierarchyLoss(queue_size=2048)
        self.assertEqual(list(loss.parameters()), [])
        value, stats = loss(whole, parts, labels, ids, step=0, seed=4780)
        self.assertEqual(stats["queue_candidates"], 0)
        self.assertEqual(stats["sample"]["self_k_triplets"], 0)
        self.assertGreater(stats["sample"]["triplets"], 0)
        self.assertEqual(stats["current_candidates"], len(whole) * 3)
        value.backward()
        for scale in range(3):
            self.assertGreater(float(parts.grad[:, scale].abs().sum()), 0.)
        self.assertGreater(float(whole.grad.abs().sum()), 0.)
        json.dumps(stats, allow_nan=False)
        loss.enqueue(parts, ids, step=0)
        fresh_whole, fresh_parts, labels, ids = native_features()
        next_value, next_stats = loss(fresh_whole, fresh_parts, labels, ids, step=1, seed=4781)
        self.assertEqual(next_stats["queue_candidates"], 72)
        self.assertEqual(next_stats["candidate_count"], 144)
        self.assertTrue(bool(torch.isfinite(next_value)))
        for role in next_stats["roles"].values():
            self.assertEqual(sum(role["scale_counts"]), role["noncollision_draws"])
            self.assertEqual(role["current_count"] + role["queue_count"], role["noncollision_draws"])

    def test_queue_restore_reproduces_loss_selection_and_metadata(self):
        whole, parts, labels, ids = native_features()
        first = PartHierarchyLoss(queue_size=100)
        first.enqueue(parts, ids, step=0)
        second = PartHierarchyLoss(queue_size=100)
        second.load_queue_state(first.queue_state())
        loss_a, stats_a = first(whole, parts, labels, ids, step=1, seed=81)
        loss_b, stats_b = second(whole, parts, labels, ids, step=1, seed=81)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        self.assertEqual(stats_a, stats_b)

    def test_chain_uses_grain_margin_for_all_three_links(self):
        whole, parts, labels, ids = native_features()
        # Equal depths make every link's margin visible, including large->whole.
        whole = torch.zeros_like(whole)
        whole[:, 0] = .3
        parts = whole[:, None, :].expand(-1, 3, -1).clone().requires_grad_()
        loss = PartHierarchyLoss(queue_size=0, lambda_hier=0., lambda_order=1.,
                                 radial_margin=.91, grain_margin=.13)
        empty = {"triplets": torch.empty((0, 3), dtype=torch.long), "stats": {"self_k_triplets": 0}}
        with patch(MODULE + ".mine_sample_triplets", return_value=empty):
            value, stats = loss(whole, parts, labels, ids, step=0, seed=9)
        self.assertAlmostEqual(float(value), .39)
        self.assertAlmostEqual(stats["chain_loss"], .39)
        self.assertEqual(stats["order_loss"], 0.)


if __name__ == "__main__":
    unittest.main()
