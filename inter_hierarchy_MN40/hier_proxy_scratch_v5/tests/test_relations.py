"""CPU numerical gates against the released HIER relation/loss expressions."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F

from hier_proxy_scratch_v5.relations import mine_reciprocal_triplets, mine_sample_triplets, poincare_distance
from hier_proxy_scratch_v5.hier_loss import HIERLoss, expmap0_c1, ghhc_loss, shadow_geometry


def _source_triplets(sim_matrix, topk, count, seed):
    # Literal released HIER get_reciprocal_triplets, SHA3986a744a1a54fd.
    np.random.seed(seed)
    nearest = torch.topk(sim_matrix, topk)[1]
    matrix = torch.zeros_like(sim_matrix).scatter_(1, nearest, torch.ones_like(sim_matrix))
    matrix = ((matrix + matrix.t()) / 2).float().fill_diagonal_(-1)
    rows = []
    for i in range(len(matrix)):
        if len(torch.nonzero(matrix[i] == 1)) <= 1:
            continue
        js = np.random.choice(torch.nonzero(matrix[i] == 1).squeeze().cpu().numpy(), count, replace=True)
        ks = np.random.choice(torch.nonzero(matrix[i] < 1).squeeze().cpu().numpy(), count, replace=True)
        rows.extend((i, int(j), int(k)) for j, k in zip(js, ks))
    return torch.tensor(rows).reshape(-1, 3)


def _fixture():
    sim = torch.eye(6, dtype=torch.float64) * 3
    for group in ([0, 1, 2], [3, 4, 5]):
        for i in group:
            for j in group:
                if i != j:
                    sim[i, j] = 2 - abs(i - j) * 0.1
    return sim


def test_source_reciprocal_and_numpy_draw_order_match():
    expected = _source_triplets(_fixture(), 3, 50, 93)
    mined = mine_reciprocal_triplets(_fixture(), topk=3, seed=93)
    assert torch.equal(expected, mined["triplets"])
    assert mined["stats"]["eligible_anchors"] == 6
    assert mined["stats"]["self_k_triplets"] > 0
    assert mined["stats"]["candidate_triplets"] == 6 * 2 * 4
    assert mined["stats"]["any_role_position_coverage"] == 1
    assert mined["stats"]["repeat_fraction"] > 0


def test_two_positive_threshold_and_empty_negative_pool():
    assert mine_reciprocal_triplets(_fixture(), topk=2)["stats"]["triplets"] == 0
    full = mine_reciprocal_triplets(_fixture(), topk=6)
    assert (full["triplets"][:, 0] == full["triplets"][:, 2]).all()
    skipped = mine_reciprocal_triplets(_fixture(), topk=6, exclude_self_negative=True)
    assert skipped["stats"]["triplets"] == 0


def test_data_id_denominators_and_distinct_negative_mode():
    data_ids = torch.tensor([10, 10, 11, 12, 13, 14])
    mined = mine_reciprocal_triplets(_fixture(), topk=3, data_ids=data_ids,
                                    exclude_self_negative=True)
    assert mined["stats"]["batch_size"] == 6
    assert mined["stats"]["batch_unique_data_ids"] == 5
    assert mined["stats"]["self_k_triplets"] == 0
    assert mined["stats"]["any_role_unique_data_ids"] == 5
    assert mined["stats"]["candidate_negative_pairs"] == 6 * 3


def _source_loss(cp_dist, triplets, tau, margin):
    i, j, k = triplets.unbind(-1)
    pair_cost = torch.maximum(cp_dist[i], cp_dist[j])
    pair_prob = F.gumbel_softmax(-pair_cost / tau, dim=1, hard=True)
    triple_cost = torch.maximum(cp_dist[k], pair_cost)
    triple_prob = F.gumbel_softmax(-triple_cost / tau, dim=1, hard=True)
    dp = [(cp_dist[index] * pair_prob).sum(1) for index in (i, j, k)]
    dt = [(cp_dist[index] * triple_prob).sum(1) for index in (i, j, k)]
    hc = F.relu(dp[0] - dt[0] + margin) + F.relu(dp[1] - dt[1] + margin) + F.relu(dt[2] - dp[2] + margin)
    return (hc * (pair_prob.argmax(-1) != triple_prob.argmax(-1)).float()).mean()


def test_hard_gumbel_three_hinge_values_and_gradients_match_source():
    triplets = torch.tensor([[0, 1, 2], [1, 2, 1], [2, 0, 3], [3, 1, 0]] * 20)
    distances = torch.tensor([[0.4, 0.6, 1.0], [0.5, 0.7, 1.2],
                              [0.8, 0.2, 0.9], [0.7, 0.6, 0.3]], dtype=torch.float64)
    source = distances.clone().requires_grad_(True)
    current = distances.clone().requires_grad_(True)
    torch.manual_seed(19)
    expected = _source_loss(source, triplets, .1, .1)
    expected.backward()
    torch.manual_seed(19)
    observed, stats, details = ghhc_loss(current, triplets, return_details=True)
    observed.backward()
    assert torch.equal(expected, observed)
    assert torch.allclose(source.grad, current.grad, atol=1e-12, rtol=1e-12)
    assert stats["collisions"] > 0
    assert torch.equal(details["per_draw_loss"].mean(), observed)
    # For i=k, the first/third hinge sum is >=2m whenever unmasked.
    self_noncollision = details["self_k_mask"] & (details["pair_proxy_idx"] != details["triple_proxy_idx"])
    assert self_noncollision.any()
    first_third = details["masked_hinges"][self_noncollision][:, [0, 2]].sum(-1)
    assert (first_third >= .2 - 1e-12).all()


def test_empty_and_collision_means_remain_differentiable_zero():
    distances = torch.ones(3, 1, requires_grad=True)
    loss, stats = ghhc_loss(distances, torch.tensor([[0, 1, 0], [1, 2, 0]]))
    assert loss == 0 and stats["collisions"] == 2
    loss.backward()
    assert torch.isfinite(distances.grad).all()
    empty, stats = ghhc_loss(distances, torch.empty(0, 3, dtype=torch.long))
    assert empty == 0 and stats["triplets"] == 0


def test_c1_random_initialization_shadow_and_live_forward():
    hierarchy = HIERLoss(num_proxies=16, dim=4, seed=22).double()
    assert hierarchy.c == 1 and hierarchy.margin == .1 and hierarchy.tau == .1
    assert torch.equal(hierarchy.tangent_proxies, HIERLoss(num_proxies=16, dim=4, seed=22).double().tangent_proxies)
    assert (hierarchy.proxies().norm(dim=-1) < 1).all()
    points = expmap0_c1(torch.randn(8, 4, dtype=torch.float64) * .2).requires_grad_(True)
    labels = torch.arange(4).repeat_interleave(2)
    loss, report = hierarchy(points, labels, topk=4)
    assert report["sample"]["triplets"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(points.grad).all() and torch.isfinite(hierarchy.tangent_proxies.grad).all()
    assert not report["geometry"]["whole"]["extra_tangent_clip_enabled"]
    near_boundary = torch.tensor([[.9995, 0.0]], dtype=torch.float64)
    assert shadow_geometry(near_boundary)["shadow_clip_count"] == 1
    assert near_boundary[0, 0] == .9995


def test_coincident_distance_has_finite_zero_gradient():
    points = torch.tensor([[.3, 0.0], [.3, 0.0], [-.2, .1]], dtype=torch.float64, requires_grad=True)
    distances = poincare_distance(points)
    assert distances[0, 1] == 0
    distances.sum().backward()
    assert torch.isfinite(points.grad).all()


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
