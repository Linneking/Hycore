"""CPU protocol checks for unified relations and full-coverage 4x8 batches."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hier_proxy_scratch_v4.mining import mine_batch_triplets, pairwise_poincare_distance_c1
from hier_proxy_scratch_v4.sampler import EpochCoveringClassBatchSampler


def balanced_features():
    generator = torch.Generator().manual_seed(813)
    labels = torch.arange(4).repeat_interleave(8)
    # Geometry deliberately overlaps classes: class priority must still hold.
    mu = (0.12 * torch.randn(32, 4, generator=generator)).requires_grad_(True)
    return mu, labels


def expected_mutual(mu, labels, k):
    distance = pairwise_poincare_distance_c1(mu).tolist()
    directed = []
    for i in range(len(mu)):
        candidates = [j for j in range(len(mu)) if j != i]
        candidates.sort(key=lambda j: (labels[i].item() != labels[j].item(), distance[i][j], j))
        directed.append(set(candidates[:k]))
    return directed, [set(j for j in row if i in directed[j])
                      for i, row in enumerate(directed)]


def test_unified_replacement_draws_obey_mutual_complement_and_class_priority():
    mu, labels = balanced_features()
    for k in (3, 4, 5):
        result = mine_batch_triplets(mu, labels, k=k, t_per_anchor=256, seed=19)
        repeated = mine_batch_triplets(mu, labels, k=k, t_per_anchor=256, seed=19)
        assert torch.equal(result["inter"], repeated["inter"])
        directed, mutual = expected_mutual(mu, labels, k)
        all_mutual = torch.tensor([[j in mutual[i] for j in range(32)] for i in range(32)])
        assert torch.equal(result["mutual"], all_mutual)
        anchors = set(result["inter"][:, 0].tolist())
        assert anchors == {i for i in range(32) if len(mutual[i]) >= 2}
        assert len(result["inter"]) == len(anchors) * 256
        for i, j, negative in result["inter"].tolist():
            assert len({i, j, negative}) == 3
            assert labels[i] == labels[j]
            assert j in mutual[i]
            assert negative not in mutual[i]
        # A directed but nonreciprocal neighbour is a legal negative.
        observed = {tuple(row) for row in result["inter"].tolist()}
        possible = {(i, negative) for i in anchors for negative in directed[i] - mutual[i]}
        assert possible
        assert any((i, negative) in possible for i, _, negative in observed)
        assert result["stats"]["same_class_triplets"] > 0
        assert result["stats"]["cross_class_triplets"] > 0
        assert result["stats"]["repeated_triplets"] > 0
        assert result["stats"]["positive_pair_coverage"] == 1.0
        assert result["stats"]["unique_triplets"] == len(observed)
        assert not result["inter"].requires_grad
        assert not result["mutual"].requires_grad
        assert mu.grad is None


def test_topk_seven_removes_same_class_negatives_in_strict_4x8():
    mu, labels = balanced_features()
    result = mine_batch_triplets(mu, labels, k=7, t_per_anchor=16, seed=19)
    assert result["stats"]["eligible_anchors"] == 32
    assert result["stats"]["eligible_same_negative_candidates"] == 0
    assert result["stats"]["same_class_triplets"] == 0
    assert result["stats"]["cross_class_triplets"] == 512


def test_detached_distance_and_empty_or_ineligible_batches():
    mu = torch.tensor([[0.0], [0.5]], requires_grad=True)
    distance = pairwise_poincare_distance_c1(mu)
    assert not distance.requires_grad
    assert torch.allclose(distance[0, 1], torch.tensor(math.log(3)), atol=1e-6)
    result = mine_batch_triplets(mu, torch.tensor([0, 1]), k=1)
    assert result["inter"].shape == (0, 3)
    assert result["stats"]["anchor_coverage"] == 0
    result = mine_batch_triplets(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert result["inter"].shape == (0, 3)
    assert result["stats"]["triplets"] == 0
    try:
        pairwise_poincare_distance_c1(torch.tensor([[1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("boundary embeddings must be rejected")


def assert_sampler_plan(labels, sampler):
    batches = list(sampler)
    assert len(batches) == len(sampler)
    seen = set()
    c, s = sampler.classes_per_batch, sampler.samples_per_class
    for batch in batches:
        assert len(batch) == c * s
        targets = np.asarray(labels)[batch]
        block_classes = targets.reshape(c, s)
        assert np.all(block_classes == block_classes[:, :1])
        assert len(set(block_classes[:, 0].tolist())) == c
        assert np.all(targets != targets[::-1])
        for row in np.asarray(batch).reshape(c, s):
            if len(sampler.by_class[int(np.asarray(labels)[row[0]])]) >= s:
                assert len(set(row.tolist())) == s
        seen.update(batch)
    assert seen == set(range(len(labels)))
    assert sampler.stats["planned_instance_coverage"] == 1.0
    assert sampler.stats["planned_repeated_draws"] == len(batches) * c * s - len(labels)
    chunk_counts = [math.ceil(len(indices) / s) for indices in sampler.by_class.values()]
    assert len(batches) == max(max(chunk_counts), math.ceil(sum(chunk_counts) / c))
    return batches


def test_epoch_covering_sampler_handles_tail_and_class_imbalance():
    labels = np.repeat(np.arange(6), [9, 16, 7, 40, 8, 11])
    sampler = EpochCoveringClassBatchSampler(labels, seed=77)
    first = assert_sampler_plan(labels, sampler)
    assert first == list(EpochCoveringClassBatchSampler(labels, seed=77))
    assert first == list(sampler)
    sampler.set_epoch(1)
    second = assert_sampler_plan(labels, sampler)
    assert first != second


def test_epoch_covering_schedule_random_imbalanced_class_sizes():
    rng = np.random.default_rng(98)
    for _ in range(100):
        nclasses = int(rng.integers(4, 41))
        counts = rng.integers(1, 600, size=nclasses)
        labels = np.repeat(np.arange(nclasses), counts)
        assert_sampler_plan(labels, EpochCoveringClassBatchSampler(labels, seed=31))


if __name__ == "__main__":
    for check in (
        test_unified_replacement_draws_obey_mutual_complement_and_class_priority,
        test_topk_seven_removes_same_class_negatives_in_strict_4x8,
        test_detached_distance_and_empty_or_ineligible_batches,
        test_epoch_covering_sampler_handles_tail_and_class_imbalance,
        test_epoch_covering_schedule_random_imbalanced_class_sizes,
    ):
        check()
        print(f"PASS {check.__name__}")
