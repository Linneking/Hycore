"""CPU unit checks for source class blocks, rank splits and coverage accounting."""

import unittest
from collections import Counter

from inter_hierarchy_MN40.hier_proxy_scratch_v5.diagnose_batch import (
    audit_sampler, theoretical_coverage)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler


class SourceSamplerTests(unittest.TestCase):
    def setUp(self):
        self.labels = [label for label in range(40) for _ in range(5 + label)]

    def test_two_ranks_share_plan_and_preserve_source_blocks(self):
        rank0 = SourceClassBatchSampler(self.labels, steps=20, rank=0)
        rank1 = SourceClassBatchSampler(self.labels, steps=20, rank=1)
        for step, (left, right) in enumerate(zip(rank0, rank1)):
            plan = rank0.global_plan(step)
            self.assertEqual(plan, rank1.global_plan(step))
            self.assertEqual(tuple(left + right), plan.indices)
            self.assertEqual(len(plan.indices), 64)
            self.assertEqual(len(set(plan.classes)), 32)
            self.assertTrue(set(plan.rank_classes(0)).isdisjoint(plan.rank_classes(1)))
            self.assertEqual(set(Counter(plan.labels).values()), {2})
            self.assertTrue(all(a != b for a, b in zip(plan.labels, reversed(plan.labels))))
            for rank in (0, 1):
                targets = plan.rank_labels(rank)
                self.assertEqual(len(targets), 32)
                self.assertTrue(all(a != b for a, b in zip(targets, reversed(targets))))
                self.assertTrue(all(targets[start] == targets[start + 1]
                                    for start in range(0, 32, 2)))

    def test_random_access_replay_and_epoch_changes(self):
        sampler = SourceClassBatchSampler(self.labels)
        before = sampler.global_plan(10)
        sampler.global_plan(100)
        self.assertEqual(before, sampler.global_plan(10))
        replay = SourceClassBatchSampler(self.labels)
        self.assertEqual(before, replay.global_plan(10))
        sampler.set_epoch(1)
        self.assertNotEqual(before.indices, sampler.global_plan(10).indices)
        replay.load_state_dict(sampler.state_dict())
        self.assertEqual(list(sampler), list(replay))
        self.assertNotEqual(before.indices, SourceClassBatchSampler(
            self.labels, seed=42).global_plan(10).indices)

    def test_replacement_allows_same_id_and_accounting_is_unique(self):
        labels = [0, 1, 2, 3]
        sampler = SourceClassBatchSampler(labels, world_size=2,
                                          classes_per_rank=2, steps=3)
        report = audit_sampler(sampler)
        self.assertEqual(report["draws"], 24)
        self.assertEqual(report["unique"], 4)
        self.assertEqual(report["within_class_block_duplicate_blocks"], 12)
        self.assertEqual(report["repeated_draws"], 20)
        self.assertEqual(report["coverage"], 1)
        self.assertEqual(report["flip_same_class_pairs"], 0)
        self.assertEqual(report["global_flip_same_class_pairs"], 0)
        self.assertEqual(report["rank_overlap_steps"], 0)
        self.assertEqual(report["class_block_failures"], 0)

    def test_theory_distinguishes_budget_from_unique_coverage(self):
        counts = {label: 800 for label in range(40)}
        result = theoretical_coverage(counts, steps=200)
        expected = 1 - (.2 + .8 * (1 - 1 / 800) ** 2) ** 200
        self.assertAlmostEqual(result["expected_coverage"], expected)
        self.assertAlmostEqual(result["expected_coverage"], .3297805156905289)
        self.assertEqual(result["per_class"]["0"]["hard_max_coverage"], .5)
        self.assertEqual(result["draws"], 12800)
        self.assertEqual(result["per_class"]["0"]["expected_draws"], 320)

    def test_invalid_rank_shape_empty_data_and_resume_identity(self):
        for kwargs in ({"rank": 2}, {"classes_per_rank": 15},
                       {"instances_per_class": 0}, {"steps": 0}):
            with self.assertRaises(ValueError):
                SourceClassBatchSampler(self.labels, **kwargs)
        with self.assertRaises(ValueError):
            SourceClassBatchSampler([])
        sampler = SourceClassBatchSampler(self.labels)
        state = sampler.state_dict()
        state["seed"] += 1
        with self.assertRaises(ValueError):
            sampler.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
