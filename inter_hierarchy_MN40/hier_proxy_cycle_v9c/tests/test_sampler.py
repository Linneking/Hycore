"""CPU-only contracts for balanced class schedules and persistent queues."""
from __future__ import annotations

import copy
import json
import unittest
from collections import Counter, defaultdict

from ..sampler import ClassCycleBatchSampler, SAMPLER_VERSION


def labels_for_counts(counts):
    return [label for label, count in enumerate(counts) for _ in range(count)]


def class_draws(plans):
    result = defaultdict(list)
    for plan in plans:
        for index, label in zip(plan.indices, plan.labels):
            result[label].append(index)
    return result


class ClassCycleSamplerContract(unittest.TestCase):
    def setUp(self):
        self.labels = labels_for_counts([37 + 22 * label for label in range(40)])

    def test_global_and_local_batches_have_distinct_classes_and_objects(self):
        sampler = ClassCycleBatchSampler(self.labels)
        self.assertEqual((len(sampler), sampler.local_batch_size, sampler.global_batch_size), (300, 32, 64))
        for plan in sampler.iter_global_plans():
            self.assertEqual((len(plan.classes), len(set(plan.classes)), len(plan.indices), len(set(plan.indices))),
                             (32, 32, 64, 64))
            self.assertEqual(len(set(plan.rank_classes(0)) & set(plan.rank_classes(1))), 0)
            for rank in range(2):
                self.assertEqual(len(plan.rank_indices(rank)), 32)
                self.assertEqual(len(set(plan.rank_classes(rank))), 16)
                self.assertEqual(set(Counter(plan.rank_labels(rank)).values()), {2})
            self.assertEqual(tuple(self.labels[index] for index in plan.indices), plan.labels)

    def test_each_five_step_block_gives_every_class_four_appearances(self):
        sampler = ClassCycleBatchSampler(self.labels)
        plans = list(sampler.iter_global_plans())
        for block_start in range(0, 300, 5):
            counts = Counter(label for plan in plans[block_start:block_start + 5] for label in plan.classes)
            self.assertEqual(counts, Counter({label: 4 for label in range(40)}))
        draws = class_draws(plans)
        self.assertEqual({len(values) for values in draws.values()}, {480})

    def test_each_initial_class_permutation_is_exhausted_before_repeat(self):
        sampler = ClassCycleBatchSampler(self.labels)
        draws = class_draws(sampler.iter_global_plans())
        for label, values in draws.items():
            first_traversal = values[:len(sampler.by_class[label])]
            self.assertEqual(len(first_traversal), len(set(first_traversal)))

    def test_odd_queue_boundaries_never_repeat_within_a_pair(self):
        labels = labels_for_counts([2 + label % 4 for label in range(40)])
        sampler = ClassCycleBatchSampler(labels)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            for plan in sampler.iter_global_plans():
                for offset in range(0, 64, 2):
                    self.assertNotEqual(plan.indices[offset], plan.indices[offset + 1])
            draws = class_draws(sampler.iter_global_plans())
            for label, values in draws.items():
                self.assertEqual(set(values), set(sampler.by_class[label]))
        self.assertTrue(any(row["cycle"] > 200 for row in sampler.state_dict()["next_epoch_queues"].values()))

    def test_cross_epoch_cursor_retains_all_coverage_and_changes_batches(self):
        counts = [2 + label * 17 for label in range(40)]
        counts[-1] = 889
        sampler = ClassCycleBatchSampler(labels_for_counts(counts))
        first_plans = tuple(sampler.iter_global_plans())
        first_seen = {index for plan in first_plans for index in plan.indices}
        self.assertEqual(len(first_seen), sum(min(count, 480) for count in counts))
        first_boundary = sampler.state_dict()["next_epoch_queues"]
        sampler.set_epoch(1)
        self.assertEqual(sampler.state_dict()["epoch_start_queues"], first_boundary)
        second_plans = tuple(sampler.iter_global_plans())
        all_seen = first_seen | {index for plan in second_plans for index in plan.indices}
        self.assertEqual(len(all_seen), sum(counts))
        self.assertNotEqual([plan.classes for plan in first_plans], [plan.classes for plan in second_plans])
        self.assertEqual(len({plan.indices for plan in first_plans + second_plans}), 600)

    def test_queries_repeated_iteration_and_current_set_epoch_are_read_only(self):
        sampler = ClassCycleBatchSampler(self.labels)
        baseline = sampler.state_dict()
        first = sampler.global_plan(0)
        sampler.global_plan(299, epoch=0)
        sampler.global_plan(12)
        sampler.set_epoch(0)
        list(sampler)
        list(sampler)
        list(sampler.iter_global_plans(epoch=0))
        self.assertEqual(sampler.global_plan(0), first)
        self.assertEqual(sampler.state_dict(), baseline)

    def test_json_checkpoint_restores_current_epoch_and_exact_next_epoch(self):
        sampler = ClassCycleBatchSampler(self.labels)
        for epoch in range(3):
            sampler.set_epoch(epoch)
        saved = json.loads(json.dumps(sampler.state_dict()))
        resumed = ClassCycleBatchSampler(self.labels)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.state_dict(), saved)
        self.assertEqual(list(resumed.iter_global_plans()), list(sampler.iter_global_plans()))
        for epoch in range(3, 6):
            resumed.set_epoch(epoch)
            sampler.set_epoch(epoch)
            self.assertEqual(list(resumed.iter_global_plans()), list(sampler.iter_global_plans()))
            self.assertEqual(resumed.state_dict(), sampler.state_dict())

    def test_rank_local_iteration_and_state_are_rank_independent(self):
        rank0 = ClassCycleBatchSampler(self.labels, rank=0)
        rank1 = ClassCycleBatchSampler(self.labels, rank=1)
        self.assertEqual(rank0.state_dict(), rank1.state_dict())
        self.assertNotEqual(list(rank0), list(rank1))
        for step, (local0, local1) in enumerate(zip(rank0, rank1)):
            self.assertEqual(tuple(local0 + local1), rank0.global_plan(step).indices)
        rank0.set_epoch(1)
        rank1.load_state_dict(rank0.state_dict())
        self.assertEqual(rank0.state_dict(), rank1.state_dict())

    def test_state_snapshots_are_deep_copies(self):
        sampler = ClassCycleBatchSampler(self.labels)
        expected = sampler.state_dict()
        mutated = sampler.state_dict()
        mutated["epoch_start_queues"]["0"]["permutation"].reverse()
        mutated["next_epoch_queues"]["0"]["permutation"].clear()
        self.assertEqual(sampler.state_dict(), expected)

    def test_configuration_and_queue_corruption_are_rejected_atomically(self):
        sampler = ClassCycleBatchSampler(self.labels)
        baseline = sampler.state_dict()
        corruptions = (("version", "other"), ("seed", 22), ("steps", 307),
                       ("labels_sha256", "invalid"), ("next_epoch", 2))
        for key, value in corruptions:
            with self.subTest(key=key):
                bad = copy.deepcopy(baseline)
                bad[key] = value
                with self.assertRaises(ValueError):
                    sampler.load_state_dict(bad)
                self.assertEqual(sampler.state_dict(), baseline)
        for location in ("epoch_start_queues", "next_epoch_queues"):
            bad = copy.deepcopy(baseline)
            bad[location]["0"]["permutation"][0] = 999999
            with self.subTest(location=location), self.assertRaises(ValueError):
                sampler.load_state_dict(bad)
            self.assertEqual(sampler.state_dict(), baseline)
        changed_labels = list(self.labels)
        changed_labels[0], changed_labels[-1] = changed_labels[-1], changed_labels[0]
        with self.assertRaises(ValueError):
            ClassCycleBatchSampler(changed_labels).load_state_dict(baseline)

    def test_epoch_order_and_invalid_queries_fail_without_consuming_queue(self):
        sampler = ClassCycleBatchSampler(self.labels)
        with self.assertRaises(ValueError):
            sampler.set_epoch(1)
        baseline = sampler.state_dict()
        for step in (-1, 300, .5, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                sampler.global_plan(step)
        for epoch in (-1, 1):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                sampler.global_plan(0, epoch)
        with self.assertRaises(ValueError):
            list(sampler.iter_global_plans(1))
        self.assertEqual(sampler.state_dict(), baseline)
        sampler.set_epoch(1)
        with self.assertRaises(ValueError):
            sampler.set_epoch(0)
        with self.assertRaises(ValueError):
            sampler.set_epoch(3)

    def test_smoke_two_steps_reports_partial_block_and_is_replayable(self):
        sampler = ClassCycleBatchSampler(self.labels, steps=2)
        self.assertEqual(len(list(sampler)), 2)
        saved = sampler.state_dict()
        self.assertEqual(saved["version"], SAMPLER_VERSION)
        self.assertEqual(saved["complete_schedule_blocks_per_epoch"], 0)
        self.assertEqual(saved["partial_schedule_block_steps"], 2)
        resumed = ClassCycleBatchSampler(self.labels, steps=2)
        resumed.load_state_dict(saved)
        self.assertEqual(list(sampler), list(resumed))
        sampler.set_epoch(1)
        resumed.set_epoch(1)
        self.assertEqual(sampler.state_dict(), resumed.state_dict())

    def test_invalid_shapes_and_class_pools_are_rejected(self):
        for kwargs in ({"world_size": 1}, {"world_size": 0}, {"classes_per_rank": 15},
                       {"instances_per_class": 1}, {"steps": 0}, {"steps": 2.5},
                       {"rank": -1}, {"rank": 2}, {"seed": -1}, {"steps": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ClassCycleBatchSampler(self.labels, **kwargs)
        for labels in ([], labels_for_counts([2] * 39), labels_for_counts([2] * 41),
                       labels_for_counts([1] + [2] * 39)):
            with self.subTest(classes=len(set(labels))), self.assertRaises(ValueError):
                ClassCycleBatchSampler(labels)


if __name__ == "__main__":
    unittest.main()
