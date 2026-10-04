"""CPU checks for global no-replacement B64 and contiguous two-rank halves."""

import copy
import itertools
import json
import random
import unittest

from inter_hierarchy_MN40.hycore_b64_v6.sampling import SourceShuffleBatchSampler


class SourceShuffleSamplerTests(unittest.TestCase):
    def setUp(self):
        # Synthetic IDs deliberately differ from positions and are not sorted.
        self.ids = tuple(500_000 + 7 * position for position in reversed(range(9840)))

    def test_rank_concatenation_reconstructs_one_global_shuffle(self):
        rank0 = SourceShuffleBatchSampler(self.ids, rank=0)
        rank1 = SourceShuffleBatchSampler(self.ids, rank=1)
        expected = list(range(len(self.ids)))
        random.Random(22).shuffle(expected)
        reconstructed = []
        for step, (left, right) in enumerate(zip(rank0, rank1)):
            plan = rank0.global_plan(step)
            self.assertEqual(plan, rank1.global_plan(step))
            self.assertEqual(len(left), 32)
            self.assertEqual(len(right), 32)
            self.assertEqual(tuple(left + right), plan.indices)
            self.assertEqual(tuple(left), plan.rank_indices(0))
            self.assertEqual(tuple(right), plan.rank_indices(1))
            self.assertEqual(plan.ids, tuple(self.ids[index] for index in left + right))
            self.assertEqual(plan.rank_ids(0) + plan.rank_ids(1), plan.ids)
            self.assertTrue(set(left).isdisjoint(right))
            reconstructed.extend(left + right)
        self.assertEqual(reconstructed, expected[:9792])
        self.assertEqual(rank0.epoch_permutation(), tuple(expected))

    def test_full_epoch_coverage_and_dropped_tail_are_without_replacement(self):
        sampler = SourceShuffleBatchSampler(self.ids)
        report = sampler.epoch_summary()
        self.assertEqual(len(sampler), 153)
        self.assertEqual(report["steps"], 153)
        self.assertEqual(report["draws"], 9792)
        self.assertEqual(report["unique"], 9792)
        self.assertEqual(report["dropped_count"], 48)
        self.assertEqual(report["padding_count"], 0)
        self.assertEqual(report["repeated_draws"], 0)
        self.assertAlmostEqual(report["coverage"], 9792 / 9840)
        used, dropped = report["used_ids"], report["dropped_ids"]
        self.assertEqual(len(set(used)), len(used))
        self.assertEqual(len(set(dropped)), 48)
        self.assertTrue(set(used).isdisjoint(dropped))
        self.assertEqual(set(used + dropped), set(self.ids))
        self.assertEqual(dropped, [self.ids[index] for index in sampler.epoch_permutation()[9792:]])
        self.assertEqual([value for plan in report["plans"] for value in plan["ids"]], used)
        self.assertEqual(len(report["plans"]), 153)
        # The complete artifact is directly JSON serializable.
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_epoch_replay_random_access_and_rng_independence(self):
        sampler = SourceShuffleBatchSampler(self.ids)
        initial = list(sampler)
        before_rng = random.getstate()
        sampler.global_plan(152)
        sampler.epoch_summary()
        self.assertEqual(random.getstate(), before_rng)
        self.assertEqual(initial, list(sampler))
        sampler.set_epoch(1)
        epoch1 = sampler.epoch_permutation()
        self.assertNotEqual(epoch1, sampler.epoch_permutation(0))
        self.assertEqual(sorted(epoch1), list(range(9840)))
        self.assertNotEqual(initial, list(sampler))
        self.assertEqual(sampler.epoch, 1)
        self.assertEqual(sampler.global_plan(10, epoch=0),
                         SourceShuffleBatchSampler(self.ids).global_plan(10))
        self.assertEqual(sampler.epoch, 1)
        self.assertNotEqual(sampler.epoch_permutation(),
                            SourceShuffleBatchSampler(self.ids, seed=42).epoch_permutation())
        sampler.set_epoch(0)
        self.assertEqual(initial, list(sampler))

    def test_smoke_prefix_does_not_cap_nominal_plan(self):
        sampler = SourceShuffleBatchSampler(self.ids)
        prefix = list(itertools.islice(iter(sampler), 2))
        self.assertEqual(prefix, list(sampler)[:2])
        report = sampler.epoch_summary()
        self.assertEqual(len(sampler), 153)
        self.assertEqual(len(report["plans"]), 153)
        self.assertEqual(report["draws"], 9792)
        scalar = sampler.epoch_summary(include_ids=False)
        for key in ("used_ids", "dropped_ids", "plans"):
            self.assertNotIn(key, scalar)
        self.assertEqual(scalar["plan_sha256"], report["plan_sha256"])
        self.assertEqual(scalar["dropped_ids_sha256"], report["dropped_ids_sha256"])

    def test_state_roundtrip_is_strict_and_does_not_store_training_ids(self):
        sampler = SourceShuffleBatchSampler(self.ids, rank=1)
        sampler.set_epoch(17)
        state = sampler.state_dict()
        replay = SourceShuffleBatchSampler(self.ids, rank=1)
        replay.load_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(replay.state_dict(), state)
        self.assertEqual(list(replay), list(sampler))
        self.assertNotIn("ids", state)
        for key, changed in (("version", "other-version"), ("seed", 23),
                             ("rank", 0), ("global_batch", 32), ("world_size", 1),
                             ("dataset_size", 9841), ("steps", 154),
                             ("ids_sha256", "incorrect"), ("drop_last", False),
                             ("padding_count", 1), ("rank", True),
                             ("epoch", -1), ("epoch", 1.5)):
            bad = copy.deepcopy(state)
            bad[key] = changed
            with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                replay.load_state_dict(bad)
            self.assertEqual(replay.epoch, 17)
        for identity in (self.ids[:-1], tuple(reversed(self.ids)),
                         (999_999_999,) + self.ids[1:]):
            with self.assertRaises(ValueError):
                SourceShuffleBatchSampler(identity, rank=1).load_state_dict(state)
        for bad in ({}, {**state, "extra": 1}, None):
            with self.assertRaises(ValueError):
                replay.load_state_dict(bad)

    def test_no_padding_for_short_exact_and_partial_global_batches(self):
        for count in (0, 1, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129):
            with self.subTest(count=count):
                ids = tuple(10_000 + 3 * position for position in range(count))
                rank0 = SourceShuffleBatchSampler(ids, rank=0)
                rank1 = SourceShuffleBatchSampler(ids, rank=1)
                joined = [index for left, right in zip(rank0, rank1) for index in left + right]
                report = rank0.epoch_summary()
                self.assertEqual(len(rank0), count // 64)
                self.assertEqual(len(joined), (count // 64) * 64)
                self.assertEqual(len(set(joined)), len(joined))
                self.assertEqual(report["dropped_count"], count % 64)
                self.assertEqual(report["padding_count"], 0)
                self.assertEqual(report["unique"], len(joined))
                self.assertEqual(len(report["used_ids"]) + len(report["dropped_ids"]), count)

    def test_global_flip_pairs_cross_rank_halves_in_reversed_order(self):
        sampler = SourceShuffleBatchSampler(self.ids)
        for plan in sampler.iter_global_plans():
            global_negative_ids = tuple(reversed(plan.ids))
            # Global child.flip(0) gives each local anchor the opposite rank's
            # reversed child, rather than reversing its own local32 children.
            self.assertEqual(global_negative_ids[:32], tuple(reversed(plan.rank_ids(1))))
            self.assertEqual(global_negative_ids[32:], tuple(reversed(plan.rank_ids(0))))
            self.assertNotEqual(global_negative_ids[:32], tuple(reversed(plan.rank_ids(0))))
            self.assertTrue(all(anchor != negative
                                for anchor, negative in zip(plan.ids, global_negative_ids)))
            for position, negative in enumerate(global_negative_ids):
                self.assertEqual(negative, self.ids[plan.indices[63 - position]])

    def test_invalid_config_ids_and_plan_coordinates_are_rejected(self):
        for kwargs in ({"rank": 2}, {"rank": -1}, {"rank": True},
                       {"world_size": 1}, {"global_batch": 32},
                       {"seed": -1}, {"seed": 22.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SourceShuffleBatchSampler(self.ids, **kwargs)
        for ids in ([1, 1], [1, 1.0], [True, 2], ["1", "2"], [-1, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                SourceShuffleBatchSampler(ids)
        sampler = SourceShuffleBatchSampler(self.ids)
        for step in (-1, 153, 1.5, True):
            with self.assertRaises(ValueError):
                sampler.global_plan(step)
        for epoch in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                sampler.set_epoch(epoch)
            with self.assertRaises(ValueError):
                sampler.global_plan(0, epoch)
        for rank in (-1, 2, True):
            with self.assertRaises(ValueError):
                sampler.global_plan(0).rank_indices(rank)


if __name__ == "__main__":
    unittest.main()
