import random
import unittest

from history_candidates import candidate_statistics, select_history


class HistoryCandidatesContract(unittest.TestCase):
    def test_exact_pairs_and_unique_objects(self):
        labels = [label for label in range(24) for _ in range(4)]
        chosen = select_history(labels, "class_pairs16x2", 55)
        counts = {label: sum(labels[i] == label for i in chosen) for label in set(labels)}
        self.assertEqual(len(set(chosen)), 32)
        self.assertEqual(sum(n == 2 for n in counts.values()), 16)
        self.assertTrue(all(n in (0, 2) for n in counts.values()))

    def test_small_history_and_insufficient_paired_classes(self):
        for labels in ([], list(range(32)), [0] * 96, [0, 0, 1], list(range(96))):
            for mode in ("class_pairs16x2", "uniform32"):
                chosen = select_history(labels, mode, 0)
                self.assertEqual(len(chosen), min(32, len(labels)))
                self.assertEqual(len(set(chosen)), len(chosen))
                self.assertTrue(all(0 <= index < len(labels) for index in chosen))

    def test_replay_and_main_random_stream_isolation(self):
        labels = [i % 40 for i in range(96)]
        before = random.getstate()
        for mode in ("class_pairs16x2", "uniform32"):
            self.assertEqual(select_history(labels, mode, 9), select_history(labels, mode, 9))
            self.assertNotEqual(select_history(labels, mode, 9), select_history(labels, mode, 10))
        self.assertEqual(random.getstate(), before)

    def test_current_anchor_and_age_statistics(self):
        row = candidate_statistics([0, 1], [0, 0, 2], [0, 1], [1, 2, 3])
        self.assertEqual(row["current_anchors_with_peer"], 1)
        self.assertEqual(row["current_anchors_with_two_peers"], 1)
        self.assertEqual(row["selected_age_counts"], {"1": 1, "2": 1, "3": 0})


if __name__ == "__main__":
    unittest.main()
