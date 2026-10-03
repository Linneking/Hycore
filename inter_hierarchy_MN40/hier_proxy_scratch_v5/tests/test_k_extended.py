"""Diagnostic-denominator checks, independent of training/model behavior."""
import unittest

import torch

from ..diagnose_k_extended import (_conditional_eligibility, _cross_distance_ranks,
                                  geometric_graph_report, matched_view_report, spearman)


class ExtendedKDiagnosticTests(unittest.TestCase):
    def test_spearman_average_ties_and_constant(self):
        self.assertAlmostEqual(spearman(torch.tensor([1, 1, 3, 4]), torch.tensor([2, 2, 6, 8])), 1.0)
        self.assertAlmostEqual(spearman(torch.tensor([1, 1, 3, 4]), torch.tensor([8, 8, 4, 2])), -1.0)
        self.assertIsNone(spearman(torch.ones(4), torch.arange(4)))

    def test_cross_gap_excludes_self_and_label_boosted_pairs(self):
        labels = torch.tensor([0, 0, 1, 2])
        distances = torch.tensor([[0., 9., 2., 1.], [9., 0., 3., 4.],
                                  [2., 3., 0., 5.], [1., 4., 5., 0.]])
        mutual = torch.zeros((4, 4), dtype=torch.bool)
        mutual[0, 1] = mutual[1, 0] = True
        mutual[0, 2] = mutual[2, 0] = True
        eligible = torch.tensor([True, False, False, False])
        mined = {"mutual": mutual, "negative": ~mutual, "eligible": eligible}
        _, ranks, percentages = _cross_distance_ranks(distances, labels)
        report = geometric_graph_report(distances, labels, torch.arange(4) / 10,
                                        mined, ranks, percentages, mutual, mutual)
        gap = report["cross_only_candidate_gap"]
        # Eligible i=0 has cross-positive j=2 and cross-negative k=3 only.
        # Closer one-way negative makes this raw-distance gap -1; valid result.
        self.assertEqual(gap["gap"]["count"], 1)
        self.assertEqual(gap["gap"]["mean"], -1)
        self.assertEqual(gap["negative_gap_fraction"], 1)
        self.assertEqual(report["cross_reciprocal_edges"]["directed_count"], 2)

    def test_id_conditionals_keep_position_and_batch_denominators(self):
        cases = [{"gold": torch.tensor([0, 0, 1]), "ids": torch.tensor([7, 7, 8]), "view": 0},
                 {"gold": torch.tensor([0, 0, 1]), "ids": torch.tensor([7, 7, 8]), "view": 1},
                 {"gold": torch.tensor([0, 1, 1]), "ids": torch.tensor([7, 8, 9]), "view": 0}]
        graphs = [{"eligible": torch.tensor([True, False, False])},
                  {"eligible": torch.tensor([True, True, True])},
                  {"eligible": torch.tensor([False, True, False])}]
        result = _conditional_eligibility(cases, [0, 1, 2], graphs)
        sample = result["per_id"]["7"]
        self.assertEqual(result["primary_view_cases"], 2)
        self.assertEqual(sample["sampled_positions"], 3)
        self.assertEqual(sample["eligible_positions"], 1)
        self.assertEqual(sample["batches_present"], 2)
        self.assertEqual(sample["any_eligible_fraction_given_batch_present"], .5)

    def test_empty_paired_graph_jaccard_is_undefined(self):
        cases = [{"stage": "reference", "step": 0, "view": view,
                  "gold": torch.tensor([0, 1]), "ids": torch.tensor([7, 8])} for view in (0, 1)]
        graphs = [{"mutual": torch.zeros((2, 2), dtype=torch.bool),
                   "eligible": torch.zeros(2, dtype=torch.bool)} for _ in cases]
        result = matched_view_report(cases, [0, 1], graphs, graphs)
        self.assertEqual(result["matched_pairs"], 1)
        self.assertEqual(result["cross_edge_undefined_empty_union_pairs"], 1)
        self.assertIsNone(result["cross_edge_jaccard_pooled"])
        self.assertIsNone(result["eligible_jaccard_pooled"])
        self.assertEqual(result["eligible_state_agreement"]["mean"], 1)


if __name__ == "__main__":
    unittest.main()
