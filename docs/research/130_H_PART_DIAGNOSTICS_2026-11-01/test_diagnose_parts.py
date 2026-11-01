"""CPU-only metric contracts; no model, server, or CUDA use."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("diagnose_parts", Path(__file__).with_name("diagnose_parts.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def score_arrays(target):
    return {key: np.ones_like(target, dtype=np.float64) for key in module.SCORE_NAMES}


class DiagnosticContracts(unittest.TestCase):
    def test_bag_constant_prediction_matches_official_iou(self):
        target = np.array([[4, 5, 5, 5], [4, 4, 5, 5]])
        pred = np.full_like(target, 5)
        stat = module.PartStatistics()
        rows = stat.update(pred, target, [1, 1], [0, 1], ["a", "b"], score_arrays(target))
        result = stat.result()
        self.assertAlmostEqual(np.mean([row["shape_miou"] for row in rows]), .3125)
        self.assertEqual(result["parts"]["4"]["recall"], 0.)
        self.assertIsNone(result["parts"]["4"]["precision"])
        self.assertEqual(result["parts"]["4"]["missing_fraction_given_gt_present"], 1.)
        self.assertEqual(result["categories"]["bag"]["constant_prediction_label_counts"], {"5": 2})
        self.assertEqual(result["categories"]["bag"]["single_predicted_label_fraction"], 1.)
        self.assertEqual(result["categories"]["bag"]["illegal_prediction_fraction"], 0.)

    def test_absent_both_is_one_only_for_object_metric(self):
        target = np.array([[5, 5, 5, 5]])
        stat = module.PartStatistics()
        rows = stat.update(target, target, [1], [0], ["a"], score_arrays(target))
        result = stat.result()
        self.assertEqual(rows[0]["shape_miou"], 1.)
        self.assertEqual(result["parts"]["4"]["object_mean_iou_owner_category"], 1.)
        self.assertIsNone(result["parts"]["4"]["object_mean_iou_gt_present_only"])
        self.assertIsNone(result["parts"]["4"]["pooled_iou"])
        self.assertIsNone(result["parts"]["4"]["missing_fraction_given_gt_present"])

    def test_wrong_category_prediction_is_counted_as_false_positive(self):
        target = np.array([[4, 5, 5, 5], [41, 41, 42, 43]])
        pred = np.array([[41, 5, 5, 5], [41, 41, 42, 43]])
        stat = module.PartStatistics()
        stat.update(pred, target, [1, 13], [0, 1], ["bag", "rocket"], score_arrays(target))
        result = stat.result()
        self.assertAlmostEqual(result["parts"]["41"]["precision"], 2 / 3)
        self.assertEqual(result["parts"]["41"]["pred_present_objects_all_categories"], 2)
        self.assertEqual(result["categories"]["bag"]["illegal_prediction_fraction"], .25)
        self.assertEqual(result["parts"]["41"]["object_mean_iou_owner_category"], 1.)

    def test_zero_recall_does_not_imply_no_predictions(self):
        target = np.array([[4, 5]])
        pred = np.array([[5, 4]])
        stat = module.PartStatistics()
        stat.update(pred, target, [1], [0], ["a"], score_arrays(target))
        result = stat.result()
        self.assertEqual(result["parts"]["4"]["gt_present_but_zero_recall_objects"], 1)
        self.assertEqual(result["parts"]["4"]["gt_present_but_no_prediction_objects"], 0)

    def test_score_means_and_histogram_are_weighted_by_true_points(self):
        target = np.array([[4, 5, 5, 5]])
        score = score_arrays(target)
        score["correct_probability"] = np.array([[.1, .2, .4, .6]])
        score["correct_rank"] = np.array([[5, 1, 2, 2]])
        stat = module.PartStatistics()
        stat.update(target, target, [1], [0], ["a"], score)
        result = stat.result()
        self.assertAlmostEqual(result["parts"]["5"]["correct_probability"]["mean"], .4)
        self.assertEqual(result["parts"]["5"]["correct_rank_histogram_1_to_50"][:3], [1, 2, 0])
        self.assertEqual(result["parts"]["4"]["correct_rank_histogram_1_to_50"][4], 1)

    def test_rejects_illegal_ground_truth(self):
        target = np.array([[41, 5]])
        with self.assertRaises(ValueError):
            module.PartStatistics().update(target, target, [1], [0], ["a"], score_arrays(target))

    def test_metric_tolerance_is_reported_without_exception(self):
        expected = {"instance_miou": .8, "class_miou": .7, "objects": 2,
                    "per_class": {"bag": {"miou": .6}}}
        actual = {"instance_miou": .8, "class_miou": .7, "objects": 2,
                  "per_class": {"bag": {"miou": .6001}}}
        result = module.metric_differences(actual, expected)
        self.assertFalse(result["within_1e_minus5"])
        self.assertTrue(result["objects_match"])


if __name__ == "__main__":
    unittest.main()
