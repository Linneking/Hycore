"""Small CPU contracts for the scientific readout, not encoder training."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import metrics as m


class MetricContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_basic_order_and_margin_are_distinct_and_empty_is_na(self):
        result = m.gap_summary(np.array([-.1, 0., .1, .2, .3], np.float32), .2)
        self.assertEqual(result["count"], 5)
        self.assertAlmostEqual(result["basic_fraction"], .6)
        self.assertAlmostEqual(result["margin_fraction"], .4)
        self.assertIsNone(m.gap_summary(np.array([], np.float32), .2)["margin_fraction"])
        with self.assertRaises(ValueError):
            m.gap_summary([1., float("nan")], .2)

    def test_source_fp32_margin_rounding_is_retained(self):
        # 1000/600 rounds down in FP32. Native zero hinge is still a pass.
        value = np.array([1000 / 600], dtype=np.float32)
        result = m.gap_summary(value, 1000 / 600)
        self.assertEqual(result["margin_fraction"], 1.)
        self.assertEqual(result["mean_hinge"], 0.)

    def test_joint_chain_not_inferred_from_marginal_means(self):
        gaps = np.array([[.4, .4, -.01], [-.01, .4, .4], [.4, -.01, .4]], np.float32)
        self.assertTrue((gaps.mean(0) > .2).all())
        info = m.joint_fraction(gaps, .2)
        self.assertEqual(info["joint_margin_fraction"], 0.)
        self.assertEqual(info["joint_basic_fraction"], 0.)

    def test_class_centers_average_unit_directions_not_raw_radius(self):
        clean = np.array([[.9, 0], [0, .1], [-.2, 0], [0, -.8]], np.float32)
        centers, ids, lengths = m.class_centers(clean, np.array([0, 0, 1, 1]))
        expected = torch.tensor([[1., 1.], [-1., -1.]], dtype=torch.float64) / np.sqrt(2)
        torch.testing.assert_close(centers, expected)
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertTrue(bool((lengths > 0).all()))

    def test_angles_use_descendant_labels_and_deduplicate_pooled_classes(self):
        centers = torch.tensor([[1., 0], [0, 1.], [-1., 0]], dtype=torch.float64)
        vectors = torch.tensor([[.2, 0], [0, .3], [0, 0]], dtype=torch.float32)
        desc = torch.tensor([[0, 0, 1], [0, 2, 2], [0, 1, 2]])
        info = m.angle_measurements(vectors, centers, torch.arange(3), desc)
        np.testing.assert_allclose(info["nearest"][:2], [0, 0], atol=1e-8)
        self.assertEqual(info["nearest_in_descendants"].tolist(), [True, False, False])
        self.assertEqual(info["valid"].tolist(), [True, True, False])
        self.assertTrue(np.isnan(info["nearest"][2]))
        self.assertEqual(np.isfinite(info["descendant_angles"][0]).sum(), 2)
        self.assertEqual(np.isfinite(info["descendant_angles"][1]).sum(), 2)

    def test_histogram_counts_and_percentages_keep_boundary_mass(self):
        rows = m.histogram_rows("x", [0., .1, .1, .2], [0., .1, .2], view="whole")
        self.assertEqual(sum(x["count"] for x in rows), 4)
        self.assertAlmostEqual(sum(x["percentage"] for x in rows), 100.)
        with self.assertRaises(ValueError):
            m.histogram_rows("x", [1.], [0., .1])

    def test_cache_identity_reorders_all_views_and_rejects_duplicate_ids(self):
        z = np.array([[.2, 0], [.1, 0]], np.float32)
        data = {"ids": np.array([1, 0]), "labels": np.array([9, 8]),
                "clean": z, "whole": z, "parts": np.repeat(z[:, None], 3, axis=1)}
        out = m.validate_cache(data, 2)
        self.assertEqual(out["labels"].tolist(), [8, 9])
        np.testing.assert_array_equal(out["clean"][0], z[1])
        data["ids"] = np.array([0, 0])
        with self.assertRaises(ValueError):
            m.validate_cache(data, 2)

    def test_collision_zero_loss_is_not_a_success_denominator(self):
        acc = m.RelationAccumulator()
        acc.draws, acc.collisions = 8, 8
        out = acc.finish()
        self.assertEqual(out["collision_fraction"], 1.)
        self.assertEqual(out["valid_draws"], 0)
        self.assertIsNone(out["ord"]["joint_margin_fraction"])
        self.assertIsNone(out["hier"]["joint_margin_fraction"])

    def test_actual_production_selection_keeps_all_draw_loss_denominator(self):
        # One candidate forces every A/B index collision. The correct loss is
        # zero, but the audit reports no valid radial or distance successes.
        distances = torch.tensor([[1.], [2.], [3.]])
        trips = torch.tensor([[0, 1, 2], [1, 0, 2]])
        h, o, detail = m._selection_losses(distances, trips, torch.tensor([3., 4., 5.]),
                                          torch.tensor([1.]), generator=torch.Generator().manual_seed(1))
        self.assertEqual(float(h), 0.)
        self.assertEqual(float(o), 0.)
        self.assertFalse(bool(detail["valid"].any()))


if __name__ == "__main__":
    unittest.main()
