"""Epoch telemetry denominator checks; requires NumPy only, no GPU."""
import json
import unittest

import numpy as np

from ..telemetry import EpochTelemetry


class TelemetryTests(unittest.TestCase):
    def new(self):
        return EpochTelemetry([10, 11, 12, 13], [0, 0, 1, 1], num_classes=2)

    def update(self, telemetry, ids, labels, predictions, metrics=None, stats=None, mining=None):
        n = len(ids)
        logits = np.eye(2)[predictions]
        mu = np.tile([[.1, .0]], (n, 1))
        nu = np.tile([[.05, .0]], (n, 1))
        telemetry.update(ids=ids, labels=labels, logits=logits, mu=mu, nu=nu,
                         scalar_metrics=metrics or {}, hier_stats=stats, sample_mining=mining)

    def test_class_and_id_denominators_and_sampled_accuracy(self):
        telemetry = self.new()
        self.update(telemetry, [10, 10, 12], [0, 0, 1], [0, 1, 1], {"loss": 2})
        first = telemetry.summary()
        self.assertEqual(first["sample_coverage"]["unique_ids"], 2)
        self.assertEqual(first["sample_coverage"]["per_class_fraction"], [.5, .5])
        self.assertEqual(first["sample_coverage"]["per_class_draws"], [2, 1])
        self.assertEqual(first["sample_coverage"]["id_draw_count_histogram_all_training_ids"], {"0": 2, "1": 1, "2": 1})
        self.update(telemetry, [11, 13], [0, 1], [1, 1], {"loss": 4})
        summary = telemetry.summary()
        self.assertEqual(summary["sample_coverage"]["fraction"], 1)
        self.assertEqual(summary["training"]["oa"], .6)
        self.assertAlmostEqual(summary["training"]["aa_observed_classes"], 2 / 3)
        self.assertEqual(summary["scalars"]["loss"]["mean"], 3)
        self.assertEqual(summary["scalars"]["loss"]["batch_size_weighted_mean"], 2.8)
        self.assertIsNone(summary["hierarchy"])
        json.dumps(summary, allow_nan=False)

    def test_pooled_relation_counts_not_mean_of_batch_fractions(self):
        telemetry = self.new()
        stats1 = {"batch_size": 3, "eligible_anchors": 1, "triplets": 10, "unique_triplets": 8,
                  "repeated_triplets": 2, "candidate_triplets": 20, "candidate_positive_pairs": 2,
                  "unique_positive_pairs": 2, "candidate_negative_pairs": 5, "unique_negative_pairs": 4,
                  "collisions": 2, "noncollision_triplets": 8, "active_triplets": 6, "self_k_triplets": 1}
        stats2 = {"batch_size": 2, "eligible_anchors": 1, "triplets": 30, "unique_triplets": 15,
                  "repeated_triplets": 15, "candidate_triplets": 100, "candidate_positive_pairs": 10,
                  "unique_positive_pairs": 5, "candidate_negative_pairs": 20, "unique_negative_pairs": 8,
                  "collisions": 15, "noncollision_triplets": 15, "active_triplets": 9, "self_k_triplets": 6}
        mining1 = {"eligible": [True, False, False], "triplets": [[0, 2, 0]] * 5 + [[0, 2, 1]] * 5}
        mining2 = {"eligible": [True, False], "triplets": [[0, 1, 0]] * 30}
        self.update(telemetry, [10, 10, 12], [0, 0, 1], [0, 0, 1], {"lambda_hier": 0},
                    {"sample": stats1, "sample_loss": 1}, mining1)
        self.update(telemetry, [11, 13], [0, 1], [0, 1], {"lambda_hier": .5},
                    {"sample": stats2, "sample_loss": 2}, mining2)
        hierarchy = telemetry.summary()["hierarchy"]
        rates = hierarchy["sample_graph"]["rates"]
        self.assertEqual(rates["collision_rate"], 17 / 40)
        self.assertEqual(rates["active_fraction_noncollision"], 15 / 23)
        self.assertEqual(rates["candidate_triplet_coverage"], 23 / 120)
        self.assertEqual(rates["positive_pair_coverage"], 7 / 12)
        self.assertEqual(rates["negative_pair_coverage"], 12 / 25)
        self.assertEqual(hierarchy["objective_active_batches"], 1)
        self.assertEqual(hierarchy["loss_evaluated_batches"], 2)
        self.assertEqual(hierarchy["eligible_anchors"]["per_id_eligibility_given_relation_input"]["10"]["fraction"], .5)
        self.assertEqual(hierarchy["roles"]["i"]["draw_occurrences"], 40)
        self.assertEqual(hierarchy["roles"]["i"]["unique_ids"], 2)
        self.assertEqual(hierarchy["any_role"]["unique_ids"], 4)

    def test_relation_monitor_without_hinge_loss(self):
        telemetry = self.new()
        mining = {"eligible": [True, False], "triplets": [[0, 1, 0]],
                  "stats": {"batch_size": 2, "eligible_anchors": 1, "triplets": 1,
                            "unique_triplets": 1, "candidate_triplets": 2}}
        self.update(telemetry, [10, 12], [0, 1], [0, 1], {"lambda_hier": 0}, mining=mining)
        hierarchy = telemetry.summary()["hierarchy"]
        self.assertEqual(hierarchy["objective_active_batches"], 0)
        self.assertEqual(hierarchy["loss_evaluated_batches"], 0)
        self.assertIsNone(hierarchy["sample_graph"]["rates"]["collision_rate"])
        self.assertIsNone(hierarchy["sample_graph"]["rates"]["cross_class_j_fraction"])

    def test_final_radius_counter_not_projection_trigger_and_finite_guards(self):
        telemetry = self.new()
        telemetry.update(ids=[10, 12], labels=[0, 1], logits=np.eye(2),
                         mu=[[.996, 0], [.1, 0]], nu=[[.05, 0], [.05, 0]], scalar_metrics={})
        whole = telemetry.summary()["geometry"]["whole"]
        self.assertEqual(whole["near_native_numerical_boundary_count"], 1)
        self.assertEqual(whole["shadow_tangent_cap_count"], 1)
        self.assertTrue(whole["near_boundary_is_not_a_projection_trigger_count"])
        with self.assertRaises(ValueError):
            self.update(self.new(), [10], [0], [0], {"loss": float("nan")})
        with self.assertRaises(ValueError):
            self.update(self.new(), [99], [0], [0])


if __name__ == "__main__":
    unittest.main()
