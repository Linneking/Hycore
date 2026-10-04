"""Synthetic CPU checks for geometry audit semantics, without model dependencies."""
import unittest
import numpy as np

from analyze_proxy_feature_geometry import (
    all_sample_nearest, analyze_geometry, pair_matrices, proxy_geometry,
    proxy_sample_neighbours, validated_cache,
)


class GeometryAuditTests(unittest.TestCase):
    def cache(self):
        return {"mu": np.array([[.1, .002], [.11, .003], [.002, .2], [.003, .21]]),
                "labels": np.array([0, 0, 1, 1]), "sample_ids": np.array([8, 9, 20, 21]),
                "proxy_ball": np.array([[.105, .0025], [.0025, .205], [-.2, .2]]),
                "proxy_eligible": np.array([True, True, False])}

    def test_self_excluded_and_same_class_nearest(self):
        data = self.cache()
        hyp, angle, distance = all_sample_nearest(data["mu"], block_size=2)
        self.assertFalse(np.any(hyp == np.arange(4)))
        self.assertFalse(np.any(angle == np.arange(4)))
        self.assertTrue(np.all(data["labels"][hyp] == data["labels"]))
        self.assertTrue(np.all(distance > 0))

    def test_class_compactness_against_foreign_pairs(self):
        report = analyze_geometry(self.cache(), seed=22, cross_pairs=100, block_size=2)
        classes = report["class_geometry"]
        self.assertEqual(classes["exact_within_distinct_pair_count"], 2)
        self.assertEqual(classes["sampled_cross_pair_count"], 200)
        self.assertLess(classes["pooled"]["within_hyperbolic"]["max"], classes["pooled"]["cross_hyperbolic"]["min"])
        self.assertEqual(classes["all_sample_nearest"]["hyperbolic_same_class_fraction"], 1.)

    def test_equal_top4_sets_do_not_imply_identical_proxies(self):
        proxies = np.array([[.2, .01], [.01, .2]])
        neighbors = np.array([[1, 2, 3, 4], [4, 2, 3, 1]])
        geometry = proxy_geometry(proxies, neighbors, np.array([True, True]))
        same = geometry["same_unordered_top4"]
        self.assertEqual(same["hyperbolic_distance"]["count"], 1)
        self.assertGreater(same["hyperbolic_distance"]["min"], .1)
        self.assertGreater(same["angle_degrees"]["min"], 80)
        self.assertEqual(same["distance_le_1e_minus6_pair_count"], 0)

    def test_sample_to_proxy_is_not_proxy_to_sample_occupancy(self):
        data = self.cache()
        result = proxy_sample_neighbours(data["mu"], data["proxy_ball"], data["sample_ids"], block_size=1)
        # Every proxy has four samples to retrieve, yet no sample is assigned
        # to proxy2. The two directions of this relation have different counts.
        self.assertEqual(result["raw_ids"].shape, (3, 4))
        np.testing.assert_array_equal(np.bincount(result["sample_to_proxy_index"], minlength=3), [2, 2, 0])

    def test_duplicate_ids_rejected_and_order_canonical(self):
        data = self.cache()
        duplicate = {**data, "sample_ids": np.array([8, 8, 20, 21])}
        with self.assertRaises(ValueError):
            validated_cache(duplicate)
        permutation = np.array([3, 0, 2, 1])
        shuffled = {**data, **{key: data[key][permutation] for key in ("mu", "labels", "sample_ids")}}
        a = analyze_geometry(data, cross_pairs=50)
        b = analyze_geometry(shuffled, cross_pairs=50)
        self.assertEqual(a["class_geometry"], b["class_geometry"])

    def test_radial_distance_units(self):
        d, _angle = pair_matrices(np.array([[.1, 0.]]), np.array([[.2, 0.]]))
        self.assertAlmostEqual(d[0, 0], 2 * (np.arctanh(.2) - np.arctanh(.1)), places=13)


if __name__ == "__main__":
    unittest.main()
