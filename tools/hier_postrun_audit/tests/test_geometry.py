"""Analytic geometry and identity counterexamples, without model/GPU dependencies."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.hier_postrun_audit.geometry import (ball_geometry, expmap0, poincare_distance,
                                             pool_hash, topk_neighbours)
from tools.hier_postrun_audit.snapshots import analyze_snapshots


class GeometryTests(unittest.TestCase):
    def test_curvature_changes_coordinate_radius_not_origin_depth_for_same_tangent(self):
        tangent = np.array([[2.3, 0.]])
        for c in (.1, 1., 2.):
            points = expmap0(tangent, c)
            measured = ball_geometry(points, c)
            self.assertAlmostEqual(measured["depth"][0], 4.6, places=11)
            self.assertAlmostEqual(measured["radius"][0], np.tanh(np.sqrt(c) * 2.3) / np.sqrt(c), places=12)
        self.assertGreater(np.linalg.norm(expmap0(tangent, .1)), np.linalg.norm(expmap0(tangent, 1.)))

    def test_full_geodesic_matches_independent_same_ray_and_opposite_ray_answers(self):
        for c in (.1, 1.):
            first = np.array([[.999 / np.sqrt(c), 0.]])
            same = np.array([[.996 / np.sqrt(c), 0.]])
            other = -first
            expected_same = 2. / np.sqrt(c) * (np.arctanh(.999) - np.arctanh(.996))
            expected_opposite = 4. / np.sqrt(c) * np.arctanh(.999)
            self.assertAlmostEqual(poincare_distance(first, same, c)[0, 0], expected_same, places=10)
            self.assertAlmostEqual(poincare_distance(first, other, c)[0, 0], expected_opposite, places=10)
            self.assertEqual(poincare_distance(first, first, c)[0, 0], 0.)

    def test_numeric_projection_hides_saved_tangent_norm_overshoot(self):
        tangent = np.array([[4., 0.], [9., 0.]])
        points = expmap0(tangent, 1., numeric_radius_fraction=.999)
        np.testing.assert_allclose(points[0], points[1], atol=1e-15)
        depths = ball_geometry(points, 1.)["depth"]
        np.testing.assert_allclose(depths, 2. * np.arctanh(.999))
        self.assertLess(depths[1], 2. * np.linalg.norm(tangent[1]))

    def test_radial_bias_can_reverse_directional_nearest_neighbour(self):
        proxy = np.array([[.1, 0.]])
        candidates = np.array([[.98, 0.], [.2 * np.cos(.7), .2 * np.sin(.7)]])
        ids = np.array([100, 200])
        raw, _ = topk_neighbours(proxy, candidates, ids, 1., k=1)
        angle, _ = topk_neighbours(proxy, candidates, ids, 1., k=1, metric="direction")
        self.assertEqual(ids[raw[0, 0]], 200)
        self.assertEqual(ids[angle[0, 0]], 100)
        # Equal radii remove this ranking reversal while preserving directions.
        equal, _ = topk_neighbours(proxy, candidates, ids, 1., k=1,
                                   metric="equal_radius_hyperbolic", equal_radius_fraction=.5)
        self.assertEqual(ids[equal[0, 0]], 100)

    def test_same_object_pool_hash_ignores_row_order_but_not_identity(self):
        self.assertEqual(pool_hash(np.array([3, 1, 2])), pool_hash(np.array([2, 3, 1])))
        self.assertNotEqual(pool_hash(np.array([3, 1, 2])), pool_hash(np.array([3, 1, 9])))

    def _cache(self, root, name, reordered=False, changed_pool=False, proxy=True):
        ids = np.array([10, 20, 30, 40, 50, 60])
        angles = np.array([0., .1, -.1, 2.9, 3., 3.1])
        whole = .5 * np.column_stack([np.cos(angles), np.sin(angles)])
        labels = np.array([0, 0, 0, 1, 1, 1])
        arrays = {"sample_ids": ids, "mu": whole, "labels": labels}
        if changed_pool:
            arrays["sample_ids"] = np.array([10, 20, 30, 40, 50, 999])
        if proxy:
            arrays.update(proxy_ids=np.array([7, 3]), proxy_ball=np.array([[.3, 0.], [-.3, 0.]]))
        if reordered:
            order = np.array([5, 0, 3, 2, 1, 4])
            for key in ("sample_ids", "mu", "labels"):
                arrays[key] = arrays[key][order]
            if proxy:
                arrays["proxy_ids"] = arrays["proxy_ids"][::-1]
                arrays["proxy_ball"] = arrays["proxy_ball"][::-1]
        path = root / name
        np.savez(path, **arrays)
        return path

    def test_top4_stability_uses_sample_and_proxy_ids_not_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self._cache(root, "first.npz")
            second = self._cache(root, "second.npz", reordered=True)
            specs = [{"epoch": epoch, "cache": str(cache), "run_key": "same_run", "c": 1., "inputmode": "clean_eval", "input_sha256": "same_input_sha"}
                     for epoch, cache in ((1, first), (2, second))]
            result = analyze_snapshots(specs, root / "new_report")
            transition = result["transitions"][0]
            self.assertTrue(transition["available"])
            self.assertEqual(transition["retrieval"]["hyperbolic"]["mean_topk_jaccard"], 1.)
            self.assertEqual(transition["retrieval"]["hyperbolic"]["turnover"], 0.)
            self.assertAlmostEqual(transition["proxy"]["radial_depth_change"]["p50"], 0.)
            persisted = (root / "new_report" / "snapshot_summary.json").read_text()
            self.assertNotIn(str(root), persisted)

    def test_changed_candidate_pool_refuses_turnover(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self._cache(root, "first.npz")
            second = self._cache(root, "second.npz", changed_pool=True)
            specs = [{"epoch": epoch, "cache": str(cache), "run_key": "same_run", "c": 1., "inputmode": "clean_eval", "input_sha256": "same_input_sha"}
                     for epoch, cache in ((1, first), (2, second))]
            result = analyze_snapshots(specs, root / "new_report")
            self.assertFalse(result["transitions"][0]["available"])
            self.assertIn("pools differ", result["transitions"][0]["reason"])

    def test_same_ids_do_not_hide_different_pointcloud_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = self._cache(root, "cache.npz")
            specs = [{"epoch": epoch, "cache": str(cache), "run_key": "same_run", "c": 1.,
                      "input_mode": "clean_eval", "input_sha256": sha}
                     for epoch, sha in ((1, "input_one"), (2, "input_two"))]
            result = analyze_snapshots(specs, root / "new_report")
            self.assertFalse(result["transitions"][0]["available"])
            self.assertIn("input SHA256 differs", result["transitions"][0]["reason"])

    def test_same_inputs_require_matching_declared_inference_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = self._cache(root, "cache.npz")
            specs = [{"epoch": epoch, "cache": str(cache), "run_key": "same_run", "c": 1.,
                      "input_mode": "clean_eval", "input_sha256": "same_actual_inputs",
                      "inference_condition": condition}
                     for epoch, condition in ((1, {"rng_seed": 22, "FPS_kernel_source_sha256": "source_one"}),
                                              (2, {"rng_seed": 22, "FPS_kernel_source_sha256": "source_two"}))]
            result = analyze_snapshots(specs, root / "different_condition_report")
            self.assertFalse(result["transitions"][0]["available"])
            self.assertIn("Inference conditions differ", result["transitions"][0]["reason"])
            del specs[1]["inference_condition"]
            result = analyze_snapshots(specs, root / "missing_condition_report")
            self.assertFalse(result["transitions"][0]["available"])
            self.assertIn("Inference conditions differ", result["transitions"][0]["reason"])

    def test_baseline_geometry_and_missing_cache_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = self._cache(root, "baseline.npz", proxy=False)
            specs = [{"epoch": 3, "cache": str(baseline), "run_key": "B64", "c": 1.},
                     {"epoch": 5, "cache": str(root / "missing.npz"), "run_key": "B64", "c": 1.}]
            result = analyze_snapshots(specs, root / "new_report")
            self.assertEqual(len(result["snapshots"]), 1)
            self.assertEqual(result["snapshots"][0]["whole"]["count"], 6)
            self.assertFalse(result["snapshots"][0]["proxy"]["available"])
            self.assertFalse(result["availability"][1]["available"])
            with self.assertRaises(FileExistsError):
                analyze_snapshots([], root / "new_report")


    def test_zero_direction_rejects_angular_controls_without_fabricating_neighbours(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "origin.npz"
            np.savez(cache, mu=np.array([[.1, 0.], [-.1, 0.], [.2, .1], [.3, .1]]),
                     sample_ids=np.arange(4), labels=np.array([0, 0, 1, 1]),
                     proxy_ball=np.array([[0., 0.], [.15, .1]]), proxy_ids=np.arange(2))
            result = analyze_snapshots([{"cache": str(cache), "epoch": 1, "c": 1,
                "run_key": "test", "equal_radius_fraction": .5}], root / "audit")
            retrieval = result["snapshots"][0]["retrieval"]
            self.assertTrue(retrieval["hyperbolic"]["available"])
            self.assertFalse(retrieval["direction"]["available"])
            self.assertFalse(retrieval["equal_radius_hyperbolic"]["available"])


if __name__ == "__main__":
    unittest.main()
