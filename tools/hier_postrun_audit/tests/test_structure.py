"""Independent shape counterexamples; no training/model/GPU dependencies."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools.hier_postrun_audit.structure import (
    analyze_structure, chamfer, clustered_summary, fixed_points, spearman)


def load_clouds(path):
    with np.load(path) as archive:
        return archive["clouds"].copy()


class StructureTests(unittest.TestCase):
    def test_chamfer_is_symmetric_squared_with_known_translation(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        second = first + [0., 2., 0.]
        self.assertAlmostEqual(chamfer(first, second), 4.)
        self.assertEqual(chamfer(first, second), chamfer(second, first))
        self.assertEqual(chamfer(first, first), 0.)

    def test_normalized_mode_removes_scale_and_translation_but_native_preserves_them(self):
        rng = np.random.default_rng(4)
        cloud = rng.normal(size=(1, 32, 3))
        transformed = cloud * 7. + [10., -20., 50.]
        one, _ = fixed_points(cloud, 12)
        two, _ = fixed_points(transformed, 12)
        np.testing.assert_allclose(one, two, atol=1e-12)
        a, _ = fixed_points(cloud, 12, "native")
        b, _ = fixed_points(transformed, 12, "native")
        self.assertGreater(chamfer(a[0], b[0]), 100.)

    def test_fps_degenerate_cloud_refused(self):
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            fixed_points(np.ones((2, 32, 3)), 12)

    def test_spearman_ties_and_constant_not_fake_zero(self):
        self.assertAlmostEqual(spearman([1, 1, 3, 4], [2, 2, 6, 8]), 1.)
        self.assertAlmostEqual(spearman([1, 1, 3, 4], [8, 8, 2, 0]), -1.)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_bootstrap_object_and_class_units_differ(self):
        summary = clustered_summary([1., 1., 1., -1.], [0, 0, 0, 1], 50)
        self.assertEqual(summary["mean"], .5)
        self.assertEqual(summary["class_balanced_mean"], 0.)
        self.assertEqual(summary["object_count"], 4)
        self.assertEqual(summary["class_count"], 2)

    def _fixture(self, root):
        rng = np.random.default_rng(3)
        clouds, mus, labels = [], [], []
        for label in range(2):
            for index in range(6):
                # Within each class the independent aspect ratio and direction agree.
                cloud = rng.normal(size=(24, 3)) * [1 + index * .4, .3, .2]
                clouds.append(cloud)
                angle = label * np.pi + .12 * index
                mus.append(.5 * np.array([np.cos(angle), np.sin(angle)]))
                labels.append(label)
        ids, labels = np.arange(100, 112), np.asarray(labels)
        pointfile = root / "source_clouds.npz"
        np.savez(pointfile, sample_ids=ids, labels=labels, clouds=np.asarray(clouds))
        scores = np.zeros((12, 2))
        scores[np.arange(12), labels] = 5.
        scores[1] = [0, 6]
        cache = root / "source_cache.npz"
        np.savez(cache, sample_ids=ids, labels=labels, mu=mus, logits=scores)
        specs = [{"cache": str(cache), "epoch": 20, "c": 1, "run_key": "V7_unique",
                  "display_name": "V7-HIER64", "input_sha256": hashlib.sha256(np.asarray(clouds).tobytes()).hexdigest()}]
        # Output must be outside immutable source directory.
        out = root.parent / (root.name + "_audit")
        return pointfile, cache, specs, out

    def test_complete_fixed_query_audit_keeps_sources_and_has_failure_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            before = (pointfile.read_bytes(), cache.read_bytes())
            answer = analyze_structure(specs, pointfile, out, shape_points=12,
                                       per_class=4, bootstrap=20, max_seconds=60)
            self.assertEqual(answer["status"], "completed")
            self.assertEqual(answer["audit_object_count"], 8)
            self.assertTrue(answer["sources_unchanged"])
            self.assertEqual(before, (pointfile.read_bytes(), cache.read_bytes()))
            row = answer["snapshots"][0]
            self.assertEqual(row["candidate_count"], 12)
            self.assertEqual(row["classification"]["error_count"], 1)
            self.assertFalse(row["training_usage_by_object"]["available"])
            for mode in ("native", "center_unit_rms"):
                for method in ("hyperbolic", "direction"):
                    values = row["shape"][mode]["whole_relations"][method]
                    self.assertEqual(values["within_class_spearman"]["object_count"], 8)
                    self.assertEqual(len(values["objects"]), 8)
                    # Four queries/class and k4 retrieve all3 peers: raw overlap1
                    # is exactly chance, so no false adjusted relationship gain.
                    self.assertEqual(values["shape_topk_overlap_expected_random"]["mean"], 1.)
                    self.assertIsNone(values["chance_adjusted_overlap"]["mean"])
            text = (out / "structure_summary.json").read_text()
            self.assertNotIn(str(root), text)
            json.loads(text)
            # Clean separate sibling output ourselves; avoids leaking tempfile sibling.
            import shutil
            shutil.rmtree(out)

    def test_cloud_identity_or_label_mismatch_refused_before_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            with np.load(cache) as values:
                arrays = {name: values[name].copy() for name in values.files}
            arrays["labels"][0] = 9
            np.savez(cache, **arrays)
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                analyze_structure(specs, pointfile, out, shape_points=12)
            self.assertFalse(out.exists())

    def test_different_pools_have_shared_query_identity_not_fake_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            with np.load(cache) as values:
                arrays = {name: values[name].copy()[2:] for name in values.files}
            cache2 = root / "second.npz"
            np.savez(cache2, **arrays)
            specs.append({**specs[0], "cache": str(cache2), "run_key": "V6_unique", "epoch": 40,
                          "input_sha256": hashlib.sha256(np.asarray(load_clouds(pointfile))[2:].tobytes()).hexdigest()})
            answer = analyze_structure(specs, pointfile, out, shape_points=12, per_class=4,
                                       bootstrap=0, normalization="native", max_seconds=60)
            self.assertEqual([row["candidate_count"] for row in answer["snapshots"]], [12, 10])
            self.assertFalse(answer["candidate_pools_match"])
            self.assertNotIn(100, answer["audit_ids"])
            self.assertNotIn(101, answer["audit_ids"])
            import shutil
            shutil.rmtree(out)

    def test_origin_direction_is_unavailable_not_random_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            with np.load(cache) as values:
                arrays = {name: values[name].copy() for name in values.files}
            arrays["mu"][:] = 0.
            np.savez(cache, **arrays)
            answer = analyze_structure(specs, pointfile, out, shape_points=12, per_class=4,
                                       bootstrap=0, normalization="native", max_seconds=60)
            direction = answer["snapshots"][0]["shape"]["native"]["whole_relations"]["direction"]
            self.assertEqual(direction["within_class_spearman"]["object_count"], 0)
            self.assertIsNone(direction["shape_topk_overlap"]["mean"])
            self.assertIsNone(direction["full_pool_chamfer_enrichment"]["mean"])
            import shutil
            shutil.rmtree(out)

    def test_cloud_input_bytes_must_match_cache_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            specs[0]["input_sha256"] = "different_input"
            with self.assertRaisesRegex(ValueError, "bytes do not match"):
                analyze_structure(specs, pointfile, out, shape_points=12)
            self.assertFalse(out.exists())

    def test_cpu_budget_timeout_is_explicit_partial_not_missing_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            calls = [0]
            def clock():
                calls[0] += 1
                return 0. if calls[0] == 1 else 100.
            with patch("tools.hier_postrun_audit.structure.time.monotonic", side_effect=clock):
                answer = analyze_structure(specs, pointfile, out, shape_points=12,
                                           per_class=4, max_seconds=1)
            self.assertEqual(answer["status"], "partial_budget")
            self.assertEqual(answer["snapshots"], [])
            self.assertTrue(answer["sources_unchanged"])
            self.assertTrue(answer["warnings"])
            import shutil
            shutil.rmtree(out)

    def test_proxy_shape_control_preserves_class_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            pointfile, cache, specs, out = self._fixture(root)
            snapshot = {"run_key": "V7_unique", "epoch": 20, "retrieval": {
                "proxy_ids": [999],
                "hyperbolic": {"topk_sample_ids": [[100, 101, 106, 107]]},
                "direction": {"topk_sample_ids": [[100, 102, 108, 109]]}}}
            answer = analyze_structure(specs, pointfile, out, shape_points=12, per_class=4,
                                       bootstrap=0, normalization="native", max_seconds=60,
                                       snapshots={"snapshots": [snapshot]})
            rows = answer["snapshots"][0]["shape"]["native"]["proxy_relations"]
            self.assertTrue(rows["available"])
            proxy = rows["methods"]["hyperbolic"]["proxy_rows"][0]
            self.assertEqual(proxy["proxy_id"], 999)
            self.assertEqual(proxy["labels"], [0, 0, 1, 1])
            self.assertIsNotNone(proxy["class_composition_matched_random_chamfer"])
            self.assertIsNone(rows["methods"]["hyperbolic"]["confidence_interval"])
            import shutil
            shutil.rmtree(out)


if __name__ == "__main__":
    unittest.main()
