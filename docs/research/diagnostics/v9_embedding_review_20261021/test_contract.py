"""Audit contract tests: geometry units, identity alignment, and missing evidence."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

import analyze


class AuditContract(unittest.TestCase):
    def test_cap_is_depth_4_6_in_both_curvatures(self):
        for c in (1., .1):
            radius = np.tanh(np.sqrt(c) * 2.3) / np.sqrt(c)
            points = np.array([[radius, 0., 0.], [0., radius, 0.]])
            depth, q = analyze.radial(points, c)
            np.testing.assert_allclose(depth, [4.6, 4.6], rtol=1e-12)
            self.assertTrue(np.all(q < 1))
        # A c=.1 coordinate radius above one is valid; it cannot be read as c=1.
        points = np.array([[1.9, 0., 0.]])
        analyze.radial(points, .1)
        with self.assertRaises(ValueError):
            analyze.radial(points, 1.)

    def test_quartet_order_does_not_hide_redundancy(self):
        nn = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [4, 1, 2, 3]])
        report, slots = analyze.retrieval(nn, np.arange(6.), np.array([0, 0, 0, 0, 1, 1]))
        self.assertEqual(report["maximum_repeated_quartet"], 2)
        self.assertEqual(report["covered_samples"], 5)
        self.assertEqual(int(slots.sum()), 12)
        flat, _ = analyze.retrieval(nn, np.full(6, 4.6), np.zeros(6))
        self.assertFalse(flat["bottom_1pct_meaningful_radial_order"])
        self.assertEqual(flat["bottom_depth_boundary_tie_n"], 6)

    def make_export(self, root, reorder=False, break_identity=False, bad_usage_epoch=False):
        rng = np.random.default_rng(10)
        unit = rng.normal(size=(8, 3)); unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        punit = rng.normal(size=(4, 3)); punit /= np.linalg.norm(punit, axis=1, keepdims=True)
        labels, ids = np.repeat([0, 1], 4), np.arange(8)
        native, hier = unit * .5, unit * (np.tanh(np.sqrt(.1) * 2.3) / np.sqrt(.1))
        np.savez(root / "base.npz", native_mu=native, labels=labels, canonical_ids=ids)
        order = np.arange(8)[::-1] if reorder else np.arange(8)
        selected_ids = ids[order].copy()
        if break_identity:
            selected_ids[0] = 100
        np.savez(root / "hier.npz", native_mu=native[order], hier_mu=hier[order],
                 labels=labels[order], canonical_ids=selected_ids,
                 proxy_ball=punit * 1.5, proxy_tangent=punit * 2,
                 nn4=np.tile(np.arange(4), (4, 1)))
        common = dict(label="synthetic", stage="best", epoch=5, c_native=1., native_cap_tangent=None)
        observations = [dict(common, key="base", kind="baseline", cache="base.npz"),
                        dict(common, key="hier", kind="hier", c_hier=.1, cache="hier.npz")]
        if bad_usage_epoch:
            observations[1]["usage_source_epoch"] = 6
        (root / "export_manifest.json").write_text(json.dumps(dict(observations=observations)), encoding="utf-8")

    def invoke(self, root):
        old = sys.argv
        try:
            sys.argv = ["analyze.py", "--export-root", str(root), "--out", str(root / "out")]
            with contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
        finally:
            sys.argv = old

    def test_permuted_identity_alignment_and_missing_usage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder); self.make_export(root, reorder=True); self.invoke(root)
            result = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(result["observations"][0]["proxy"])
            proxy = result["observations"][1]["proxy"]
            self.assertIsNone(proxy["usage"]["sample"])
            self.assertNotIn("sample_used", proxy["groups"])
            with np.load(root / "out" / "plot_data.npz") as plot:
                self.assertTrue(all(plot[k].ndim == 1 for k in plot.files))
                self.assertFalse(any("canonical" in k or "embedding" in k for k in plot.files))
            header = (root / "out" / "hotspots.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertNotIn("sample_id", header)
            self.assertNotIn("row_index", header)

    def test_different_object_pool_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder); self.make_export(root, break_identity=True)
            with self.assertRaisesRegex(ValueError, "same canonical"):
                self.invoke(root)

    def test_wrong_epoch_usage_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder); self.make_export(root, bad_usage_epoch=True)
            with self.assertRaisesRegex(ValueError, "own epoch"):
                self.invoke(root)

    def test_two_stages_same_model_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder); self.make_export(root)
            manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
            best = manifest["observations"][1]
            endpoint = dict(best, stage="e300", epoch=300)
            manifest["observations"] = [best, endpoint]
            (root / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.invoke(root)
            result = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([r["key"] for r in result["observations"]], ["hier", "hier"])
            self.assertEqual([r["observation_id"] for r in result["observations"]], ["hier__best", "hier__e300"])
            with np.load(root / "out" / "plot_data.npz") as plot:
                self.assertIn("hier__best__native__whole__all__depth", plot.files)
                self.assertIn("hier__e300__native__whole__all__depth", plot.files)


if __name__ == "__main__":
    unittest.main()
