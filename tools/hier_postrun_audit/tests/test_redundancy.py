"""Analytic proxy partner/identity controls without model or GPU dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.hier_postrun_audit.redundancy import analyze_redundancy, save_redundancy_figures

MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class RedundancyTests(unittest.TestCase):
    def spec(self):
        return {"run_key": "own_run", "display_name": "V7-HIER64", "epoch": 20, "c": 1.,
                "proxy_ids": np.array([80, 4, 9, 11, 99]),
                "proxy_coords": np.array([[.1, 0.], [.1, 0.], [.3, 0.], [-.2, 0.], [0., 0.]])}

    def test_self_exclusion_true_id_ties_origin_na_and_explicit_tolerances(self):
        with tempfile.TemporaryDirectory() as folder:
            answer = analyze_redundancy([self.spec()], Path(folder) / "new", chunk_size=2)
            row = answer["snapshots"][0]
            indices = {identity: index for index, identity in enumerate(row["proxy_ids"])}
            self.assertEqual(row["nearest_geodesic"]["partner_ids"][indices[4]], 80)
            self.assertEqual(row["nearest_geodesic"]["partner_ids"][indices[80]], 4)
            self.assertIsNone(row["nearest_direction"]["partner_ids"][indices[99]])
            self.assertEqual(row["zero_direction_count"], 1)
            self.assertEqual(row["near_pairs"]["coordinate_near_pair_count"], 1)
            self.assertEqual(row["near_pairs"]["geodesic_near_pair_count"], 1)
            self.assertEqual(row["near_pairs"]["direction_near_pair_count"], 3)
            self.assertEqual(row["near_pairs"]["total_unordered_pairs"], 10)
            self.assertEqual(row["near_pairs"]["tolerances"]["angular_radians"], 1e-6)

    def test_small_and_big_chunks_have_identical_numeric_results(self):
        with tempfile.TemporaryDirectory() as folder:
            first = analyze_redundancy([self.spec()], Path(folder) / "one", chunk_size=1)["snapshots"][0]
            second = analyze_redundancy([self.spec()], Path(folder) / "many", chunk_size=32)["snapshots"][0]
            self.assertEqual(first, second)

    def test_single_proxy_nearest_is_unavailable_not_self_distance_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = {**self.spec(), "proxy_ids": [7], "proxy_coords": [[.2, .1]]}
            row = analyze_redundancy([spec], Path(folder) / "one")["snapshots"][0]
            self.assertEqual(row["nearest_geodesic"]["available_proxy_count"], 0)
            self.assertEqual(row["nearest_geodesic"]["partner_ids"], [None])
            self.assertIsNone(row["nearest_geodesic"]["distance_quantiles"]["p50"])
            self.assertEqual(row["nearest_direction"]["available_proxy_count"], 0)

    def test_baseline_without_proxies_is_na(self):
        with tempfile.TemporaryDirectory() as folder:
            row = analyze_redundancy([{"run_key": "B64", "epoch": 20, "c": 1.}],
                                     Path(folder) / "one")["snapshots"][0]
            self.assertFalse(row["available"])
            self.assertIn("baseline", row["reason"])

    def test_parameter_saturation_is_visible_as_coincident_mapped_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = {"run_key": "v6", "epoch": 300, "c": 1., "proxy_ids": [2, 5],
                    "proxy_tangent": [[4., 0.], [9., 0.]],
                    "proxy_mapping": {"numeric_radius_fraction": .999}}
            row = analyze_redundancy([spec], Path(folder) / "one")["snapshots"][0]
            self.assertEqual(row["stored_tangent_norm"]["p100"], 9.)
            self.assertAlmostEqual(row["depth_quantiles"]["p50"], 2 * np.arctanh(.999))
            self.assertEqual(row["near_pairs"]["coordinate_near_pair_count"], 1)

    def test_missing_forward_map_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = {"run_key": "unknown", "epoch": 20, "c": 1., "proxy_tangent": [[1., 0.]]}
            with self.assertRaisesRegex(ValueError, "explicit forward mapping"):
                analyze_redundancy([spec], Path(folder) / "one")
            record = json.loads((Path(folder) / "one" / "redundancy_summary.json").read_text())
            self.assertEqual(record["status"], "failed")

    def test_partner_retention_aligns_true_ids_not_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.spec()
            permutation = np.array([4, 2, 1, 3, 0])
            second = {**first, "epoch": 100, "proxy_coords": first["proxy_coords"][permutation],
                      "proxy_ids": first["proxy_ids"][permutation]}
            answer = analyze_redundancy([first, second], Path(folder) / "one")
            transition = answer["transitions"][0]
            self.assertTrue(transition["available"])
            self.assertEqual(transition["nearest_geodesic_partner_retention"]["fraction"], 1.)
            self.assertEqual(transition["nearest_direction_partner_retention"]["fraction"], 1.)

    def test_ids_are_never_paired_across_runs_and_missing_ids_are_na(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.spec()
            second = {**first, "run_key": "different", "epoch": 100}
            answer = analyze_redundancy([first, second], Path(folder) / "different")
            self.assertEqual(answer["transitions"], [])
            first.pop("proxy_ids")
            answer = analyze_redundancy([first, {**first, "epoch": 100}], Path(folder) / "no_ids")
            self.assertFalse(answer["transitions"][0]["available"])

    def test_two_origins_have_raw_nearest_but_no_invented_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = {**self.spec(), "proxy_ids": [7, 8], "proxy_coords": [[0., 0.], [0., 0.]]}
            row = analyze_redundancy([spec], Path(folder) / "one")["snapshots"][0]
            self.assertEqual(row["nearest_geodesic"]["distances"], [0., 0.])
            self.assertEqual(row["nearest_direction"]["available_proxy_count"], 0)
            self.assertEqual(row["near_pairs"]["valid_direction_unordered_pairs"], 0)

    def test_cache_source_unchanged_and_summary_contains_no_private_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            cache = source / "cache.npz"
            np.savez(cache, proxy_ids=[7, 8], proxy_ball=[[.1, 0.], [-.1, 0.]], c=1.)
            before = cache.read_bytes()
            answer = analyze_redundancy([{"run_key": "own", "epoch": 20, "cache": str(cache)}],
                                        root / "audit")
            self.assertEqual(answer["status"], "completed")
            self.assertEqual(cache.read_bytes(), before)
            self.assertTrue(answer["sources_unchanged"])
            text = (root / "audit" / "redundancy_summary.json").read_text()
            self.assertNotIn(str(source), text)
            json.loads(text)

    @unittest.skipUnless(MATPLOTLIB, "Server validates real PNG/SVG when local matplotlib unavailable")
    def test_report_figure_has_identity_and_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = analyze_redundancy([self.spec()], root / "audit")
            result = save_redundancy_figures(summary, root / "report")
            self.assertEqual(len(result["figures"]), 1)
            figure = result["figures"][0]
            self.assertEqual(figure["run_id"], "own_run")
            for key in ("png", "svg"):
                path = root / "report" / figure[key]
                self.assertGreater(path.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
