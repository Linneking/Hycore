"""Report contracts: observed identities, missing values, privacy and outputs."""
import builtins
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.hier_postrun_audit.plots import save_figures, usage_transition
from tools.hier_postrun_audit.report import render_report


class UsageIdentityTests(unittest.TestCase):
    def record(self, ids, counts):
        return {"proxy_ids": ids, "counts": counts}

    def test_identity_alignment_and_count_scale_invariance(self):
        a = self.record(["p2", "p1", "p0"], [0, 2, 1])
        b = self.record(["p0", "p2", "p1"], [10, 0, 20])
        result = usage_transition(a, b)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["jsd_bits"], 0.0)
        self.assertAlmostEqual(result["rank_spearman"], 1.0)
        self.assertEqual(result["used_set_jaccard"], 1.0)

    def test_disjoint_mass_and_reversed_usage(self):
        result = usage_transition(self.record([0, 1], [5, 0]), self.record([1, 0], [8, 0]))
        self.assertAlmostEqual(result["jsd_bits"], 1.0)
        self.assertAlmostEqual(result["rank_spearman"], -1.0)
        self.assertEqual(result["used_set_jaccard"], 0.0)

    def test_invalid_or_empty_identity_evidence_is_unavailable(self):
        good = self.record([0, 1], [2, 1])
        for other in (self.record([0, 2], [2, 1]), self.record([0, 0], [2, 1]),
                      self.record([0, 1], [0, 0]), self.record([0, 1], [1]),
                      self.record([0, 1], [float("nan"), 1])):
            self.assertFalse(usage_transition(good, other)["available"])
        flat = usage_transition(self.record([0, 1], [1, 1]), self.record([0, 1], [2, 2]))
        self.assertIsNone(flat["rank_spearman"])


class ReportTests(unittest.TestCase):
    def sample(self):
        return {"run_id": "v7_run", "display_name": "V7 <script>alert(1)</script>",
                "identity": {"version": "v7", "seed": 22, "commit": "a" * 40,
                             "config": {"data": "/mnt/private/account/data"}},
                "metadata": {"curvature": 1, "warmup_epochs": 0, "steps_per_epoch": 200},
                "epochs": [{"epoch": 21, "model_updates": 4200, "logged_model_updates": 200,
                            "phase": "joint", "metrics": {"val_oa_pct": 91, "whole_depth_median": 5,
                                                           "proxy_depth_median": None}}],
                "availability": [{"metric": "proxy_depth_median", "status": "missing", "source": [], "reason": "not saved"}],
                "proxy_usage": [], "warnings": ["Missing data in C:\\Users\\privateuser\\secret.npz and /mnt/private/account/run at 192.168.1.7"]}

    @patch("tools.hier_postrun_audit.report.save_figures", return_value={"figures": [], "warnings": []})
    def test_offline_html_and_csv_preserve_missing_history_and_escape_text(self, mock_figures):
        source = self.sample()
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as tmp:
            result = render_report([source], None, Path(tmp))
            document = Path(result["report_path"]).read_text(encoding="utf8")
            data = json.loads((Path(tmp) / "report_data.json").read_text(encoding="utf8"))
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
            self.assertNotIn("<script>alert(1)</script>", document)
            self.assertNotIn("privateuser", document)
            self.assertNotIn("/mnt/private", document)
            self.assertNotIn("192.168.1.7", document)
            self.assertNotIn("https://", document)
            self.assertIn("No supplied retrieval snapshots", document)
            self.assertIn("No matching saved field", document)
            self.assertEqual(len(data["runs"][0]["epochs"]), 1)
            self.assertIsNone(data["runs"][0]["epochs"][0]["metrics"]["proxy_depth_median"])
            with Path(result["table_paths"][0]).open(encoding="utf8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["epoch"], "21")
            self.assertEqual(rows[0]["proxy_depth_median"], "")
            self.assertEqual(rows[0]["logged_model_updates"], "200")
            self.assertEqual(source, original)
            mock_figures.assert_called_once()

    @patch("tools.hier_postrun_audit.report.save_figures", return_value={"figures": [], "warnings": []})
    def test_existing_report_is_never_overwritten(self, mock_figures):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "report.html"
            sentinel.write_text("existing evidence", encoding="utf8")
            with self.assertRaises(FileExistsError):
                render_report([self.sample()], None, root)
            self.assertEqual(sentinel.read_text(encoding="utf8"), "existing evidence")
            mock_figures.assert_not_called()

    def test_missing_plot_dependency_is_explicit(self):
        ordinary_import = builtins.__import__
        def blocked_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("contract test")
            return ordinary_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, "require matplotlib"):
                save_figures([], None, Path(tmp))
            self.assertFalse((Path(tmp) / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
