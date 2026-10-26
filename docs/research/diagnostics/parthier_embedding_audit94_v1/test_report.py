"""Publication-boundary and missing-data checks; no scientific report is emitted."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("audit94_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture(count=2):
    stat = {"count": count, "mean": 1., "min": .1, "p1": .1, "p5": .1,
            "p25": .5, "median": 1., "p75": 1.5, "p90": 1.8,
            "p95": 1.9, "p99": 1.9, "max": 2.}
    gap = {**stat, "margin": .2, "basic_fraction": 1., "margin_fraction": .5,
           "mean_hinge": .02, "near_zero_fraction": 0., "near_margin_fraction": 0.}
    models = {}
    for label in report.ORDER:
        models[label] = {
            "depth": {view: dict(stat) for view in report.VIEWS},
            "chain": {"count": count, "joint_basic_fraction": 1., "joint_margin_fraction": .5,
                      "edges": {edge: dict(gap) for edge in ("200_400", "400_600", "600_whole")}},
            "intra": {str(scale): {term: dict(gap) for term in ("radial", "contrastive")}
                      for scale in (200, 400, 600)},
        }
        if label in report.PH_ORDER:
            models[label]["relations"] = {
                "draws": 10, "valid_draws": 8, "collisions": 2,
                "collision_fraction": .2, "radial_and_distance_margin_fraction": .25,
                "ord": {"edges": {edge: dict(gap) for edge in ("A_i", "A_j", "A_minij", "B_A", "B_k")},
                        "joint_basic_fraction": .75, "joint_margin_fraction": .5},
                "hier": {"edges": {edge: dict(gap) for edge in ("i", "j", "k")},
                         "joint_basic_fraction": .5, "joint_margin_fraction": .25},
                "roles": {role: {"count": 8, "scale_fractions": [.8, .1, .1],
                                  "current_fraction": .1, "queue_fraction": .9,
                                  "self_source_fraction": .1, "unique_queue_entries": 2,
                                  "unique_live_occurrences": 1, "depth": dict(stat)}
                          for role in ("A", "B")},
                "angles": {group: {"count": 2, "nearest": dict(stat),
                                    "descendant_angles": dict(stat), "second_gap": dict(stat),
                                    "nearest_in_descendant_fraction": .5}
                           for group in report.ANGLE_GROUPS},
            }
    return {"version": "synthetic-test-only", "models": models}


class PublicationChecks(unittest.TestCase):
    def test_missing_or_nonfinite_values_are_not_successful_zeroes(self):
        self.assertEqual(report.pct(None), "—")
        self.assertEqual(report.fmt(float("nan")), "—")
        self.assertEqual(report.pct(.125), "12.50%")

    def test_partial_object_counts_are_rejected_by_default(self):
        summary = fixture()
        with self.assertRaisesRegex(ValueError, "expected 9840"):
            report.validate_summary(summary)
        self.assertEqual(set(report.validate_summary(summary, expected_count=2)), set(report.ORDER))

    def test_collision_denominators_must_balance(self):
        summary = fixture()
        summary["models"]["TS18_best"]["relations"]["valid_draws"] = 7
        with self.assertRaisesRegex(ValueError, "denominators"):
            report.validate_summary(summary, expected_count=2)

    def test_aggregate_copy_drops_private_columns(self):
        models = fixture()["models"]
        # Use this test's workspace, whose resolved parent is explicit, rather
        # than the desktop host's separately restricted temporary directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            source, output = root / "private", root / "public"
            source.mkdir(); output.mkdir()
            schemas = {
                "depth_hist.csv": ["model", "view", "bin_left", "bin_right", "count", "percentage"],
                "depth_ecdf.csv": ["model", "view", "depth", "percentage"],
                "angle_hist.csv": ["model", "group", "metric", "bin_left", "bin_right", "count", "percentage"],
            }
            for filename, columns in schemas.items():
                with (source / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns + ["object_id", "private_path"])
                    writer.writeheader()
                    writer.writerow({**{key: 1 for key in columns}, "object_id": "private-object", "private_path": "/private/checkpoint"})
            report.publication_tables(source, output, models)
            for filename in schemas:
                text = (output / "tables" / filename).read_text(encoding="utf-8-sig")
                self.assertNotIn("private-object", text)
                self.assertNotIn("private_path", text)
                self.assertNotIn("object_id", text)

    def test_report_keeps_queue_and_causal_limits_explicit(self):
        text = report.report_markdown(fixture()["models"])
        self.assertIn("不是精确训练重放", text)
        self.assertIn("不是匹配H0", text)
        self.assertIn("只以A≠B", text)
        self.assertIn("训练对象", text)
        self.assertNotIn("/mnt/", text)
        self.assertNotIn("D:\\", text)

    def test_public_summary_drops_incidental_checkpoint_paths(self):
        models = fixture()["models"]
        models["TS18_best"]["checkpoint_path"] = "/private/model.pth"
        self.assertNotIn("/private/model.pth", json.dumps(report.public_models(models)))


if __name__ == "__main__":
    unittest.main()
