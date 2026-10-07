"""Figure contracts for genuine compact fixed-query mechanism summaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from tools.hier_postrun_audit.mechanisms_figures import (
    _line, _temporal_value, partial_stats, save_mechanism_figures)

MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def partial(norm=5., radial=3., signed=2., weighted=2.5):
    return {"repeat": 0, "whole_dependency": True,
            "lambda_hier": .5,
            "proxy_tangent": {"gradient_norm": weighted},
            "whole_ball": {"gradient_norm": weighted},
            "unweighted_proxy_tangent": {"gradient_norm": norm, "radial_norm": radial,
                                          "signed_radial_mean": signed, "above_tolerance_row_count": 3,
                                          "radial_direction_undefined_count": 0},
            "unweighted_whole_ball": {"gradient_norm": norm, "radial_norm": radial,
                                      "signed_radial_mean": signed, "above_tolerance_row_count": 3,
                                      "radial_direction_undefined_count": 0}}


def summary():
    agreement = {"mean": .4, "quantiles": {"p05": .1, "p95": .8}}
    rows = []
    for epoch in (20, 100):
        rows.append({"run_key": "v7_unique_key", "epoch": epoch, "display_name": "V7-HIER64",
            "components": {name: {"noise_summary": {"independent_noise_pair_agreement": {
                "pair_proxy_ids_agreement": agreement, "triple_proxy_ids_agreement": agreement}}}
                for name in ("sample", "proxy")},
            "gradient_summaries": {"sample": [partial()],
                                   "proxy": [{**partial(), "whole_dependency": False}]},
            "reference_counterfactual": {name: {
                "distance_absolute_difference_quantiles": {"p50": .02, "p95": .2},
                "same_query_same_noise_operator_agreements": [{"available": True,
                    "pair_proxy_ids_agreement": .8, "triple_proxy_ids_agreement": .9}]}
                for name in ("sample", "proxy")}})
    return {"snapshots": rows, "transitions": [{"run_key": "v7_unique_key", "from_epoch": 20, "to_epoch": 100,
        "available": True, "same_query_same_noise": {name: [{"available": True,
            "pair_proxy_ids_agreement": .7, "triple_proxy_ids_agreement": .6}]
            for name in ("sample", "proxy")}}], "warnings": []}


class MechanismFigureValues(unittest.TestCase):
    def test_radial_fraction_is_energy_and_weighted_norm_remains_separate(self):
        result = partial_stats(summary()["snapshots"][0], "sample", "proxy_tangent")
        self.assertAlmostEqual(result["radial_energy_fraction"], 9 / 25)
        self.assertEqual(result["lambda_weighted_partial_norm"], 2.5)
        self.assertEqual(result["unweighted_signed_radial_mean"], 2.)
        self.assertIn("no AdamW", result["semantics"])

    def test_zero_gradient_ratio_is_missing_not_zero_energy(self):
        row = {"gradient_summaries": {"sample": [partial(norm=0., radial=0., signed=0., weighted=0.)]}}
        result = partial_stats(row, "sample", "proxy_tangent")
        self.assertIsNone(result["radial_energy_fraction"])
        self.assertEqual(result["lambda_weighted_partial_norm"], 0.)

    def test_proxy_whole_gradient_is_explicit_not_applicable(self):
        result = partial_stats(summary()["snapshots"][0], "proxy", "whole_ball")
        self.assertFalse(result["available"])
        self.assertTrue(result["not_applicable"])

    def test_no_evaluated_gradient_does_not_create_false_geometry(self):
        self.assertFalse(partial_stats({}, "sample", "proxy_tangent")["available"])

    def test_incompatible_temporal_transition_never_plotted_as_valid(self):
        incompatible = {"available": False, "same_query_same_noise": {"sample": [{
            "available": True, "pair_proxy_ids_agreement": 1.}]}}
        self.assertIsNone(_temporal_value(incompatible, "sample", "pair_proxy_ids_agreement"))

    def test_undefined_radial_axis_is_unavailable_not_fully_angular(self):
        one = partial()
        one["unweighted_proxy_tangent"]["radial_direction_undefined_count"] = 1
        row = {"gradient_summaries": {"sample": [one]}}
        self.assertIsNone(partial_stats(row, "sample", "proxy_tangent")["radial_energy_fraction"])

    def test_missing_point_breaks_line_instead_of_connecting_false_observations(self):
        class Axis:
            def plot(self, x, y, **kwargs):
                self.x, self.y = x, y
        axis = Axis()
        _line(axis, [20, 40, 100], [.5, None, .7], "own", "blue")
        self.assertEqual(axis.x, [20, 40, 100])
        self.assertTrue(__import__("math").isnan(axis.y[1]))

    def test_detail_schema_is_supported_without_compact_conversion(self):
        row = {"components": {"sample": {"gradient_repeats": [partial()]}}}
        self.assertAlmostEqual(partial_stats(row, "sample", "proxy_tangent")["radial_energy_fraction"], .36)


@unittest.skipUnless(MATPLOTLIB, "Local runtime has no matplotlib; server verifies PNG/SVG")
class MechanismFigureRendering(unittest.TestCase):
    def test_png_svg_and_scientific_scope_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rendered = save_mechanism_figures(summary(), output)
            self.assertEqual(len(rendered["figures"]), 3)
            self.assertEqual({item["kind"] for item in rendered["figures"]},
                             {"controlled_stability", "gradient_partials", "operator_counterfactual"})
            for item in rendered["figures"]:
                self.assertEqual(item["run_id"], "v7_unique_key")
                png, svg = output / item["png"], output / item["svg"]
                self.assertGreater(png.stat().st_size, 10000)
                self.assertGreater(svg.stat().st_size, 10000)
                self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
                self.assertIn("<svg", svg.read_text(encoding="utf-8"))
            gradients = next(item for item in rendered["figures"] if item["kind"] == "gradient_partials")
            self.assertIn("partial", gradients["note"])
            self.assertIn("AdamW", gradients["note"])

    def test_empty_components_produce_warnings_and_no_fake_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            rendered = save_mechanism_figures({"snapshots": [{"run_key": "empty", "epoch": 20}]}, folder)
            self.assertEqual(rendered["figures"], [])
            self.assertEqual(len(rendered["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
