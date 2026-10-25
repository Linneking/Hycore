"""Aggregate-only plot gates: denominator validation and completed-table rendering."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


a = module("audit85_test_analysis", "analyze.py")
p = module("audit85_test_plot", "plot.py")


class PlotContracts(unittest.TestCase):
    def test_histogram_and_cdf_denominators_are_checked(self):
        hist = [{"observation_id": "test__best", "domain": "native_whole", "percent": "100"}]
        cdf = [{"observation_id": "test__best", "domain": "native_whole", "depth": "0", "cumulative_percent": "0"},
               {"observation_id": "test__best", "domain": "native_whole", "depth": "1", "cumulative_percent": "100"}]
        p.validate_distributions(hist, cdf)
        with self.assertRaises(ValueError):
            p.validate_distributions([{**hist[0], "percent": "50"}], cdf)
        with self.assertRaises(ValueError):
            p.validate_distributions(hist, [cdf[1], {**cdf[0], "depth": "2"}])
        self.assertTrue(np.isnan(p.percentage(None)))

    def test_complete_public_export_renders_png_svg_without_private_inputs(self):
        labels = np.repeat(np.arange(40), 2)
        rng = np.random.RandomState(198)
        mu = a.unit_directions(rng.randn(80, 5)) * .8
        with tempfile.TemporaryDirectory(prefix="_plot85_contract_", dir=HERE) as temporary:
            root = Path(temporary)
            np.savez(root / "h0.npz", native_mu=mu, labels=labels, canonical_ids=np.arange(80))
            view = a.unit_directions(mu) * (np.tanh(np.sqrt(.1) * 4.6 / 2) / np.sqrt(.1))
            proxies = a.unit_directions(rng.randn(24, 5)) * 1.5
            np.savez(root / "h1.npz", native_mu=mu, labels=labels, canonical_ids=np.arange(80),
                     hier_mu=view, proxy_ball=proxies, sample_usage=np.arange(24) % 5,
                     proxy_usage=np.arange(24) % 7)
            manifest = {"observations": [{"key": "ts13", "run_id": "TS13", "stage": "best", "epoch": 223,
                                          "c_native": .1, "cache": "h0.npz", "primary": True},
                                         {"key": "b05", "run_id": "B05_A1", "stage": "best", "epoch": 273,
                                          "c_native": .1, "c_hier": .1, "cache": "h1.npz", "primary": True}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            a.main(["--manifest", str(root / "manifest.json"), "--out", str(root / "statistics"),
                    "--expected-count", "80", "--sample-batches", "1"])
            # Remove the raw cache before plotting: public statistics alone are sufficient.
            (root / "h0.npz").unlink()
            (root / "h1.npz").unlink()
            figures = p.Figures(root / "statistics", root / "figures")
            figures.depth_figures()
            figures.class_figures()
            figures.representative_figures()
            figures.proxy_figures()
            self.assertGreaterEqual(len(figures.files), 18)
            for item in figures.files:
                path = root / "figures" / item["file"]
                self.assertGreater(path.stat().st_size, 500)
            summary = json.loads((root / "statistics/summary.json").read_text())
            self.assertIn("class_cone_radius_p95", summary["observations"][0]["angles"])
            self.assertFalse(summary["observations"][0]["proxies"]["available"])


if __name__ == "__main__":
    unittest.main()
