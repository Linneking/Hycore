"""Small CPU checks for retrieval identity, geometry and offline artefacts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from visualize_v5_proxy_neighbors import (
    CLASS_NAMES, ancestor_usage_report, choose_proxies, nearest_distinct_ids,
    poincare_distances, proxy_graph, render_html, render_png, validate_checkpoint,
)


class ProxyNeighbourTests(unittest.TestCase):
    def test_distance_units_and_ball_validation(self):
        x = np.array([[0., 0.], [.2, 0.]])
        distance = poincare_distances(x, x)
        self.assertAlmostEqual(distance[0, 1], 2 * np.arctanh(.2), places=13)
        np.testing.assert_array_equal(np.diag(distance), [0., 0.])
        np.testing.assert_array_equal(distance, distance.T)
        with self.assertRaises(ValueError):
            poincare_distances(np.array([[1., 0.]]), x)

    def test_nearest_slots_have_distinct_ids_and_stable_ties(self):
        ids = np.array([8, 8, 4, 12, 2])
        distances = np.array([[.1, .05, .2, .2, .4], [.3, .3, .2, .1, .5]])
        positions = nearest_distinct_ids(distances, ids, 4)
        np.testing.assert_array_equal(positions, [[1, 2, 3, 4], [3, 2, 0, 4]])
        self.assertTrue(all(len(set(ids[row])) == 4 for row in positions))
        with self.assertRaises(ValueError):
            nearest_distinct_ids(distances, ids, 5)

    def test_selection_is_seeded_without_replacement(self):
        eligible = np.arange(512) % 3 == 0
        a, b = choose_proxies(eligible, 10, 22), choose_proxies(eligible, 10, 22)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 10)
        self.assertTrue(eligible[a].all())
        with self.assertRaises(ValueError):
            choose_proxies([True, False], 2, 22)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is only installed in server runtime")
    def test_proxy_graph_retains_source_two_nonself_rule(self):
        import torch
        proxies = torch.tensor([[0., 0.], [.01, 0.], [.02, 0.], [.7, 0.], [.8, 0.]])
        graph = proxy_graph(proxies, topk=3, seed=22)
        self.assertFalse(np.diag(graph["mutual"]).any())
        np.testing.assert_array_equal(graph["eligible"], (graph["degree"] >= 2) & graph["negative"].any(1))
        self.assertTrue(np.diag(graph["negative"]).all())
        self.assertTrue(graph["eligible"][:3].all())

    def test_class_order_and_historical_usage_not_fabricated(self):
        self.assertEqual(len(CLASS_NAMES), 40)
        self.assertEqual(CLASS_NAMES[21], "mantel")
        self.assertEqual(CLASS_NAMES[39], "xbox")
        result = ancestor_usage_report({"epoch": 200, "metrics": {"telemetry": {"hierarchy": {
            "sample_graph": {"counts": {"triplets": 123, "active_triplets": 7}}}}}})
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["per_proxy_pair_selected_counts"])
        self.assertEqual(result["epoch_aggregate_activity_only"]["sample_graph"]["triplets"], 123)

    def test_split_identity_rejects_selection_only_checkpoint(self):
        with self.assertRaises(ValueError):
            validate_checkpoint({"format": "hycore-hier-v5-h20-1", "epoch": 200,
                                 "model_selection_only": True})

    def test_offline_html_and_png_contain_all_requested_cells(self):
        from PIL import Image
        rng = np.random.default_rng(7)
        clouds = rng.normal(size=(10, 4, 128, 3)).astype(np.float32)
        rows = [{"proxy_id": i, "proxy_radius": .9, "reciprocal_degree": 3,
                 "neighbors": [{"sample_id": i * 4 + j, "label": j,
                                "class_name": CLASS_NAMES[j], "distance": float(j + 1)}
                               for j in range(4)]} for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            render_png(directory / "grid.png", rows, clouds)
            render_html(directory / "grid.html", rows, clouds, {"split": "train_ids", "sample_count": 8856})
            page = (directory / "grid.html").read_text(encoding="utf-8")
            self.assertNotIn('<script src=', page)
            self.assertNotIn('https://', page)
            packed = page.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
            payload = json.loads(packed)
            self.assertEqual(len(payload["rows"]), 10)
            self.assertEqual(np.asarray(payload["clouds"]).shape, (10, 4, 128, 3))
            with Image.open(directory / "grid.png") as image:
                self.assertGreaterEqual(image.width, 1400)
                self.assertGreaterEqual(image.height, 3500)


if __name__ == "__main__":
    unittest.main()
