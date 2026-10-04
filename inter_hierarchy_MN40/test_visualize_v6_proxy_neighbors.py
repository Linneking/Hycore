"""CPU checks for V6 checkpoint identity and the reused top-four workflow."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from visualize_v5_proxy_neighbors import (
    V6_FORMAT, ancestor_usage_report, arguments, proxy_graph, render_html,
    validate_checkpoint,
)


def full_v6_checkpoint(epoch=300):
    train = np.arange(8856, dtype=np.int64)
    validation = np.arange(8856, 9840, dtype=np.int64)
    return {
        "format": V6_FORMAT, "model_selection_only": False,
        "epoch": epoch, "completed_epochs": epoch,
        "net": {"current": 1}, "proxy": {"current": 2},
        "best_net": {"earlier": 3},
        "training_config": {
            "c": 1, "D": 256, "P": 512, "proxy_K": 20,
            "epochs": 300, "global_batch": 64, "steps_per_epoch": 200,
            "self_negative": False, "smoke": False,
        },
        "train_ids": train, "validation_ids": validation,
        "split_sha256": hashlib.sha256(train.tobytes() + validation.tobytes()).hexdigest(),
    }


class V6ProxyNeighbourTests(unittest.TestCase):
    def test_complete_checkpoint_accepts_final_comparable_and_best_epochs(self):
        for epoch in (300, 200, 99):
            saved = full_v6_checkpoint(epoch)
            train, validation = validate_checkpoint(saved, version="v6", expected_epoch=epoch)
            self.assertEqual((len(train), len(validation)), (8856, 984))
            self.assertNotEqual(saved["net"], saved["best_net"])

    def test_rejects_wrong_epoch_version_or_mixed_selection_only_state(self):
        saved = full_v6_checkpoint(99)
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=300)
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6")
        saved["model_selection_only"] = True
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99)
        saved = full_v6_checkpoint(99)
        saved.pop("proxy")
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99)
        saved = full_v6_checkpoint(99)
        saved["completed_epochs"] = 100
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99)
        saved = full_v6_checkpoint(99)
        saved["format"] = "hycore-hier-v5-h20-1"
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99)

    def test_best_path_requires_matching_full_best_epoch_identity(self):
        saved = full_v6_checkpoint(99)
        saved["best"] = {"epoch": 99}
        saved["metrics"] = {"best_updated": True}
        validate_checkpoint(saved, version="v6", expected_epoch=99, checkpoint_name="best.pth")
        saved["best"]["epoch"] = 80
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99, checkpoint_name="best.pth")
        saved["best"]["epoch"] = 99
        saved["metrics"]["best_updated"] = False
        with self.assertRaises(ValueError):
            validate_checkpoint(saved, version="v6", expected_epoch=99, checkpoint_name="best.pth")

    def test_v6_cli_requires_declared_source_epoch(self):
        common = ["--checkpoint", "in.pth", "--data-dir", "data", "--output-dir", "out"]
        with self.assertRaises(SystemExit):
            arguments(common, version="v6")
        parsed = arguments(common + ["--expected-epoch", "99"], version="v6")
        self.assertEqual(parsed.expected_epoch, 99)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is only installed in server runtime")
    def test_v6_proxy_graph_excludes_index_self_from_negative_pool(self):
        import torch
        proxies = torch.tensor([[0., 0.], [.01, 0.], [.02, 0.], [.7, 0.], [.8, 0.]])
        old = proxy_graph(proxies, topk=3, version="v5")
        new = proxy_graph(proxies, topk=3, version="v6")
        self.assertTrue(np.diag(old["negative"]).all())
        self.assertFalse(np.diag(new["negative"]).any())
        np.testing.assert_array_equal(old["eligible"], new["eligible"])

    def test_v6_epoch_usage_is_labeled_as_observed_training_selection(self):
        saved = full_v6_checkpoint(99)
        roles = ("all_draws/pair", "all_draws/triple",
                 "active_noncollision/pair", "active_noncollision/triple")
        saved["metrics"] = {"structure": {
            graph: {"proxy_count": 512,
                    "selected_ancestor_usage": {
                        role: {"counts": [2] * 512, "selection_count": 1024} for role in roles}}
            for graph in ("sample", "proxy")}}
        report = ancestor_usage_report(saved, version="v6")
        self.assertEqual(report["status"], "observed_epoch_aggregate")
        self.assertEqual(report["per_proxy_counts"]["sample"]["all_draws/pair"][0], 2)
        saved["metrics"] = {}
        self.assertEqual(ancestor_usage_report(saved, version="v6")["status"], "unavailable")

    def test_v6_offline_html_displays_actual_epoch(self):
        rows = [{"proxy_id": 1, "proxy_radius": .9, "reciprocal_degree": 3,
                 "historical_ancestor_usage": {"all_draws/pair": 5, "all_draws/triple": 4},
                 "neighbors": [{"sample_id": i, "label": 0,
                                "class_name": "airplane", "distance": float(i + 1)}
                               for i in range(4)]}]
        clouds = np.zeros((1, 4, 4, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "neighbors.html"
            render_html(target, rows, clouds,
                        {"split": "train_ids", "sample_count": 8856,
                         "version_label": "V6-H20", "epoch": 99})
            page = target.read_text(encoding="utf-8")
            self.assertIn("V6-H20 第99轮", page)
            self.assertNotIn("V5-H20 第200轮", page)
            self.assertIn('"all_draws/pair":5', page)
            self.assertIn("若权重含本轮训练结构记录", page)


if __name__ == "__main__":
    unittest.main()
