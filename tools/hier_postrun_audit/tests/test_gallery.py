"""Actual-object gallery budgets and deterministic proxy representatives."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.hier_postrun_audit.gallery import representative_proxy_rows, save_object_galleries


class GalleryTests(unittest.TestCase):
    def snapshot(self):
        return {"run_key": "fixture", "epoch": 1, "proxy": {"proxy_ids_stable": True},
                "retrieval": {"available": True, "proxy_ids": [20, 21, 22, 23],
                              "hyperbolic": {"topk_sample_ids": [[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 6, 7]]},
                              "direction": {"topk_sample_ids": [[4, 5, 6, 7], [0, 4, 6, 7], [0, 1, 2, 3], [0, 1, 2, 3]]}}}

    def test_distinct_deterministic_proxy_representatives(self):
        selected = representative_proxy_rows(self.snapshot(), [2, 1, 4, 3], [False, False, True, True])
        self.assertEqual([index for index, _ in selected], [0, 1, 2, 3])
        self.assertIn("most repeated", selected[0][1])
        self.assertEqual(len(representative_proxy_rows(self.snapshot(), maximum=2)), 2)

    def test_missing_actual_clouds_do_not_generate_fake_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = save_object_galleries({"snapshots": [self.snapshot()]}, Path(tmp))
            self.assertEqual(result["figures"], [])
            self.assertIn("actual point-cloud", result["warnings"][0])
            self.assertFalse((Path(tmp) / "figures").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed in this test runtime")
    def test_real_3d_gallery_pixel_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(12)
            points = rng.normal(size=(8, 32, 3)).astype(np.float32)
            pool = root / "actual_clouds.npz"
            np.savez(pool, sample_ids=np.arange(8), labels=np.arange(8) % 2, clouds=points)
            result = save_object_galleries({"pointcloud_pool": str(pool), "snapshots": [self.snapshot()]}, root / "out", max_proxies=1, max_snapshots=1)
            self.assertEqual(len(result["figures"]), 1)
            figure = result["figures"][0]
            self.assertGreater((root / "out" / figure["png"]).stat().st_size, 1000)
            self.assertGreater((root / "out" / figure["svg"]).stat().st_size, 1000)
            self.assertIn("not embedding", figure["note"])
            self.assertNotIn(str(pool), str(result))


if __name__ == "__main__":
    unittest.main()
