"""The launch gates reject wrong arms and test-reading smoke manifests."""
import json
from pathlib import Path
import tempfile
import unittest
from .. import queue


class QueueContracts(unittest.TestCase):
    def test_only_three_authorized_single_gpu_commands(self):
        self.assertEqual(len(queue.VARIANTS), 3)
        for variant in queue.VARIANTS:
            cmd = queue.command(variant, Path("/data"), Path("/new"), smoke=True)
            self.assertIn("--smoke", cmd)
            self.assertEqual(cmd[cmd.index("--epochs") + 1], "2")
            self.assertEqual(cmd[cmd.index("--steps-per-epoch") + 1], "2")
            if variant == "v8_nohook":
                self.assertIn("--nproc_per_node=1", cmd)
                self.assertIn("--skip-final-test", cmd)
            else:
                self.assertEqual(cmd[cmd.index("--variant") + 1],
                                 "c01" if variant == "orig_c01" else "c01-cap")
        with self.assertRaises(ValueError):
            queue.command("unapproved", Path("/data"), Path("/new"))

    def test_gpu_filter_never_accepts_busy_or_active_or_low_memory(self):
        def row(**changes):
            value = dict(index=0, uuid="fake", compute_busy=False,
                         utilization=0, free_MiB=32607)
            value.update(changes)
            return value
        self.assertEqual(len(queue.idle_gpus([row()])), 1)
        self.assertEqual(queue.idle_gpus([
            row(compute_busy=True), row(utilization=1), row(free_MiB=29999)]), [])

    def test_original_smoke_rejects_test_read_and_wrong_gamma(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {
                "status": "completed", "completed_epochs": 2,
                "official_test_read": False, "commit": "frozen",
                "train_size": 9840, "validation_size": 0,
                "canonical_name": queue.NAMES["orig_c01"],
                "initialized_model_sha256": "init",
                "training_config": dict(c=.1, radial_gamma=1000,
                    contrastive_margin=4, gradient_hook=False, seed=4780, clip_r=None),
            }
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(queue.verify_smoke(directory, "orig_c01", "frozen")["updates"], 4)
            manifest["official_test_read"] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                queue.verify_smoke(directory, "orig_c01", "frozen")
            manifest["official_test_read"] = False
            manifest["training_config"]["radial_gamma"] = 500
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                queue.verify_smoke(directory, "orig_c01", "frozen")


if __name__ == "__main__":
    unittest.main()
