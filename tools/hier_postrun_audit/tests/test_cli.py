"""CLI isolation contracts: immutable source and explicitly partial evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.hier_postrun_audit.__main__ import main


class CliIsolationTests(unittest.TestCase):
    def source(self, root, status="completed"):
        run = root / "v7_run"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({
            "status": status, "completed_epochs": 1,
            "training_config": {"seed": 22, "steps_per_epoch": 2, "c": 1, "warmup_epochs": 0}}))
        (run / "metrics.jsonl").write_text(json.dumps({
            "epoch": 1, "phase": "joint", "telemetry": {"batches": 2},
            "validation": {"val_oa": 91.0}}) + "\n")
        return run

    def argv(self, run, out, extra=()):
        return ["audit", "run", "--run-dir", str(run), "--out-dir", str(out), "--no-plots", *extra]

    def test_cpu_audit_preserves_source_bytes_and_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = self.source(root); out = root / "report"
            before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in run.iterdir()}
            with patch("sys.argv", self.argv(run, out)), patch("builtins.print"):
                main()
            after = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in run.iterdir()}
            self.assertEqual(before, after)
            result = json.loads((out / "audit_manifest.json").read_text())
            self.assertTrue(result["read_only_source"])
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertFalse(result["new_test_forward"])
            self.assertEqual(result["status"], "completed")
            normalized = json.loads((out / "normalized_runs.json").read_text())
            self.assertIsNone(normalized[0]["epochs"][0]["metrics"]["proxy_depth_median"])

    def test_busy_or_failed_source_requires_explicit_partial_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = self.source(root, "running"); out = root / "report"
            with patch("sys.argv", self.argv(run, out)):
                with self.assertRaises(ValueError):
                    main()
            self.assertFalse(out.exists())
            with patch("sys.argv", self.argv(run, out, ["--allow-incomplete"])), patch("builtins.print"):
                main()
            self.assertEqual(json.loads((out / "audit_manifest.json").read_text())["status"], "partial")

    def test_report_cannot_be_written_inside_source_or_over_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = self.source(root)
            with patch("sys.argv", self.argv(run, run / "audit")):
                with self.assertRaises(ValueError):
                    main()
            out = root / "old_report"; out.mkdir()
            with patch("sys.argv", self.argv(run, out)):
                with self.assertRaises(FileExistsError):
                    main()

    def test_gpu_export_is_not_implicit_or_underspecified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = self.source(root); out = root / "report"
            with patch("sys.argv", self.argv(run, out, ["--extract"])):
                with self.assertRaises(ValueError):
                    main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
