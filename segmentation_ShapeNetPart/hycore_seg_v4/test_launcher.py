"""CPU checks for the predecessor gate and fresh-smoke/fresh-production order."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import launch_experiment as launcher


class PredecessorGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hycore_tail2_gate_")
        self.root = Path(self.temporary.name)
        self.old = self.root / "old"
        (self.old / "main").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, value):
        path = self.old / relative
        path.write_text(json.dumps(value), encoding="utf-8")

    def complete(self):
        self.write("queue.json", {"state": "complete"})
        self.write("main/status.json", {"state": "complete", "epoch": 200})
        self.write("main/acceptance.json", {"complete": True})

    def test_three_success_records_required(self):
        self.assertFalse(launcher.predecessor_state(self.old)["complete"])
        self.write("queue.json", {"state": "complete"})
        self.assertFalse(launcher.predecessor_state(self.old)["complete"])
        self.write("main/status.json", {"state": "complete", "epoch": 200})
        self.assertFalse(launcher.predecessor_state(self.old)["complete"])
        self.write("main/acceptance.json", {"complete": True})
        self.assertTrue(launcher.predecessor_state(self.old)["complete"])

    def test_each_failure_evidence_refuses_replacement(self):
        for relative, value in (("queue.json", {"state": "failed"}),
                                ("main/status.json", {"state": "failed"}),
                                ("main/failure.json", {"error": "fixture"})):
            with self.subTest(relative=relative):
                self.complete()
                self.write(relative, value)
                with self.assertRaisesRegex(RuntimeError, "Predecessor failed"):
                    launcher.predecessor_state(self.old)
                (self.old / relative).unlink()

    def test_partial_json_is_not_completion(self):
        self.complete()
        (self.old / "main/status.json").write_text('{"state":', encoding="utf-8")
        self.assertFalse(launcher.predecessor_state(self.old)["complete"])

    def test_main_directory_is_not_parent_interface(self):
        with self.assertRaisesRegex(ValueError, "run parent"):
            launcher.predecessor_state(self.old / "main")

    def test_wait_updates_only_new_queue(self):
        self.complete()
        before = {p.relative_to(self.old): p.read_bytes() for p in self.old.rglob("*.json")}
        new = self.root / "new"
        new.mkdir()
        pending = {"complete": False, "state": "training", "epoch": 199}
        done = {"complete": True, "state": "complete", "epoch": 200}
        with mock.patch.object(launcher, "predecessor_state", side_effect=[pending, done]), \
                mock.patch.object(launcher.time, "sleep") as sleep:
            self.assertEqual(launcher.wait_for_predecessor(self.old, new), done)
        sleep.assert_called_once_with(30)
        self.assertEqual(json.loads((new / "queue.json").read_text())["state"], "predecessor_complete")
        self.assertEqual(before, {p.relative_to(self.old): p.read_bytes() for p in self.old.rglob("*.json")})

    def test_launch_waits_then_fresh_smoke_then_fresh_production(self):
        target = self.root / "new_run"
        config = Path(launcher.__file__).parent / "configs/tail2.json"
        events = []
        commands = []
        def run(command, **kwargs):
            events.append("subprocess")
            commands.append(command)
        argv = ["--smoke-data", str(self.root / "smoke_data"),
                "--production-data", str(self.root / "production_data"),
                "--root", str(target), "--gpu", "1", "--config", str(config),
                "--wait-for-run", str(self.old)]
        with mock.patch.dict(launcher.os.environ, {}, clear=False), \
                mock.patch.object(launcher, "git_identity", return_value={"source_status": "", "commit": "fixture"}), \
                mock.patch.object(launcher, "wait_for_predecessor", side_effect=lambda *a: events.append("predecessor")), \
                mock.patch.object(launcher, "wait_for_idle", side_effect=lambda *a: events.append("idle")), \
                mock.patch.object(launcher.subprocess, "run", side_effect=run), \
                mock.patch.object(launcher, "validate_smoke", return_value={"initial_model_sha256": "fresh"}), \
                mock.patch.object(launcher, "_json", return_value={"initial_model_sha256": "fresh"}):
            launcher.main(argv)
        self.assertEqual(events, ["predecessor", "idle", "subprocess", "idle", "subprocess"])
        self.assertEqual(len(commands), 2)
        smoke, production = commands
        self.assertEqual(smoke[smoke.index("--protocol") + 1], "validation")
        self.assertIn("--no-final-test", smoke)
        self.assertEqual(smoke[smoke.index("--epochs") + 1], "2")
        self.assertEqual(production[production.index("--protocol") + 1], "antao_testsel")
        self.assertEqual(production[production.index("--epochs") + 1], "200")
        self.assertEqual(production[production.index("--expected-initial-sha256") + 1], "fresh")
        self.assertNotIn("--resume", smoke + production)
        self.assertEqual(json.loads((target / "queue.json").read_text())["state"], "complete")


if __name__ == "__main__":
    unittest.main()
