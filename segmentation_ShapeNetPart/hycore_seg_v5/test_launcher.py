"""CPU checks for the predecessor gate and fresh-smoke/fresh-production order."""

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if __package__:
    from . import launch_experiment as launcher
else:
    spec = importlib.util.spec_from_file_location(
        "pseg5_staged_launcher", Path(__file__).with_name("launch_experiment.py"))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)


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

    def check_launch(self, arm, gpu):
        target = self.root / ("new_run_" + arm)
        config = self.root / ("config_" + arm + ".json")
        alpha, beta = launcher.approved_coefficients(arm)
        config.write_text(json.dumps({"alpha": alpha, "beta": beta}), encoding="utf-8")
        events = []
        commands = []
        def run(command, **kwargs):
            events.append("subprocess")
            commands.append(command)
        argv = ["--smoke-data", str(self.root / "smoke_data"),
                "--production-data", str(self.root / "production_data"),
                "--root", str(target), "--gpu", str(gpu), "--arm", arm, "--config", str(config),
                "--wait-for-run", str(self.old)]
        read_json = launcher._json
        with mock.patch.dict(launcher.os.environ, {}, clear=False), \
                mock.patch.object(launcher, "git_identity", return_value={"source_status": "", "commit": "fixture"}), \
                mock.patch.object(launcher, "wait_for_predecessor", side_effect=lambda *a: events.append("predecessor")), \
                mock.patch.object(launcher, "wait_for_idle", side_effect=lambda *a: events.append("idle")), \
                mock.patch.object(launcher.subprocess, "run", side_effect=run), \
                mock.patch.object(launcher, "validate_smoke", return_value={"initial_model_sha256": "fresh"}), \
                mock.patch.object(launcher, "_json", side_effect=lambda path:
                                  read_json(path) if Path(path).resolve() == config.resolve() else
                                  {"initial_model_sha256": "fresh"}):
            launcher.main(argv)
        self.assertEqual(events, ["predecessor", "idle", "subprocess", "idle", "subprocess"])
        self.assertEqual(len(commands), 2)
        smoke, production = commands
        self.assertIn("segmentation_ShapeNetPart.hycore_seg_v5.train", smoke)
        self.assertEqual(smoke[smoke.index("--arm") + 1], arm)
        self.assertEqual(production[production.index("--arm") + 1], arm)
        self.assertEqual(smoke[smoke.index("--physical-gpu") + 1], str(gpu))
        self.assertEqual(smoke[smoke.index("--protocol") + 1], "validation")
        self.assertIn("--no-final-test", smoke)
        self.assertEqual(smoke[smoke.index("--epochs") + 1], "2")
        self.assertEqual(production[production.index("--protocol") + 1], "antao_testsel")
        self.assertEqual(production[production.index("--epochs") + 1], "200")
        self.assertEqual(production[production.index("--expected-initial-sha256") + 1], "fresh")
        self.assertNotIn("--resume", smoke + production)
        self.assertEqual(json.loads((target / "queue.json").read_text())["state"], "complete")
        dispatch = read_json(target / "dispatch.json")
        self.assertIs(dispatch["regularizers_active"], arm == "d")
        self.assertIs(dispatch["regularizer_diagnostics_only"], arm == "e")

    def test_d_waits_then_fresh_smoke_then_fresh_production(self):
        self.check_launch("d", 0)

    def test_e_waits_then_fresh_smoke_then_fresh_production(self):
        self.check_launch("e", 1)

    def test_wrong_arm_or_mixed_coefficients_refused_before_queue_creation(self):
        for arm, pair in (("d", (0., 0.)), ("e", (.01, .01)),
                          ("d", (.01, 0.)), ("e", (0., .01))):
            with self.subTest(arm=arm, pair=pair):
                config = self.root / "wrong.json"
                config.write_text(json.dumps(dict(zip(("alpha", "beta"), pair))), encoding="utf-8")
                target = self.root / "refused_run"
                with mock.patch.object(launcher, "wait_for_idle") as idle, \
                        mock.patch.object(launcher.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "approved D/E"):
                        launcher.main(["--arm", arm, "--gpu", "0", "--config", str(config),
                                       "--root", str(target), "--smoke-data", "smoke",
                                       "--production-data", "production"])
                self.assertFalse(target.exists())
                idle.assert_not_called()
                run.assert_not_called()


class DisabledRegularizerGateTests(unittest.TestCase):
    def row(self):
        return {"gradients": {name: {
            "loss": 0., "loss_requires_grad": False,
            "nodes": {"h": {"status": "none"}},
            "parameter_groups": {"head": {"status": "none", "numel_with_gradient": 0,
                                              "none_count": 4, "parameter_tensors": 4},
                                 "euclidean9": {"status": "none", "numel_with_gradient": 0,
                                                "none_count": 3, "parameter_tensors": 3}}}
                              for name in ("contr", "hier")}}

    def test_detached_zero_is_accepted(self):
        self.assertTrue(launcher._regularizers_disabled([self.row()]))

    def test_disabled_gate_rejects_loss_or_gradient_leak(self):
        mutations = [lambda item: item.update(loss=1e-20),
                     lambda item: item.update(loss_requires_grad=True),
                     lambda item: item["nodes"]["h"].update(status="zero"),
                     lambda item: item["parameter_groups"]["euclidean9"].update(
                         status="zero", numel_with_gradient=65536, none_count=0)]
        for name in ("contr", "hier"):
            for mutate in mutations:
                with self.subTest(loss=name, mutation=mutate):
                    row = self.row()
                    mutate(row["gradients"][name])
                    self.assertFalse(launcher._regularizers_disabled([row]))

    def test_missing_disabled_loss_evidence_is_refused(self):
        self.assertFalse(launcher._regularizers_disabled([]))
        row = self.row()
        del row["gradients"]["hier"]
        self.assertFalse(launcher._regularizers_disabled([row]))

    def test_ce_must_connect_euclidean9_and_both_mobius_maps(self):
        row = {"gradients": {"ce": {"loss_requires_grad": True, "parameter_groups": {
            name: {"status": "nonzero", "finite": True, "numel_with_gradient": 10}
            for name in ("euclidean9", "m2", "m3")}}}}
        self.assertTrue(launcher._ce_reaches_euclidean9_and_head([row]))
        del row["gradients"]["ce"]["parameter_groups"]["m2"]
        self.assertFalse(launcher._ce_reaches_euclidean9_and_head([row]))


if __name__ == "__main__":
    unittest.main()
