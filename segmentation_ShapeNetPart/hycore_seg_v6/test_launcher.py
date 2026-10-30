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
        "pseg6_staged_launcher", Path(__file__).with_name("launch_experiment.py"))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)


class PredecessorGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hycore_fgh_gate_")
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

    def test_queued_predecessor_without_main_can_be_waited_for(self):
        (self.old / "main").rmdir()
        self.write("dispatch.json", {"arm": "f", "gpu": 0,
                                    "scope": "pseg3_fgh_v6_fresh_smoke_then_fresh200"})
        self.write("queue.json", {"state": "waiting_for_idle_gpu"})
        evidence = launcher.predecessor_state(self.old)
        self.assertFalse(evidence["complete"])
        self.assertEqual(evidence["queue_state"], "waiting_for_idle_gpu")
        self.write("queue.json", {"state": "failed"})
        with self.assertRaisesRegex(RuntimeError, "Predecessor failed"):
            launcher.predecessor_state(self.old)

    def test_missing_main_without_valid_dispatch_is_rejected(self):
        (self.old / "main").rmdir()
        with self.assertRaisesRegex(ValueError, "run parent"):
            launcher.predecessor_state(self.old)
        self.write("dispatch.json", {"arm": "f", "gpu": 2,
                                    "scope": "pseg3_fgh_v6_fresh_smoke_then_fresh200"})
        with self.assertRaisesRegex(ValueError, "run parent"):
            launcher.predecessor_state(self.old)

    def test_busy_gpu_waits_and_does_not_modify_other_runs(self):
        new = self.root / "new"
        new.mkdir()
        with mock.patch.object(launcher, "ensure_idle", side_effect=[RuntimeError("occupied"), {"idle": True}]) as idle, \
                mock.patch.object(launcher.time, "sleep") as sleep:
            self.assertEqual(launcher.wait_for_idle(0, new), {"idle": True})
        self.assertEqual(idle.call_count, 2)
        sleep.assert_called_once_with(30)
        self.assertEqual(json.loads((new / "queue.json").read_text())["state"], "idle_gpu_acquired")

    def test_gpu2_is_refused_before_run_creation(self):
        target = self.root / "unauthorized_gpu"
        with self.assertRaisesRegex(ValueError, "GPU0/1"):
            launcher.main(["--arm", "f", "--gpu", "2", "--root", str(target),
                           "--smoke-data", "smoke", "--production-data", "production"])
        self.assertFalse(target.exists())

    def check_launch(self, arm, gpu):
        target = self.root / ("new_run_" + arm)
        config = self.root / ("config_" + arm + ".json")
        alpha, beta = launcher.approved_coefficients(arm)
        config.write_text(json.dumps({"arm": arm, "alpha": alpha, "beta": beta}), encoding="utf-8")
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
        self.assertIn("segmentation_ShapeNetPart.hycore_seg_v6.train", smoke)
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
        self.assertIs(dispatch["regularizers_active"], True)
        self.assertIs(dispatch["regularizer_diagnostics_only"], False)

    def test_f_waits_then_fresh_smoke_then_fresh_production(self):
        self.check_launch("f", 0)

    def test_g_waits_then_fresh_smoke_then_fresh_production(self):
        self.check_launch("g", 1)

    def test_h_waits_then_fresh_smoke_then_fresh_production(self):
        self.check_launch("h", 0)

    def test_wrong_arm_or_mixed_coefficients_refused_before_queue_creation(self):
        for arm, pair in (("f", (0., 0.)), ("g", (0., 0.)),
                          ("h", (.01, 0.)), ("g", (0., .01))):
            with self.subTest(arm=arm, pair=pair):
                config = self.root / "wrong.json"
                config.write_text(json.dumps({"arm": arm, **dict(zip(("alpha", "beta"), pair))}), encoding="utf-8")
                target = self.root / "refused_run"
                with mock.patch.object(launcher, "wait_for_idle") as idle, \
                        mock.patch.object(launcher.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "approved F/G/H"):
                        launcher.main(["--arm", arm, "--gpu", "0", "--config", str(config),
                                       "--root", str(target), "--smoke-data", "smoke",
                                       "--production-data", "production"])
                self.assertFalse(target.exists())
                idle.assert_not_called()
                run.assert_not_called()

    def test_existing_result_directory_is_not_modified(self):
        target = self.root / "existing"
        target.mkdir()
        sentinel = target / "result.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        config = self.root / "valid.json"
        config.write_text(json.dumps({"arm": "f", "alpha": .01, "beta": .01}), encoding="utf-8")
        with mock.patch.object(launcher, "wait_for_idle") as idle:
            with self.assertRaises(FileExistsError):
                launcher.main(["--arm", "f", "--gpu", "0", "--config", str(config),
                               "--root", str(target), "--smoke-data", "smoke", "--production-data", "production"])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(list(target.iterdir()), [sentinel])
        idle.assert_not_called()

    def test_configured_arm_must_match_selected_arm(self):
        target = self.root / "wrong_arm"
        config = self.root / "wrong_arm.json"
        config.write_text(json.dumps({"arm": "g", "alpha": .01, "beta": .01}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "approved F/G/H"):
            launcher.main(["--arm", "f", "--gpu", "0", "--config", str(config),
                           "--root", str(target), "--smoke-data", "smoke", "--production-data", "production"])
        self.assertFalse(target.exists())


class RoutingGateTests(unittest.TestCase):
    @staticmethod
    def connected():
        return {"status": "zero", "finite": True, "numel_with_gradient": 10,
                "none_count": 0, "parameter_tensors": 3}

    @staticmethod
    def unused():
        return {"status": "none", "numel_with_gradient": 0,
                "none_count": 3, "parameter_tensors": 3}

    def row(self, arm):
        return {"gradients": {name: {
            "loss": .01, "loss_requires_grad": True,
            "nodes": {"pre_conv9": self.connected(), "regularizer_source": self.connected(),
                      "post_conv9": self.connected() if arm == "h" else self.unused(),
                      "h": self.connected() if arm == "h" else self.unused()},
            "parameter_groups": {"shared": self.connected(), "head": self.unused(), "m2": self.unused(), "m3": self.unused(),
                                 "euclidean9": self.connected() if arm == "h" else self.unused()}}
                             for name in ("contr", "hier")}}

    def test_pre_and_post_conv9_routes_accept_connected_zero(self):
        for arm in ("f", "g", "h"):
            with self.subTest(arm=arm):
                self.assertTrue(launcher._regularizer_routing([self.row(arm)], arm))
                self.assertTrue(launcher._head_gradients_unused([self.row(arm)]))

    def test_pre9_regularizer_must_not_leak_into_conv9(self):
        for arm in ("f", "g"):
            for name in ("contr", "hier"):
                row = self.row(arm)
                row["gradients"][name]["parameter_groups"]["euclidean9"] = self.connected()
                self.assertFalse(launcher._regularizer_routing([row], arm))

    def test_pre9_regularizer_must_not_reach_post9_nodes(self):
        for arm in ("f", "g"):
            for node in ("post_conv9", "h"):
                row = self.row(arm)
                row["gradients"]["hier"]["nodes"][node] = self.connected()
                self.assertFalse(launcher._regularizer_routing([row], arm))

    def test_post9_regularizer_must_connect_conv9(self):
        for name in ("contr", "hier"):
            row = self.row("h")
            row["gradients"][name]["parameter_groups"]["euclidean9"] = self.unused()
            self.assertFalse(launcher._regularizer_routing([row], "h"))

    def test_loss_detach_or_missing_pre9_is_rejected(self):
        for arm in ("f", "g", "h"):
            row = self.row(arm)
            row["gradients"]["contr"]["loss_requires_grad"] = False
            self.assertFalse(launcher._regularizer_routing([row], arm))
            row = self.row(arm)
            del row["gradients"]["contr"]["nodes"]["pre_conv9"]
            self.assertFalse(launcher._regularizer_routing([row], arm))
        self.assertFalse(launcher._regularizer_routing([], "f"))

    def test_regularizer_must_bypass_each_mobius_map(self):
        for arm in ("f", "g", "h"):
            for name in ("m2", "m3"):
                row = self.row(arm)
                row["gradients"]["hier"]["parameter_groups"][name] = self.connected()
                self.assertFalse(launcher._head_gradients_unused([row]))

    def test_regularizer_detached_from_shared_backbone_is_rejected(self):
        for arm in ("f", "g", "h"):
            row = self.row(arm)
            row["gradients"]["hier"]["parameter_groups"]["shared"] = self.unused()
            self.assertFalse(launcher._regularizer_routing([row], arm))

    def test_ce_must_connect_euclidean9_and_both_mobius_maps(self):
        row = {"gradients": {"ce": {"loss_requires_grad": True,
                    "nodes": {name: self.connected() for name in
                              ("pre_conv9", "post_conv9", "h", "regularizer_source")},
                    "parameter_groups": {name: self.connected()
                                         for name in ("euclidean9", "m2", "m3")}}}}
        self.assertTrue(launcher._ce_reaches_euclidean9_and_head([row]))
        del row["gradients"]["ce"]["parameter_groups"]["m2"]
        self.assertFalse(launcher._ce_reaches_euclidean9_and_head([row]))


if __name__ == "__main__":
    unittest.main()
