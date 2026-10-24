"""Synthetic fixtures test the gate validator; none are execution evidence."""
import copy
import unittest

from inter_hierarchy_MN40.parthier_v1.gate import validate_full_gate


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def gate_fixture():
    def completed(mode, epochs, per_epoch):
        return {"status": "completed", "commit": COMMIT,
                "config": {"mode": mode, "activation_checkpoint": True,
                           "save_on_cpu": False, "backbone_passes_per_step": 4,
                           "queue": {"capacity": 2048}, "epochs": epochs,
                           "steps_per_epoch": per_epoch},
                "completed_epochs": epochs, "updates": epochs * per_epoch,
                "checks": {"bn_updates_four": True, "finite": True,
                           "replica_consistent": True, "queue_consistent": True}}
    diagnostic = completed("diagnostic", 1, 14)
    diagnostic["steps"] = [
        {"update": update, "queue_before": min((update - 1) * 192, 2048),
         "whole_count": 1024 if update == 12 else 900, "finite": True}
        for update in range(1, 15)]
    return {"source_commit": COMMIT, "identity": {"commit": COMMIT},
            "cuda_checkpoint_contract": {"status": "passed", "commit": COMMIT, "device": "cuda"},
            "smoke": completed("smoke", 2, 2), "diagnostic": diagnostic}


class DispatchGateTests(unittest.TestCase):
    def test_complete_gate_passes_without_mutating_evidence(self):
        gate = gate_fixture()
        before = copy.deepcopy(gate)
        self.assertIsNone(validate_full_gate(gate, COMMIT))
        self.assertEqual(gate, before)

    def test_all_artifacts_must_match_the_expected_source(self):
        for location in ("source_commit", "identity", "cuda_checkpoint_contract", "smoke", "diagnostic"):
            with self.subTest(location=location):
                gate = gate_fixture()
                if location == "source_commit":
                    gate[location] = "different"
                else:
                    gate[location]["commit"] = "different"
                with self.assertRaises(ValueError):
                    validate_full_gate(gate, COMMIT)

    def test_cpu_or_incomplete_checkpoint_contract_is_rejected(self):
        for field, value in (("device", "cpu"), ("status", "running")):
            with self.subTest(field=field):
                gate = gate_fixture()
                gate["cuda_checkpoint_contract"][field] = value
                with self.assertRaises(ValueError):
                    validate_full_gate(gate, COMMIT)

    def test_run_config_and_acceptance_checks_are_strict(self):
        for run_name in ("smoke", "diagnostic"):
            for field in ("bn_updates_four", "finite", "replica_consistent", "queue_consistent"):
                for unverified in (False, None, 1):
                    with self.subTest(run=run_name, field=field, value=unverified):
                        gate = gate_fixture()
                        gate[run_name]["checks"][field] = unverified
                        with self.assertRaises(ValueError):
                            validate_full_gate(gate, COMMIT)
            for key, value in (("activation_checkpoint", False), ("activation_checkpoint", 1),
                               ("save_on_cpu", True), ("backbone_passes_per_step", True)):
                with self.subTest(run=run_name, config=key):
                    gate = gate_fixture()
                    gate[run_name]["config"][key] = value
                    with self.assertRaises(ValueError):
                        validate_full_gate(gate, COMMIT)
            gate = gate_fixture()
            gate[run_name]["config"]["queue"]["capacity"] = 0
            with self.assertRaises(ValueError):
                validate_full_gate(gate, COMMIT)

    def test_smoke_must_really_complete_two_by_two_updates(self):
        for changed in ("budget", "status", "completed_epochs", "updates"):
            with self.subTest(changed=changed):
                gate = gate_fixture()
                smoke = gate["smoke"]
                if changed == "budget":
                    smoke["config"]["epochs"], smoke["config"]["steps_per_epoch"] = 1, 4
                    smoke["completed_epochs"] = 1
                elif changed == "status":
                    smoke["status"] = "running"
                else:
                    smoke[changed] -= 1
                with self.assertRaises(ValueError):
                    validate_full_gate(gate, COMMIT)

    def test_partial_missing_or_reordered_diagnostic_steps_are_rejected(self):
        for changed in ("only13", "missing", "duplicate", "nonfinite", "bad_count"):
            with self.subTest(changed=changed):
                gate = gate_fixture()
                diagnostic = gate["diagnostic"]
                if changed == "only13":
                    diagnostic["updates"] = diagnostic["config"]["steps_per_epoch"] = 13
                    diagnostic["steps"].pop()
                elif changed == "missing":
                    diagnostic["steps"].pop(3)
                elif changed == "duplicate":
                    diagnostic["steps"][3]["update"] = 3
                elif changed == "nonfinite":
                    diagnostic["steps"][1]["finite"] = False
                else:
                    diagnostic["steps"][1]["whole_count"] = 600
                with self.assertRaises(ValueError):
                    validate_full_gate(gate, COMMIT)

    def test_three_actual_full_queue_steps_and_full_shape_are_required(self):
        gate = gate_fixture()
        gate["diagnostic"]["steps"][-2]["queue_before"] = 1920
        with self.assertRaises(ValueError):
            validate_full_gate(gate, COMMIT)
        gate = gate_fixture()
        # A whole1024 step while the queue is still filling is insufficient.
        gate["diagnostic"]["steps"][0]["whole_count"] = 1024
        gate["diagnostic"]["steps"][11]["whole_count"] = 900
        with self.assertRaises(ValueError):
            validate_full_gate(gate, COMMIT)


    def test_checkpoint_policy_evidence_must_match(self):
        gate = gate_fixture()
        gate["diagnostic"]["config"]["checkpoint_policy"] = "small_medium"
        with self.assertRaises(ValueError):
            validate_full_gate(gate, COMMIT)
        gate["smoke"]["config"]["checkpoint_policy"] = "small_medium"
        with self.assertRaises(ValueError):
            validate_full_gate(gate, COMMIT)


if __name__ == "__main__":
    unittest.main()
