"""Public allowlist/source immutability and real-probe identity counterexamples."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.hier_postrun_audit.delivery import build_delivery, joint_input_identity, sanitize, validate_offline_links

MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def probe(path, name="V7", status="completed", seed=22):
    manifest = {"status": status, "run_key": name, "epoch": 200, "source_unchanged": True,
                "source_checkpoint_sha256": "same_checkpoint_" + name, "max_seconds": 600,
                "audit_code_commit": "probe_audit_commit_" + name}
    write(path / "backbone_probe_manifest.json", manifest)
    if status != "completed":
        return
    identity = {"input_sha256": "actual_inputs", "sample_ids": list(range(64)), "labels": [i // 2 for i in range(64)],
                "whole_count": 1024, "child_count": 256, "microbatch_size": 4,
                "whole_centers": [1] * 64, "child_centers": [2] * 64, "crop_seed": seed,
                "mode": "clean fixed crop / eval BN / original alias overwrite",
                "classifier_on_concatenated_mu": True, "activation_checkpoint": False,
                "checkpoint_early_stop": False, "gradient_method": "two_pass_feature_adjoint_vjp"}
    value = {"format": "hier-frozen-backbone-gradient-v1", "run_key": name, "display_name": name,
             "epoch": 200, "checkpoint_sha256": "same_checkpoint_" + name, "source_commit": "commit",
             "gradient_method": "two_pass_feature_adjoint_vjp",
             "model_source_sha256": "same_model_source",
             "parameter_groups": {"shared_encoder": {"parameter_elements": 100,
                 "norms": {"ce": 2., "weighted_intra": 1., "base": 2.5, "proxy": 0., "hier_weighted": .5},
                 "hier_vs_base": {"first_norm": .5, "second_norm": 2.5, "norm_ratio": .2, "cosine": -.4}}},
             "input_identity": identity,
             "input_condition": {"mode": "fixed clean crop evalBN original alias", "microbatch_size": 4,
                                 "seed": seed, "whole_count": 1024, "child_count": 256,
                                 "gradient_method": "two_pass_feature_adjoint_vjp", "activation_checkpoint": False},
             "replay_validation": {"available": True, "passed": True, "same_actual_forward_RNG": True,
                                   "rtol": 1e-5, "atol": 1e-6, "max_absolute_feature_difference": 0.,
                                   "forward_replays": 32, "expected_forward_replays": 32,
                                   "records": [{"role": role, "start": start, "end": start + 4,
                                                "max_absolute_difference": 0.}
                                               for start in range(0, 64, 4) for role in ("child", "whole")]},
             "read_only": {"optimizer_updates": 0, "BN_parameters_grad_buffers_unchanged": True,
                           "RNG_restored": True, "source_clouds_unchanged": True,
                           "new_validation_forwards": 0, "new_test_forwards": 0}}
    write(path / "backbone_probe_summary.json", value)


class DeliveryTests(unittest.TestCase):
    def fixture(self, root):
        system = root / "system"
        system.mkdir()
        write(system / "system_manifest.json", {"status": "completed", "source_files_unchanged": True,
              "audit_code_commit": "audit_commit", "budgets": {"max_seconds": 3600}})
        (system / "report.html").write_text("<html><body><main><h1>Audit</h1><p>Original source</p></main></body></html>", encoding="utf-8")
        write(system / "report_data.json", {"rows": [{"depth": 6.2, "message": "path /mnt/private/server/data host192.168.1.50"}]})
        (system / "epochs.csv").write_text('epoch,note\n200,"C:\\private\\folder\\artifact, remains"\n', encoding="utf-8")
        write(system / "normalized_runs.json", {"private": "raw secret"})
        write(system / "combined_snapshot_spec.json", {"cache": "/mnt/private/raw.npz"})
        (system / "weights.pth").write_bytes(b"weight_secret")
        (system / "raw.log").write_text("log_secret")
        write(system / "structure" / "structure_summary.json", {"nested": ["D:\\private\\cache.npz", {"host": "10.20.30.40"}]})
        (system / "structure" / "shape.npz").write_bytes(b"npz_secret")
        return system

    def test_replay_pass_flag_count_contract_method_and_execution_commit(self):
        for field, value in (("passed", False), ("passed", None), ("expected_forward_replays", 31)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                system = self.fixture(root)
                source = root / "probe"
                probe(source)
                path = source / "backbone_probe_summary.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                record["replay_validation"][field] = value
                write(path, record)
                answer = build_delivery(system, [source], root / "public", make_plots=False)
                self.assertEqual(answer["backbone_success_count"], 0)
                self.assertIn("replay", answer["probe_availability"][0]["reason"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            source = root / "probe"
            probe(source)
            answer = build_delivery(system, [source], root / "public", make_plots=False)
            self.assertEqual(answer["probe_availability"][0]["audit_code_commit"], "probe_audit_commit_V7")
            self.assertEqual(answer["backbone"]["successful_probes"][0]["audit_code_commit"], "probe_audit_commit_V7")
            path = source / "backbone_probe_summary.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["input_condition"]["gradient_method"] = "ordinary_autograd"
            write(path, record)
            answer = build_delivery(system, [source], root / "public_bad", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 0)
            self.assertIn("gradient_method", answer["probe_availability"][0]["reason"])

    def test_allowlist_recursive_redaction_html_insert_once_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one, two = root / "V5_probe", root / "V7_probe"
            probe(one, "V5")
            probe(two, "V7")
            source_before = {path: path.read_bytes() for path in system.rglob("*") if path.is_file()}
            answer = build_delivery(system, [one, two], root / "delivery", make_plots=False)
            out = root / "delivery"
            self.assertEqual(answer["status"], "completed")
            self.assertEqual(answer["backbone_success_count"], 2)
            self.assertTrue(answer["joint_input_comparison"]["available"])
            self.assertTrue(answer["sources_unchanged"])
            self.assertTrue((out / "backbone.json").is_file())
            self.assertFalse((out / "system_manifest.json").exists())
            self.assertFalse((out / "normalized_runs.json").exists())
            self.assertFalse((out / "combined_snapshot_spec.json").exists())
            self.assertFalse((out / "weights.pth").exists())
            self.assertFalse((out / "raw.log").exists())
            self.assertFalse((out / "structure" / "shape.npz").exists())
            for path in out.rglob("*.json"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/mnt/private", text)
                self.assertNotIn("10.20.30.40", text)
                self.assertNotIn("192.168.1.50", text)
                self.assertNotIn("D:\\\\private", text)
                json.loads(text)
            self.assertEqual((out / "report.html").read_text().count('id="backbone-probe-overview"'), 1)
            self.assertEqual(source_before, {path: path.read_bytes() for path in system.rglob("*") if path.is_file()})

    def test_failed_attempt_is_missing_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            failed, done = root / "r1_oom", root / "r3"
            probe(failed, "V5", "failed")
            probe(done, "V7")
            answer = build_delivery(system, [failed, done], root / "delivery", make_plots=False)
            self.assertEqual(answer["status"], "partial")
            self.assertEqual(answer["backbone_success_count"], 1)
            self.assertFalse(answer["probe_availability"][0]["available"])
            self.assertFalse(answer["joint_input_comparison"]["available"])

    def test_changed_crop_seed_disables_joint_claim_even_with_same_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one, two = root / "one", root / "two"
            probe(one, "V5", seed=22)
            probe(two, "V7", seed=23)
            answer = build_delivery(system, [one, two], root / "delivery", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 2)
            self.assertFalse(answer["joint_input_comparison"]["available"])
            self.assertIn("crop_seed", answer["joint_input_comparison"]["reason"])

    def test_false_read_only_flag_rejects_completed_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one = root / "one"
            probe(one)
            path = one / "backbone_probe_summary.json"
            record = json.loads(path.read_text())
            record["read_only"]["BN_parameters_grad_buffers_unchanged"] = False
            write(path, record)
            answer = build_delivery(system, [one], root / "delivery", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 0)
            self.assertIn("read-only", answer["probe_availability"][0]["reason"])

    def test_outputs_cannot_exist_or_be_nested_in_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            with self.assertRaises(ValueError):
                build_delivery(system, [], system / "delivery", make_plots=False)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                build_delivery(system, [], existing, make_plots=False)

    def test_recursive_redaction_preserves_numeric_observations(self):
        values = sanitize({"metric": 6.2126, "nested": ["ssh://secret@10.2.3.4", {"source": "C:\\Users\\secret\\data.npy"}]})
        self.assertEqual(values["metric"], 6.2126)
        self.assertNotIn("10.2.3.4", str(values))
        self.assertNotIn("C:\\Users", str(values))

    def test_delivery_cli_accepts_repeated_backbone_dirs_and_cpu_only_flag(self):
        from tools.hier_postrun_audit.__main__ import arguments
        with patch("sys.argv", ["audit", "delivery", "--system-dir", "system",
                               "--backbone-dir", "one", "--backbone-dir", "two",
                               "--out-dir", "new", "--no-plots"]):
            args = arguments()
        self.assertEqual(args.command, "delivery")
        self.assertEqual(args.backbone_dir, [Path("one"), Path("two")])
        self.assertTrue(args.no_plots)

    def test_relative_links_copy_safe_summaries_and_replace_server_local_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            report = system / "report.html"
            report.write_text('<html><body><main><h1>Audit</h1>'
                '<a href="snapshots/snapshot_summary.json">Snapshots</a>'
                '<a href="feature_exports/export_manifest.json">Export identity</a>'
                '<a href="system_manifest.json">CPU manifest</a>'
                '<a href="snapshots/arrays.npz">Cache</a>'
                '<img src="figures/missing.png"></main></body></html>', encoding="utf-8")
            write(system / "snapshots" / "snapshot_summary.json", {"geometry": 6.2})
            write(system / "feature_exports" / "export_manifest.json",
                  {"status": "completed", "source_commit": "commit",
                   "input_sha256": "inputs", "source_training_config": {"cache": "/mnt/private/secret"},
                   "command": "private command"})
            before = report.read_bytes()
            out = root / "delivery"
            result = build_delivery(system, [], out, make_plots=False)
            self.assertTrue(result["offline_link_integrity"]["available"])
            self.assertGreaterEqual(result["offline_link_integrity"]["relative_file_links_checked"], 5)
            self.assertTrue((out / "snapshots" / "snapshot_summary.json").exists())
            exported = json.loads((out / "feature_exports" / "export_manifest.json").read_text())
            self.assertTrue(exported["public_summary"])
            self.assertNotIn("source_training_config", exported)
            self.assertNotIn("command", exported)
            text = (out / "report.html").read_text()
            self.assertIn('href="delivery_manifest.json"', text)
            self.assertIn('href="#public-delivery-exclusions"', text)
            self.assertIn('src="figures/server_local_artifact_placeholder.svg"', text)
            self.assertEqual(before, report.read_bytes())

    def test_changed_primary_source_during_copy_causes_failed_delivery(self):
        from tools.hier_postrun_audit.delivery import _copy_public
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            def mutate(source, target):
                _copy_public(source, target)
                if source.name == "report_data.json":
                    source.write_text(source.read_text() + " ", encoding="utf-8")
            with patch("tools.hier_postrun_audit.delivery._copy_public", side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    build_delivery(system, [], root / "delivery", make_plots=False)
            failed = json.loads((root / "delivery" / "delivery_manifest.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed["sources_unchanged"])

    def test_failed_replay_is_not_successful_backbone_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one = root / "one"
            probe(one)
            path = one / "backbone_probe_summary.json"
            record = json.loads(path.read_text())
            record["replay_validation"]["available"] = False
            write(path, record)
            answer = build_delivery(system, [one], root / "delivery", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 0)
            self.assertIn("replay", answer["probe_availability"][0]["reason"])

    def test_replay_must_cover_every_object_in_both_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one = root / "one"
            probe(one)
            path = one / "backbone_probe_summary.json"
            record = json.loads(path.read_text())
            record["replay_validation"]["records"][-1]["end"] = 63
            write(path, record)
            answer = build_delivery(system, [one], root / "delivery", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 0)
            self.assertIn("coverage", answer["probe_availability"][0]["reason"])

    def test_gradient_methods_must_match_for_joint_interpretation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one, two = root / "one", root / "two"
            probe(one, "V5")
            probe(two, "V7")
            path = two / "backbone_probe_summary.json"
            record = json.loads(path.read_text())
            record["gradient_method"] = "single_pass_autograd"
            record["input_identity"]["gradient_method"] = "single_pass_autograd"
            record["input_condition"]["gradient_method"] = "single_pass_autograd"
            write(path, record)
            answer = build_delivery(system, [one, two], root / "delivery", make_plots=False)
            self.assertEqual(answer["backbone_success_count"], 2)
            self.assertFalse(answer["joint_input_comparison"]["available"])
            self.assertIn("gradient_method", answer["joint_input_comparison"]["reason"])

    def test_false_activation_checkpoint_needs_no_old_checkpoint_identity_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one, two = root / "one", root / "two"
            probe(one, "V5")
            probe(two, "V7")
            write(system / "comparison_summary.json", {"runs": [
                {"run_key": "V5", "display_name": "V5-HIER64-K20-W20"},
                {"run_key": "V7", "display_name": "V7-HIER64-K20-W0"}]})
            answer = build_delivery(system, [one, two], root / "delivery", make_plots=False)
            self.assertTrue(answer["joint_input_comparison"]["available"])
            self.assertEqual(answer["backbone"]["successful_probes"][0]["display_name"], "V5-HIER64-K20-W20")
            record = answer["backbone"]["successful_probes"][0]
            self.assertEqual(record["gradient_method"], "two_pass_feature_adjoint_vjp")
            self.assertTrue(record["replay_validation"]["available"])
            self.assertNotIn("checkpoint_use_reentrant", record["input_identity"])

    @unittest.skipUnless(MATPLOTLIB, "Server validates generated scientific PNG/SVG")
    def test_actual_shared_encoder_plot_is_exported_and_linked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            system = self.fixture(root)
            one, two = root / "one", root / "two"
            probe(one, "V5")
            probe(two, "V7")
            answer = build_delivery(system, [one, two], root / "delivery", make_plots=True)
            figure = answer["backbone_figure"]
            self.assertEqual(figure["kind"], "real_backbone_gradient")
            for key in ("png", "svg"):
                path = root / "delivery" / figure[key]
                self.assertGreater(path.stat().st_size, 10000)
            self.assertIn(figure["png"], (root / "delivery" / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
