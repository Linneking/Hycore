"""Meaningful adapter contracts: units, missing history, usage, and privacy."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.hier_postrun_audit.adapters import normalize_run, summarize_steps
from tools.hier_postrun_audit.inventory import inventory_run, resolve_run


class AdapterContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def create_run(self, name="v7_hier", config=None, extra=None, rows=None):
        directory = self.root / name
        directory.mkdir()
        manifest = {"status": "completed", "commit": "a" * 40,
                    "training_config": config or {"seed": 22, "c": 1, "P": 3,
                    "steps_per_epoch": 2, "warmup_epochs": 0, "proxy_optimizer": {"name": "AdamW"}}}
        manifest.update(extra or {})
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in (rows or [])), encoding="utf-8")
        return directory

    def row(self, epoch=1, active=2):
        return {"epoch": epoch, "phase": "joint", "lambda_hier": .1,
                "lr_used": {"model": .1, "proxy": .01},
                "validation": {"val_oa": 92., "val_aa": 90., "val_count": 984},
                "clean_train": {"train_eval_oa": 96., "train_eval_aa": 95.},
                "telemetry": {"batches": 2, "training": {"oa": .875, "aa_observed_classes": .8},
                    "scalars": {"loss": {"mean": 3.}, "hier_monitored_loss": {"mean": .5},
                                "proxy_cap_post_max_tangent_norm": {"mean": 2.9, "max": 3.0000007}},
                    "geometry": {"whole": {"depth": {"mean": 4., "median": 5.},
                                                       "radius": {"mean": .8, "median": .9}}},
                    "hierarchy": {"objective_active_batches": active}},
                "structure": {"sample": {"selected_ancestor_usage": {
                    "all_draws/pair": {"counts": [2, 0, 0]},
                    "all_draws/triple": {"counts": [0, 1, 1]},
                    "noncollision/pair": {"counts": [1, 0, 0]},
                    "noncollision/triple": {"counts": [0, 0, 1]}}}}}

    def test_fraction_conversion_and_clean_metric_separation(self):
        analysis = normalize_run(self.create_run(rows=[self.row()]))
        metric = analysis["epochs"][0]["metrics"]
        self.assertEqual(metric["train_aug_oa_pct"], 87.5)
        self.assertEqual(metric["clean_train_oa_pct"], 96.)
        self.assertEqual(metric["val_oa_pct"], 92.)
        self.assertAlmostEqual(metric["weighted_hier_loss"], .05)
        self.assertEqual(metric["whole_logmap_norm_mean"], 2.)
        self.assertEqual(metric["whole_q_mean"], .8)
        self.assertIsNone(metric["proxy_depth_mean"])
        self.assertEqual(metric["proxy_parameter_max_tangent_norm"], 3.0000007)

    def test_counts_domains_and_entropy_are_not_scalar_activation_shortcuts(self):
        analysis = normalize_run(self.create_run(rows=[self.row()]))
        rows = [row for row in analysis["proxy_usage"] if row["domain"] == "all_draws" and row["role"] == "combined"]
        self.assertEqual(rows[0]["counts"], [2, 1, 1])
        self.assertEqual(rows[0]["used_proxy_count"], 3)
        self.assertAlmostEqual(rows[0]["effective_proxy_count"], math.exp(-.5 * math.log(.5) - .5 * math.log(.25)))
        noncollision = [row for row in analysis["proxy_usage"] if row["domain"] == "noncollision" and row["role"] == "combined"]
        self.assertEqual(noncollision[0]["used_proxy_count"], 2)
        self.assertIsNone(analysis["epochs"][0]["metrics"]["sample_collision_fraction"])

    def test_continuation_keeps_original_epoch_axis_without_fabricating_prefix(self):
        config = {"c": 1, "seed": 22, "steps_per_epoch": 200, "proxy_optimizer": None,
                  "shared_historical_prefix_epochs": 20, "warmup_epochs": 20}
        row = {"epoch": 21, "phase": "V6_balanced_B0", "lambda_hier": 0,
               "actual_optimizer_steps": 200, "telemetry": {"batches": 200}}
        analysis = normalize_run(self.create_run("v6_b64", config=config, rows=[row]))
        self.assertEqual(len(analysis["epochs"]), 1)
        record = analysis["epochs"][0]
        self.assertEqual(record["model_updates"], 4200)
        self.assertEqual(record["logged_model_updates"], 200)
        self.assertEqual(record["logged_proxy_updates"], 0)
        self.assertEqual(analysis["metadata"]["inherited_prefix_epochs"], 20)
        self.assertIsNone(record["metrics"]["hier_loss"])

    def test_warmup_monitoring_does_not_count_as_proxy_updates(self):
        first, second = self.row(1, active=0), self.row(2, active=2)
        first["lambda_hier"] = 0
        analysis = normalize_run(self.create_run("v6_hier", rows=[first, second]))
        self.assertEqual([row["logged_proxy_updates"] for row in analysis["epochs"]], [0, 2])
        self.assertEqual(analysis["epochs"][0]["metrics"]["hier_loss"], .5)
        self.assertEqual(analysis["epochs"][0]["metrics"]["weighted_hier_loss"], 0)

    def test_legacy_original_baseline_uses_percent_and_absent_geometry_stays_missing(self):
        directory = self.create_run("v6_orig_b32", config={"seed": 22, "steps_per_epoch": 307},
            extra={"protocol": "v6_original_hycore_b32_source_defaults_engineering_baseline"}, rows=[])
        (directory / "metrics.jsonl").unlink()
        row = {"epoch": 1, "protocol": "v6_original_hycore", "training": {"acc": 81., "acc_avg": 77., "loss": 3.},
               "official_test": {"acc": 80., "acc_avg": 75.}, "train_steps": 307,
               "gradient": {"clip_trigger_fraction": 1., "preclip_norm_mean": 5.}}
        (directory / "metrics_epoch_001.json").write_text(json.dumps(row), encoding="utf-8")
        analysis = normalize_run(directory)
        metric = analysis["epochs"][0]["metrics"]
        self.assertEqual(metric["train_aug_oa_pct"], 81.)
        self.assertEqual(metric["test_oa_pct"], 80.)
        self.assertIsNone(metric["whole_depth_mean"])
        self.assertIsNone(metric["val_oa_pct"])
        self.assertEqual(metric["model_clip_fraction"], 1.)

    def test_v5_budget_baseline_training_nested_telemetry(self):
        row = self.row()
        row["training"] = row.pop("telemetry")
        row["train_clean"] = {"oa": 93., "aa": 91., "ce": 3., "examples": 8856}
        row.pop("clean_train")
        row["validation"] = {"oa": 91., "aa": 89., "ce": 3., "examples": 984}
        row["protocol"] = "v5_hycore_b32_random_permutation_budget_cap"
        row.pop("phase")
        row["training"]["hierarchy"] = None
        row.pop("structure")
        analysis = normalize_run(self.create_run("v5_b32", config={"steps_per_epoch": 2},
            extra={"fixed": {"c": 1, "hierarchy": None}}, rows=[row]))
        record = analysis["epochs"][0]
        self.assertEqual(record["metrics"]["train_aug_oa_pct"], 87.5)
        self.assertEqual(record["metrics"]["clean_train_oa_pct"], 93.)
        self.assertEqual(record["metrics"]["val_count"], 984)
        self.assertEqual(record["logged_proxy_updates"], 0)
        self.assertEqual(record["metrics"]["whole_depth_mean"], 4.)
        self.assertIn("training/", record["availability"]["whole_depth_mean"])

    def test_no_implicit_step_scan_and_private_values_removed(self):
        directory = self.create_run(rows=[self.row()], extra={
            "physical_gpu_uuids": "private-gpu-uuid", "command_argv": ["private-shell"],
            "initialization": {"source_checkpoint": "/private/user/checkpoint.pth", "teacher": None},
            "resume_identity": {"path": "D:\\secret\\last.pth", "sha256": "b" * 64}})
        (directory / "steps.jsonl").write_text("malformed deliberately\n", encoding="utf-8")
        analysis = normalize_run(directory)
        public = json.dumps(analysis)
        self.assertNotIn("private-gpu-uuid", public)
        self.assertNotIn("private-shell", public)
        self.assertNotIn("/private/user", public)
        self.assertNotIn("secret", public)
        self.assertEqual(analysis["identity"]["resume_identity"]["path"], "last.pth")
        self.assertFalse(analysis["inventory"]["capabilities"]["step_log_scanned"])
        self.assertEqual(summarize_steps(directory)["rows"], 0)

    def test_unique_arm_resolution_and_duplicate_epoch_warning(self):
        directory = self.create_run(rows=[self.row(1), self.row(1), self.row(3)])
        self.assertEqual(resolve_run(self.root), directory.resolve())
        inventory = inventory_run(self.root)
        self.assertEqual(inventory["duplicate_epochs"], [1])
        self.assertEqual(inventory["missing_epochs_within_logged_span"], [2])
        analysis = normalize_run(self.root)
        self.assertEqual([row["epoch"] for row in analysis["epochs"]], [1, 3])
        self.create_run("second_arm", rows=[self.row()])
        with self.assertRaises(ValueError):
            resolve_run(self.root)

    def test_names_resolve_only_inside_confirmed_registry_scope(self):
        config = {"seed": 22, "c": 1, "P": 512, "steps_per_epoch": 200,
                  "global_batch": 64, "warmup_epochs": 20, "sample_K": 20,
                  "proxy_optimizer": {"name": "AdamW"}}
        parent = self.root / "hier_proxy_v6_20261004_0134"
        parent.mkdir()
        run = parent / "H20"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"training_config": config}), encoding="utf-8")
        (run / "metrics.jsonl").write_text(json.dumps(self.row()), encoding="utf-8")
        result = normalize_run(run)
        self.assertEqual(result["display_name"], "V6-HIER64-K20-W20")
        self.assertIn("experiment_registry.json", result["identity"]["display_name_source"])
        unknown = self.create_run("H20", config=config, rows=[self.row()],
                                  extra={"protocol": "v6_unconfirmed_experiment"})
        self.assertEqual(normalize_run(unknown)["display_name"], "H20")
        self.assertEqual(normalize_run(unknown, name="My explicit arm")["display_name"], "My explicit arm")

    def test_v7_structured_name_and_n_proxies_alias(self):
        config = {"seed": 22, "c": 1, "n_proxies": 512, "steps_per_epoch": 200,
                  "global_batch": 64, "warmup_epochs": 0, "sample_K": 20,
                  "proxy_optimizer": {"name": "AdamW"}}
        run = self.create_run("V7_HIER64_K20_W0", config=config, rows=[self.row()])
        result = normalize_run(run)
        self.assertEqual(result["display_name"], "V7-HIER64-K20-W0")
        self.assertEqual(result["metadata"]["n_proxy"], 512)


if __name__ == "__main__":
    unittest.main()
