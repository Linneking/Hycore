"""Comparison contracts: selected models, real coordinates and pool identity."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.hier_postrun_audit.comparison import build_comparison


def run(key, rows=None, best=None, final=None, seed=22, budget=200):
    return {"run_id": "H20", "audit_run_key": key, "display_name": key,
            "identity": {"seed": seed, "split_sha256": "a" * 64,
                         "best": best or {"epoch": 2, "val_oa": 92.},
                         "final_test": final or {"test_oa": 90., "test_aa": 88., "test_count": 2468},
                         "config": {"global_batch": 64, "steps_per_epoch": budget}},
            "metadata": {"steps_per_epoch": budget, "curvature": 1},
            "epochs": rows or [{"epoch": 1, "model_updates": budget, "metrics": {"val_oa_pct": 90.}},
                               {"epoch": 2, "model_updates": 2 * budget, "metrics": {"val_oa_pct": 92.}}]}


def snap(key, epoch=2, pool="p" * 64, inputs="i" * 64, labelsha="l" * 64):
    return {"run_key": key, "epoch": epoch, "source": {"sample_pool_sha256": pool, "input_sha256": inputs,
             "labels_sha256": labelsha, "inputmode": "clean/eval", "inference_condition": {"batch_size": 32}},
             "whole": {"c": 1, "dimension": 2, "depth_quantiles": {"mean": 3., "median": 3.}},
             "proxy": {"available": True, "depth_quantiles": {"median": 2.}},
             "retrieval": {"hyperbolic": {"available": True, "coverage": {"unique_samples": 6, "unique_fraction": .5}}}}


class ComparisonContracts(unittest.TestCase):
    def compare(self, left=None, right=None, snapshots=None, registry=None):
        return build_comparison([left or run("V5"), right or run("V6")], snapshots,
                                make_plots=False, registry=registry or {})

    def test_test_metrics_do_not_reselect_a_best_model(self):
        left = run("V5", [{"epoch": 1, "model_updates": 200, "metrics": {"val_oa_pct": 90., "test_oa_pct": 99.}},
                          {"epoch": 2, "model_updates": 400, "metrics": {"val_oa_pct": 92., "test_oa_pct": 80.}}],
                   best={"epoch": 2, "val_oa": 92.}, final={"test_oa": 91., "test_count": 2468})
        selection = self.compare(left)["runs"][0]["selection"]
        self.assertEqual(selection["selected_epoch"], 2)
        self.assertEqual(selection["saved_final_test_oa_pct"], 91.)
        self.assertTrue(selection["validation_selected"])

    def test_no_validation_best_is_invented_from_best_epoch_logs(self):
        left = run("Unknown")
        left["identity"]["best"] = {}
        left["identity"]["final_test"] = {}
        selection = self.compare(left)["runs"][0]["selection"]
        self.assertFalse(selection["validation_selected"])
        self.assertIsNone(selection["selected_epoch"])

    def test_epoch_and_update_alignment_are_separate_without_interpolation(self):
        a, b = run("V5", budget=100), run("V6", budget=200)
        pair = self.compare(a, b)["pairs"][0]
        epoch = pair["epoch_alignment"]["metrics"]["val_oa_pct"]
        update = pair["model_update_alignment"]["metrics"]["val_oa_pct"]
        self.assertEqual(epoch["paired_observation_count"], 2)
        self.assertEqual(update["paired_observation_count"], 1)
        self.assertEqual(update["last"]["left_epoch"], 2)
        self.assertEqual(update["last"]["right_epoch"], 1)
        self.assertEqual(update["last"]["delta"], -2.)

    def test_fixed_pool_requires_actual_input_sha_and_no_proxy_id_matching(self):
        good = self.compare(snapshots={"snapshots": [snap("V5"), snap("V6")]})["pairs"][0]["paired_fixed_pool_snapshots"][0]
        self.assertTrue(good["paired_pool_available"])
        self.assertFalse(good["proxy_identity_paired"])
        bad = self.compare(snapshots={"snapshots": [snap("V5"), snap("V6", inputs="different")]})["pairs"][0]["paired_fixed_pool_snapshots"][0]
        self.assertFalse(bad["paired_pool_available"])
        self.assertNotIn("metrics", bad)

    def test_actual_label_arrays_override_falsely_declared_pool_equivalence(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = snap("V5"), snap("V6")
            a["arrays_file"], b["arrays_file"] = "a.npz", "b.npz"
            arrays = {"sample_ids": [1, 2], "labels": [0, 1], "whole_depth": [2., 3.],
                      "whole_normalized_radius": [.7, .8]}
            np.savez(Path(root) / "a.npz", **arrays)
            arrays["labels"] = [1, 0]
            np.savez(Path(root) / "b.npz", **arrays)
            result = self.compare(snapshots={"snapshots": [a, b], "arrays_base_dir": root})
            paired = result["pairs"][0]["paired_fixed_pool_snapshots"][0]
            self.assertFalse(paired["paired_pool_available"])
            self.assertIn("labels", paired["reason"])

    def test_actual_same_objects_produce_radial_deltas(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = snap("V5"), snap("V6")
            a["arrays_file"], b["arrays_file"] = "a.npz", "b.npz"
            np.savez(Path(root) / "a.npz", sample_ids=[1, 2], labels=[0, 1], whole_depth=[2., 3.], whole_normalized_radius=[.7, .996])
            np.savez(Path(root) / "b.npz", sample_ids=[1, 2], labels=[0, 1], whole_depth=[3., 2.], whole_normalized_radius=[.8, .7])
            pair = self.compare(snapshots={"snapshots": [a, b], "arrays_base_dir": root})["pairs"][0]["paired_fixed_pool_snapshots"][0]
            self.assertTrue(pair["paired_pool_available"])
            radial = pair["paired_object_radial_changes"]
            self.assertEqual(radial["mean_depth_delta"], 0)
            self.assertEqual(radial["outward_depth_fraction"], .5)
            self.assertEqual(radial["left_fixed_q_ge_0_99599_fraction"], .5)

    def test_software_protocol_and_objective_changes_are_distinct(self):
        registry = {"runs": [{"display_name": "V5", "hier_config": {"self_k_excluded": False, "sample_k": 10},
                              "budget": {"total_epochs": 200}},
                             {"display_name": "V6", "hier_config": {"self_k_excluded": True, "sample_k": 20},
                              "budget": {"total_epochs": 300}}]}
        changes = self.compare(registry=registry)["pairs"][0]["changes"]
        families = {r["field"]: r["effect_family"] for r in changes}
        self.assertEqual(families["hier_config.self_k_excluded"], "correctness")
        self.assertEqual(families["hier_config.sample_k"], "hierarchy_objective")
        self.assertEqual(families["budget.total_epochs"], "training_protocol")

    def test_unqualified_duplicate_runs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(run("H20"), run("H20"))

    def test_output_guard_preserves_prior_audit(self):
        with tempfile.TemporaryDirectory() as root:
            build_comparison([run("V5"), run("V6")], output_dir=root, make_plots=False, registry={})
            file = Path(root) / "comparison_summary.json"
            before = file.read_bytes()
            with self.assertRaises(FileExistsError):
                build_comparison([run("V5"), run("V6")], output_dir=root, make_plots=False, registry={})
            self.assertEqual(before, file.read_bytes())


    def test_structure_delta_keeps_original_ci_without_inventing_significance(self):
        def shape(key, mean):
            return {"run_key": key, "epoch": 2, "sample_pool_sha256": "pool", "input_sha256": "inputs", "c": 1,
                    "shape": {"native": {"whole_relations": {"hyperbolic": {
                    "within_class_spearman": {"mean": mean, "class_bootstrap_95ci": [mean-.1, mean+.1]}}}}}}
        structure = {"audit_ids_sha256": "audited-ids", "snapshots": [shape("V5", .2), shape("V6", .3)]}
        result = build_comparison([run("V5"), run("V6")], structure=structure, make_plots=False, registry={})
        evidence = result["pairs"][0]["paired_independent_structure"][0]
        value = evidence["metrics"]["whole.native.hyperbolic.within_class_spearman"]
        self.assertTrue(evidence["available"])
        self.assertAlmostEqual(value["mean_delta"], .1)
        self.assertIsNone(value["delta_confidence_interval"])
        self.assertEqual(value["left"]["class_bootstrap_95ci"], [.1, .30000000000000004])

    def test_mechanism_query_differences_refuse_paired_ancestor_claim(self):
        def mechanism(key, query):
            return {"run_key": key, "epoch": 2, "query_plan_sha256": query, "production_operator": "exact",
                    "margin_distance_units": .1, "gumbel_distance_scale_tau": .1,
                    "identity": {"sample_pool_sha256": "pool", "input_sha256": "inputs"},
                    "components": {"sample": {"noise_summary": {"scalar_variation": {
                    "loss": {"mean": 1.}}}}}}
        summary = {"snapshots": [mechanism("V5", "q1"), mechanism("V6", "q2")]}
        result = build_comparison([run("V5"), run("V6")], mechanisms=summary, make_plots=False, registry={})
        evidence = result["pairs"][0]["controlled_mechanism_alignment"][0]
        self.assertFalse(evidence["paired_query_available"])
        self.assertNotIn("metrics", evidence)

    def test_external_provenance_does_not_expose_private_locations(self):
        summary = {"source_identity": {"artifact": "/mnt/private_account/cache.npz",
                                      "host": "192.168.0.1",
                                      "description": "from 192.168.0.1 GPU-12345678-1234-1234-1234-123456789abc"}}
        result = build_comparison([run("V5"), run("V6")], mechanisms=summary, make_plots=False, registry={})
        text = str(result["external_evidence"])
        self.assertNotIn("192.168.0.1", text)
        self.assertNotIn("private_account", text)
        self.assertNotIn("12345678-1234", text)

    def test_comparison_png_svg_are_real_files(self):
        a, b = run("V5"), run("V6")
        for row in a["epochs"]+b["epochs"]:
            row["metrics"]["whole_near_boundary_fraction"] = .9
            row["metrics"]["proxy_numerical_saturation_fraction"] = .5
        with tempfile.TemporaryDirectory() as root:
            result = build_comparison([a, b], output_dir=root, registry={})
            self.assertEqual(len(result["figures"]), 3)
            for figure in result["figures"]:
                self.assertGreater((Path(root) / figure["png"]).stat().st_size, 15000)
                self.assertTrue((Path(root) / figure["svg"]).is_file())


if __name__ == "__main__":
    unittest.main()
