"""Source immutability and actual snapshot contracts for the system runner."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from tools.hier_postrun_audit.system import run_system_audit, _selection
from tools.hier_postrun_audit.snapshots import analyze_snapshots
from tools.hier_postrun_audit.geometry import expmap0, poincare_distance


class SystemContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, name="one", key="H20_v6_verified", with_specs=True, with_usage=True):
        root = self.root / name; root.mkdir()
        export = root / "feature_exports"; export.mkdir()
        input_sha = "a" * 64
        ids = np.arange(6)
        labels = np.array([0, 0, 0, 1, 1, 1])
        whole = np.array([[.2,.1], [.3,.1], [.1,.3], [-.2,.1], [-.3,.1], [-.1,.3]])
        tangent = np.array([[.5,.5],[-.5,.1],[.1,-.5]])
        proxy_ids = [10, 20, 30]
        specs = []
        for epoch in (1, 2):
            cache = export / ("e%d_whole_cache.npz" % epoch)
            np.savez_compressed(cache, sample_ids=ids, labels=labels, mu=whole * (1 + epoch / 20),
                                proxy_tangent=tangent, proxy_ids=proxy_ids,
                                input_sha256=np.asarray(input_sha), c=np.asarray(1.))
            specs.append({"run_key": key, "epoch": epoch, "cache": str(cache), "c": 1.,
                          "version": "v6", "display_name": "V6-HIER64-test",
                          "proxy_mapping": {"numeric_radius_fraction": .999},
                          "input_sha256": input_sha, "input_mode": "clean/eval",
                          "inference_condition": {"batch_size": 32},
                          "proxy_id_policy": "stable saved parameter rows",
                          **({"proxy_activation_counts": [2,0,1], "activation_scope": "epoch training sample noncollision pair+triple; distinct from fixed-panel retrieval"} if with_usage else {})})
        analysis = {"run_id": key, "audit_run_key": key, "storage_run_id": "H20", "display_name": key,
                    "identity": {"version": "V6", "seed": 22, "best": {"epoch": 1, "val_oa": 92.},
                                 "config": {"proxy_optimizer": True}},
                    "metadata": {"curvature": 1.}, "inventory": {"manifest_completed_epochs": 2},
                    "epochs": [{"epoch": e, "model_updates": e*2, "proxy_steps": 2, "metrics": {"lambda_hier": .1, "val_oa_pct": 92.}} for e in (1,2)],
                    "proxy_usage": [{"epoch": e, "component": "sample", "domain": "noncollision", "role": "combined",
                                    "proxy_ids": proxy_ids, "counts": [2,0,1]} for e in (1,2)] if with_usage else []}
        (root / "normalized_runs.json").write_text(json.dumps([analysis]), encoding="utf-8")
        (root / "audit_manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        if with_specs:
            serial = copy.deepcopy(specs)
            for row in serial: row["cache"] = Path(row["cache"]).name
            (export / "snapshot_spec.json").write_text(json.dumps({"snapshots": serial}), encoding="utf-8")
        (export / "export_manifest.json").write_text(json.dumps({"status": "completed", "resolved_checkpoint_aliases": {"best": 1, "last": 2}}), encoding="utf-8")
        analyze_snapshots(specs, root / "snapshots")
        return root, specs, analysis

    def audit(self, roots, out="combined", **kwargs):
        return run_system_audit(roots, self.root / out, do_mechanisms=False, do_structure=False,
                                do_redundancy=False, make_plots=False, render=False, **kwargs)

    def test_reuses_actual_arrays_and_adds_exact_equal_radius_control(self):
        artifact, specs, _ = self.artifact()
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact.rglob("*") if path.is_file()}
        result = self.audit([artifact])
        self.assertEqual(result["manifest"]["status"], "completed")
        self.assertTrue(result["manifest"]["source_files_unchanged"])
        snapshot = result["snapshots"]["snapshots"][0]
        self.assertTrue(snapshot["reuse"]["actual_arrays_loaded"])
        control = snapshot["retrieval"]["equal_radius_hyperbolic"]
        self.assertTrue(control["control"]["counterfactual_only"])
        self.assertFalse(control["control"]["production_changed"])
        with np.load(specs[0]["cache"], allow_pickle=False) as cache:
            whole = cache["mu"]; proxy = expmap0(cache["proxy_tangent"], 1, numeric_radius_fraction=.999)
            first = .5 * proxy / np.linalg.norm(proxy, axis=1)[:,None]
            second = .5 * whole / np.linalg.norm(whole, axis=1)[:,None]
            distances = poincare_distance(first, second, 1)
            ids = np.array(control["topk_sample_ids"], dtype=int)
            selected = np.take_along_axis(distances, ids, axis=1)
        self.assertAlmostEqual(control["nearest_distance_statistics"]["mean"], selected.mean(), places=12)
        for path, expected in before.items(): self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_legacy_methods_without_available_flag_still_get_control(self):
        artifact, _, _ = self.artifact()
        file = artifact / "snapshots" / "snapshot_summary.json"
        value = json.loads(file.read_text(encoding="utf-8"))
        for row in value["snapshots"]:
            for method in ("hyperbolic","direction"): row["retrieval"][method].pop("available", None)
        file.write_text(json.dumps(value), encoding="utf-8")
        result = self.audit([artifact])
        control = result["snapshots"]["snapshots"][0]["retrieval"]["equal_radius_hyperbolic"]
        self.assertTrue(control["available"])
        self.assertGreater(control["nearest_distance_quantiles"]["p50"], 0)

    def test_equal_storage_arm_names_do_not_merge_run_lifetimes(self):
        a, _, _ = self.artifact("v5", "H20_v5_verified")
        b, _, _ = self.artifact("v6", "H20_v6_verified")
        result = self.audit([a,b])
        keys = {row["run_key"] for row in result["snapshots"]["snapshots"]}
        self.assertEqual(keys, {"H20_v5_verified", "H20_v6_verified"})
        self.assertEqual(len(result["extensions"]["longitudinal"]["runs"]), 2)
        arrays = [row["arrays_file"] for row in result["snapshots"]["snapshots"]]
        self.assertEqual(len(arrays), len(set(arrays)))

    def test_source_tree_is_disjoint_and_preexisting_audit_preserved(self):
        artifact, _, _ = self.artifact()
        with self.assertRaises(ValueError):
            run_system_audit([artifact], artifact / "nested", do_mechanisms=False, do_structure=False)
        out = self.root / "existing"; out.mkdir(); (out/"keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            run_system_audit([artifact], out, do_mechanisms=False, do_structure=False)
        self.assertEqual((out/"keep.txt").read_text(), "keep")

    def test_snapshot_only_artifacts_keep_real_arrays_without_fake_replay(self):
        artifact, _, _ = self.artifact(with_specs=False)
        result = self.audit([artifact])
        self.assertEqual(len(result["snapshots"]["snapshots"]), 2)
        self.assertNotIn("equal_radius_hyperbolic", result["snapshots"]["snapshots"][0]["retrieval"])
        self.assertEqual(result["manifest"]["GPU_forwards"], 0)

    def test_last_alias_uses_manifest_not_largest_available_cache(self):
        _, specs, analysis = self.artifact()
        analysis["identity"]["best"] = {}
        analysis["inventory"]["manifest_completed_epochs"] = 300
        selected, availability = _selection(specs, [analysis], {}, ["last", "best"], 2)
        self.assertEqual(selected, [])
        self.assertEqual(availability[0]["epoch"], 300)
        self.assertFalse(availability[0]["available"])
        self.assertIn("not guessed", availability[1]["reason"])

    def test_source_running_status_is_rejected_before_output_is_created(self):
        artifact, _, _ = self.artifact()
        (artifact/"audit_manifest.json").write_text('{"status":"running"}')
        with self.assertRaises(ValueError):
            self.audit([artifact])
        self.assertFalse((self.root/"combined").exists())

    def test_changed_proxy_mapping_forces_reanalysis(self):
        artifact, _, _ = self.artifact()
        file = artifact/"feature_exports"/"snapshot_spec.json"
        value = json.loads(file.read_text())
        for spec in value["snapshots"]: spec["proxy_mapping"]["numeric_radius_fraction"] = .998
        file.write_text(json.dumps(value))
        result = self.audit([artifact])
        self.assertNotIn("reuse", result["snapshots"]["snapshots"][0])
        self.assertEqual(result["snapshots"]["snapshots"][0]["proxy"]["mapping"]["numeric_radius_fraction"], .998)

    def test_missing_source_cache_is_not_invented(self):
        artifact, specs, _ = self.artifact()
        Path(specs[0]["cache"]).unlink()
        with self.assertRaises(ValueError):
            self.audit([artifact])
        self.assertFalse((self.root/"combined").exists())

    def test_ambiguous_legacy_h20_identity_is_rejected(self):
        artifact, _, analysis = self.artifact()
        other = copy.deepcopy(analysis); other["audit_run_key"]="other_verified"; other["run_id"]="other_verified"
        (artifact/"normalized_runs.json").write_text(json.dumps([analysis,other]))
        file=artifact/"feature_exports"/"snapshot_spec.json"
        value=json.loads(file.read_text())
        for spec in value["snapshots"]: spec["run_key"]="H20"
        file.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.audit([artifact])


if __name__ == "__main__":
    unittest.main()
