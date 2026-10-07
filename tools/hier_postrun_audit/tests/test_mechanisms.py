"""Meaningful query/identity/geometry controls; Torch tests run on CPU only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.hier_postrun_audit.mechanisms import (
    _balanced_positions, _compatibility, _load, _remap,
    analyze_mechanisms, draw_agreement, gradient_summary, reference_distance)

TORCH = importlib.util.find_spec("torch") is not None


class MechanismIdentityTests(unittest.TestCase):
    def test_balanced64_is_fixed_unique_and_two_objects_per_selected_class(self):
        labels = np.repeat(np.arange(40), 5)
        first = _balanced_positions(labels, size=64, seed=22)
        second = _balanced_positions(labels, size=64, seed=22)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 64)
        self.assertEqual(len(np.unique(labels[first])), 32)
        np.testing.assert_array_equal(np.unique(labels[first], return_counts=True)[1], np.full(32, 2))

    def test_queries_map_object_ids_rather_than_row_positions(self):
        rows = _remap([[20, 10, 30]], np.array([30, 20, 10]))
        np.testing.assert_array_equal(rows, [[1, 2, 0]])
        with self.assertRaisesRegex(ValueError, "absent"):
            _remap([[99, 10, 30]], np.array([30, 20, 10]))

    def test_collision_agreement_is_not_ancestor_agreement(self):
        first = {"pair_proxy_ids": [1, 2], "triple_proxy_ids": [1, 3],
                 "collision_mask": [True, False], "active_hinge_mask": [False, True], "loss": .2}
        second = {"pair_proxy_ids": [4, 2], "triple_proxy_ids": [4, 3],
                  "collision_mask": [True, False], "active_hinge_mask": [False, True], "loss": .3}
        agreement = draw_agreement(first, second)
        self.assertEqual(agreement["collision_mask_agreement"], 1.)
        self.assertEqual(agreement["pair_proxy_ids_agreement"], .5)
        self.assertEqual(agreement["selected_proxy_set_jaccard"], .5)

    def test_empty_queries_are_unavailable_not_perfectly_stable(self):
        empty = {"pair_proxy_ids": [], "triple_proxy_ids": [],
                 "collision_mask": [], "active_hinge_mask": [], "loss": 0.}
        self.assertFalse(draw_agreement(empty, empty)["available"])

    def test_load_rejects_inventing_hidden_proxy_parameters_from_ball_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.npz"
            np.savez(cache, sample_ids=np.array([10, 20, 30]), labels=np.array([0, 0, 1]),
                     mu=np.array([[.1, 0], [.2, 0], [-.1, 0]]),
                     proxy_ball=np.array([[.999, 0]]), proxy_ids=np.array([0]))
            spec = {"cache": str(cache), "version": "v7", "run_key": "own", "epoch": 1,
                    "c": 1., "proxy_mapping": {"numeric_radius_fraction": .999}}
            with self.assertRaisesRegex(ValueError, "cannot reconstruct"):
                _load(spec)

    def test_temporal_pairing_rejects_changed_input_even_when_pool_matches(self):
        identity = {"sample_pool_sha256": "same_pool", "input_sha256": "input_A", "input_mode": "eval", "inference_condition_sha256": "same_inference"}
        plan = {"run_key": "own", "reference_identity": identity,
                "proxy_ids": [0], "sample_ids": [10, 20], "labels": [0, 0],
                "mining_parameters": {"exclude_self_negative": True}}
        snapshot = {"run_key": "own", "identity": {**identity, "input_sha256": "input_B"},
                    "proxy_ids": np.array([0]), "sample_ids": np.array([10, 20]),
                    "labels": np.array([0, 0]), "exclude_self_negative": True}
        self.assertIn("input_sha256", _compatibility(plan, snapshot))


    def test_inference_contract_change_refuses_temporal_pairing(self):
        identity = {"sample_pool_sha256": "same_pool", "input_sha256": "same_input",
                    "input_mode": "eval", "inference_condition_sha256": "FPS_A"}
        plan = {"run_key": "own", "reference_identity": identity,
                "proxy_ids": [0], "sample_ids": [10, 20], "labels": [0, 0],
                "mining_parameters": {"exclude_self_negative": True}}
        snapshot = {"run_key": "own", "identity": {**identity, "inference_condition_sha256": "FPS_B"},
                    "proxy_ids": np.array([0]), "sample_ids": np.array([10, 20]),
                    "labels": np.array([0, 0]), "exclude_self_negative": True}
        self.assertIn("inference_condition_sha256", _compatibility(plan, snapshot))

@unittest.skipUnless(TORCH, "CPU PyTorch is not installed on this local runtime")
class MechanismTorchTests(unittest.TestCase):
    def test_gradient_sign_and_angular_decomposition(self):
        import torch
        parameter = torch.tensor([[2., 0.], [0., 2.]])
        gradient = torch.tensor([[3., 4.], [0., -5.]])
        result = gradient_summary(parameter, gradient, np.array([7, 8]))
        self.assertAlmostEqual(result["radial_norm"], np.sqrt(34), places=5)
        self.assertAlmostEqual(result["angular_norm"], 4.)
        self.assertEqual(result["positive_signed_radial_count_above_tolerance"], 1)
        self.assertEqual(result["negative_signed_radial_count_above_tolerance"], 1)
        self.assertEqual(result["per_id"][0]["signed_radial"], 3.)

    def test_saturated_parameter_has_angular_but_negligible_radial_derivative(self):
        import torch
        from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import expmap0_c1
        tangent = torch.tensor([[8., 0.]], dtype=torch.float64, requires_grad=True)
        loss = expmap0_c1(tangent)[0, 1] + expmap0_c1(tangent)[0, 0]
        gradient, = torch.autograd.grad(loss, (tangent,))
        result = gradient_summary(tangent, gradient, np.array([0]))
        self.assertLess(abs(result["per_id"][0]["signed_radial"]), 1e-12)
        self.assertGreater(result["angular_norm"], .01)
        self.assertIsNone(tangent.grad)

    def test_reference_operator_is_distinct_at_boundary(self):
        import torch
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance
        points = torch.tensor([[.999, 0.], [-.999, 0.]], dtype=torch.float64)
        reference, identity = reference_distance()
        actual, other = poincare_distance(points), reference(points, points)
        self.assertGreater(float(actual[0, 1] - other[0, 1]), 2.)
        self.assertAlmostEqual(float(actual[0, 1]), 4. * np.arctanh(.999), places=10)
        self.assertIn("counterfactual", identity["operator"])

    def _cache(self, folder, name, input_sha="same_input", version="v6", reorder=False):
        rng = np.random.default_rng(131)
        labels = np.repeat(np.arange(4), 5)
        whole = rng.normal(size=(20, 4)).astype(np.float32)
        whole /= np.linalg.norm(whole, axis=1, keepdims=True)
        whole *= np.linspace(.2, .8, 20)[:, None]
        tangent = rng.normal(size=(8, 4)).astype(np.float32) * .3
        ids, proxy_ids = np.arange(100, 120), np.arange(8)
        if reorder:
            whole, labels, ids = whole[::-1], labels[::-1], ids[::-1]
            tangent, proxy_ids = tangent[::-1], proxy_ids[::-1]
        path = Path(folder) / name
        np.savez(path, mu=whole, labels=labels, sample_ids=ids,
                 proxy_tangent=tangent, proxy_ids=proxy_ids,
                 input_sha256=np.asarray(input_sha), input_mode=np.asarray("clean_eval"),
                 metadata_json=np.asarray(json.dumps({"inference_condition": {"batch_size": 32, "seed": 22, "FPS_source_sha256": "toy_fps"}})))
        return {"cache": str(path), "c": 1., "version": version, "run_key": "own",
                "epoch": 1 if not reorder else 2, "topk": 4, "proxy_topk": 3,
                "t_per_anchor": 4, "lambda_hier": 0., "tau": 1., "margin": .5,
                "proxy_mapping": {"numeric_radius_fraction": .999},
                "proxy_id_policy": "stable_parameter_rows_within_run"}

    def test_identical_snapshots_preserve_ids_noise_gradients_and_source_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            specs = [self._cache(folder, "a.npz"), self._cache(folder, "b.npz", reorder=True)]
            before = [hashlib.sha256(Path(spec["cache"]).read_bytes()).hexdigest() for spec in specs]
            report = analyze_mechanisms(specs, Path(folder) / "audit",
                                        query_count=8, noise_repeats=3, gradient_repeats=1,
                                        sample_batch_size=20, compare_source_operator=True)
            after = [hashlib.sha256(Path(spec["cache"]).read_bytes()).hexdigest() for spec in specs]
            self.assertEqual(before, after)
            transition = report["transitions"][0]
            self.assertTrue(transition["available"])
            for rows in transition["same_query_same_noise"].values():
                for row in rows:
                    if row["available"]:
                        self.assertEqual(row["pair_proxy_ids_agreement"], 1.)
                        self.assertEqual(row["triple_proxy_ids_agreement"], 1.)
            snapshot = report["snapshots"][0]
            for name, rows in snapshot["gradient_summaries"].items():
                self.assertEqual(rows[0]["proxy_tangent"]["gradient_norm"], 0.)
                if name == "sample":
                    self.assertGreater(rows[0]["unweighted_proxy_tangent"]["gradient_norm"], 0.)
            manifest = json.loads((Path(folder) / "audit" / "mechanism_manifest.json").read_text())
            self.assertEqual(manifest["optimizer_updates"], 0)
            self.assertEqual(manifest["GPU_forwards"], 0)
            self.assertEqual(manifest["status"], "completed")

    def test_optimizer_checkpoint_alias_validates_parameters_and_moment_row_identity(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            spec = self._cache(folder, "cache.npz", reorder=True)
            with np.load(spec["cache"], allow_pickle=False) as cache:
                saved_tangent = torch.from_numpy(cache["proxy_tangent"][::-1].copy())
            first = torch.arange(8, dtype=torch.float32)[:, None].expand(8, 4).clone()
            checkpoint = {"epoch": 2, "proxy": {"tangent_proxies": saved_tangent},
                          "proxy_optimizer": {"state": {0: {"step": torch.tensor(3.),
                                                            "exp_avg": first,
                                                            "exp_avg_sq": torch.ones_like(first)}},
                                              "param_groups": [{"params": [0], "lr": .01,
                                                                "betas": (.9, .999), "eps": 1e-8}]}}
            path = Path(folder) / "own_full.pth"
            torch.save(checkpoint, path)
            spec.update(optimizer_checkpoint=str(path), trusted_checkpoint=True)
            loaded = _load(spec)
            torch.testing.assert_close(loaded["optimizer"]["state"][0]["exp_avg"], first)
            self.assertEqual(loaded["identity"]["optimizer_checkpoint_file"], "own_full.pth")
            checkpoint["proxy"]["tangent_proxies"] = saved_tangent + .1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "parameters differ"):
                _load(spec)

    def test_changed_clean_input_refuses_temporal_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self._cache(folder, "a.npz")
            second = self._cache(folder, "b.npz", input_sha="other_input", reorder=True)
            result = analyze_mechanisms([first, second], Path(folder) / "audit",
                                       query_count=4, noise_repeats=2, gradient_repeats=0,
                                       sample_batch_size=20, compare_source_operator=False)
            self.assertFalse(result["transitions"][0]["available"])
            self.assertIn("input_sha256", result["transitions"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
