"""CPU contracts for the S5-TAN-R0 CLI, loss and complete training loop.

Uses the existing v6 synthetic train/val fixture; never accesses real data,
production checkpoints, a test split or a GPU. Optional --evidence-out writes
only a small acceptance summary; temporary fixtures/checkpoints remain private.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v1.utils import (
    restore_rng, rng_state, state_hash,
)
from segmentation_ShapeNetPart.hycore_seg_v6.cpu_entrypoint_check import write_fixture
from segmentation_ShapeNetPart.hycore_seg_v10 import train as trainer
from segmentation_ShapeNetPart.hycore_seg_v10.losses import HyCoReSegLoss
from segmentation_ShapeNetPart.hycore_seg_v10.monitoring import GeometryMonitor

EVIDENCE = {}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError("CPU tests must not initialize CUDA")

    def test_cli_only_h_scale5_and_zero_regularizers(self):
        common = ["--data", "unused", "--out", "unused", "--arm", "h", "--device", "cpu"]
        args = trainer.arguments(common)
        self.assertEqual((args.alpha, args.beta, args.supervision_scale), (0., 0., 5.))
        for extra in (["--regularizers"], ["--supervision-scale", "1"],
                      ["--alpha", ".01"], ["--beta", ".01"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                trainer.arguments(common + extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            trainer.arguments(common + ["--arm", "g"])
        EVIDENCE["cli_rejects_unapproved_arm_scale_regularizers"] = True

    def test_scaled_ce_gradient_and_detached_regularizer_rng(self):
        generator = torch.Generator().manual_seed(391)
        q = torch.randn(2, 7, 50, generator=generator, dtype=torch.float64,
                        requires_grad=True)
        labels = torch.randint(50, (2, 7), generator=generator)
        gradient, = torch.autograd.grad(trainer.plain_ce(5 * q, labels) / 5, q)
        expected = ((5 * q).softmax(-1) - F.one_hot(labels, 50)) / 14
        torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
        features = torch.randn(4, 12, 256, generator=generator, requires_grad=True)
        categories = torch.tensor([0, 1, 2, 3])
        before = torch.get_rng_state().clone()
        loss = HyCoReSegLoss(0., 0., part_min=3, part_max=5, seed=4780)
        part_before = loss.generator.get_state().clone()
        negative_before = loss.negative_generator.get_state().clone()
        regularizer, diagnostic = loss(features, categories)
        self.assertFalse(regularizer.requires_grad)
        self.assertEqual(float(regularizer), 0.)
        self.assertTrue(all(not value.requires_grad for value in loss.last_components.values()))
        self.assertFalse(diagnostic["regularizers_active"])
        self.assertTrue(diagnostic["regularizer_diagnostics_only"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.equal(part_before, loss.generator.get_state()))
        self.assertFalse(torch.equal(negative_before, loss.negative_generator.get_state()))
        EVIDENCE["scaled_ce_exact_gradient_and_R0_no_training_graph"] = True
        EVIDENCE["diagnostics_advance_only_independent_rng"] = True

    def test_readout_observation_is_real_scores_and_correct_probability(self):
        generator = torch.Generator().manual_seed(276)
        q = torch.randn(2, 7, 50, generator=generator, requires_grad=True)
        targets = torch.randint(50, (2, 7), generator=generator)
        monitor = GeometryMonitor().begin(True, 1, 1)
        monitor.capture("final_tangent", q)
        monitor.capture_targets(targets, scale=5.)
        rng_before = torch.get_rng_state().clone()
        observation = monitor.summarize()["geometry"]["final_tangent"]
        expected = (5 * q.detach()).softmax(-1).gather(-1, targets[..., None]).mean()
        self.assertAlmostEqual(observation["scaled_correct_class_probability"]["mean"],
                               float(expected), places=7)
        for key in ("near_boundary_fraction", "projection_radius", "depth", "normalized_radius"):
            self.assertNotIn(key, observation)
        self.assertEqual(observation["coordinate_space"], "origin_tangent_real_scores")
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertIsNone(q.grad)
        monitor.clear()
        self.assertEqual(monitor.nodes, {})
        EVIDENCE["q_real_score_statistics_and_true_class_probability"] = True

    def test_complete_cpu_two_epoch_training_checkpoint_contract(self):
        with tempfile.TemporaryDirectory(prefix="hycore_tan_cpu_") as temporary:
            root = Path(temporary)
            write_fixture(root / "data")
            output = root / "run"
            output.mkdir()
            args = trainer.arguments([
                "--data", str(root / "data"), "--out", str(output), "--arm", "h",
                "--protocol", "validation", "--epochs", "2", "--schedule-epochs", "200",
                "--batch-size", "4", "--eval-batch-size", "4", "--num-points", "32",
                "--k", "4", "--workers", "0", "--cpu-threads", "2", "--device", "cpu",
                "--part-min", "4", "--part-max", "8", "--max-steps", "2",
                "--monitor-every", "1", "--gradient-every", "1",
                "--no-final-test", "--allow-dirty-cpu",
            ])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer.train(args)
            manifest = read_json(output / "manifest.json")
            status = read_json(output / "status.json")
            accepted = read_json(output / "acceptance.json")
            metrics = read_lines(output / "metrics.jsonl")
            observations = read_lines(output / "geometry.jsonl")
            checkpoints = {name: torch.load(output / (name + ".pth"), map_location="cpu",
                                             weights_only=False)
                           for name in ("e0", "last", "best")}
            initial, last, best = (checkpoints[k] for k in ("e0", "last", "best"))
            self.assertEqual(status["state"], "complete")
            self.assertTrue(accepted["complete"])
            self.assertEqual((accepted["epochs"], accepted["updates"]), (2, 4))
            self.assertEqual(manifest["format"], "hycore_seg_v10")
            self.assertEqual(last["format"], "hycore_seg_v10_complete")
            self.assertEqual(manifest["display_name"], "S5-TAN-R0")
            self.assertEqual(manifest["final_output_space"], "real_tangent_scores")
            self.assertIsNone(manifest["gpu"])
            self.assertFalse(manifest["dropout_active"])
            self.assertFalse(manifest["gradient_clipping"])
            self.assertFalse(manifest["regularizers_active"])
            self.assertTrue(manifest["regularizer_diagnostics_only"])
            self.assertEqual(manifest["supervision_scale"], 5.)
            self.assertEqual((initial["epoch"], initial["updates"]), (0, 0))
            self.assertEqual((last["epoch"], last["updates"]), (2, 4))
            self.assertIn(best["epoch"], (1, 2))
            self.assertEqual(last["scheduler"]["T_max"], 200)
            self.assertEqual(last["scheduler"]["last_epoch"], 2)
            self.assertGreater(last["optimizer"]["param_groups"][0]["lr"], .099)
            self.assertTrue(trainer.finite_tensor_state(last["model"]))
            self.assertTrue(trainer.finite_tensor_state(last["optimizer"]))
            self.assertEqual(state_hash(initial["model"]), manifest["initial_model_sha256"])
            counts = {int(v) for k, v in last["model"].items()
                      if k.endswith("num_batches_tracked")}
            self.assertEqual(counts, {4})
            self.assertEqual([m["epoch"] for m in metrics], [1, 2])
            self.assertEqual(len(observations), 4)
            for row in metrics:
                self.assertEqual(row["steps"], 2)
                self.assertEqual((row["objects_seen"], row["val"]["objects"]), (8, 8))
                self.assertEqual(len(row["train"]["parts"]), 50)
                self.assertEqual(len(row["val"]["parts"]), 50)
                self.assertEqual(row["train_reg"], 0.)
                self.assertEqual(row["train_loss"], row["train_ce"])
            for row in observations:
                geometry = row["geometry"]
                self.assertIn("final_tangent", geometry)
                self.assertNotIn("m3", geometry)
                self.assertIn("near_boundary_fraction", geometry["m2"])
                self.assertIn("analytic_projection_rate", geometry["point_exp"])
                self.assertIn("scaled_correct_class_probability", geometry["final_tangent"])
                self.assertNotIn("near_boundary_fraction", geometry["final_tangent"])
                self.assertFalse(row["clip_applied"])
                self.assertFalse(row["diagnostics"]["regularizers_active"])
                self.assertTrue(row["gradients"]["ce"]["loss_requires_grad"])
                for component in ("contr", "hier"):
                    gradient = row["gradients"][component]
                    self.assertFalse(gradient["loss_requires_grad"])
                    self.assertTrue(all(v["status"] == "none" for v in gradient["nodes"].values()))
                    self.assertTrue(all(v["status"] == "none"
                                        for v in gradient["parameter_groups"].values()))
            self.assertFalse((root / "data/test.h5").exists())
            self.assertFalse((output / "final_test.json").exists())
            self.assertEqual(accepted["final_test_evaluations"], 0)
            self.assertFalse(initial["rng"]["cuda"])
            self.assertFalse(last["rng"]["cuda"])
            saved_rng = rng_state()
            try:
                restore_rng(last["rng"])
                first = torch.rand(5)
                restore_rng(last["rng"])
                self.assertTrue(torch.equal(first, torch.rand(5)))
                restored = trainer.NativeTail2("h", k=4)
                restored.load_state_dict(last["model"], strict=True)
                self.assertEqual(state_hash(restored.state_dict()), state_hash(last["model"]))
                optimizer = trainer.geoopt.optim.RiemannianSGD(
                    restored.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
                optimizer.load_state_dict(last["optimizer"])
                self.assertEqual(len(optimizer.state), len(last["optimizer"]["state"]))
                self.assertTrue(trainer.finite_tensor_state(optimizer.state_dict()))
            finally:
                restore_rng(saved_rng)
            self.assertFalse(torch.cuda.is_initialized())
            EVIDENCE["cpu_complete_entrypoint"] = {
                "epochs": 2, "updates": 4, "BN_updates": 4, "test_accessed": False,
                "all_50_part_metrics": True, "checkpoint_and_optimizer_restorable": True,
                "scheduler_horizon": 200, "q_geometry_not_ball": True,
                "initial_model_sha256": manifest["initial_model_sha256"],
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evidence-out")
    extra, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if extra.evidence_out:
        evidence = {"complete": result.result.wasSuccessful(),
                    "tests_run": result.result.testsRun, "cuda_initialized": torch.cuda.is_initialized(),
                    "scope": "synthetic_CPU_contracts_only_not_production_score", **EVIDENCE}
        Path(extra.evidence_out).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.result.wasSuccessful() else 1)
