"""CPU contracts for configuration, paired random streams and resume identity.

Extract pure entry-point helpers with AST so Windows can verify them without
loading the CUDA-only PointMLP extension. Server launch separately imports the
real training module, checks its original model digest, then executes DDP.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "train.py"


def helpers():
    import hashlib
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"CHECKPOINT_VERSION", "EXPECTED_ORIGINAL_INITIAL_DIGEST", "EXPECTED_TRAIN_LABELS_SHA",
        "EXPECTED_TEST_LABELS_SHA", "TRAIN_SHARD_ORDER"}
    functions = {"arguments", "fixed_configuration", "array_sha256", "load_shards", "make_generators",
        "capture_generators", "restore_generators", "validate_resume"}
    kept = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name in functions) or
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    module = ast.Module(body=kept, type_ignores=[])
    env = {"__doc__": "training contract helpers", "argparse": argparse, "Path": Path,
        "np": np, "torch": torch, "json": json, "os": os, "h5py": h5py, "hashlib": hashlib}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env


class TrainContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = helpers()

    def config(self, arm="off"):
        args = self.h["arguments"](["--hier", arm, "--data-dir", "not-read", "--run-dir", "not-created"])
        return self.h["fixed_configuration"](args)

    def test_production_pair_has_identical_base_and_budget(self):
        a, b = self.config(), self.config("on")
        for key in ("base_c", "native_c", "native_extra_cap", "native_GS", "radial_gamma", "contrastive_margin",
            "seed", "epochs", "steps_per_epoch", "workers", "eval_batch_size", "model_optimizer", "sampler"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual((a["base_c"], a["native_extra_cap"], a["native_GS"]), (.1, 2.3, True))
        self.assertEqual((a["seed"], a["epochs"], a["steps_per_epoch"]), (4780, 300, 300))
        self.assertEqual((a["radial_gamma"], a["contrastive_margin"]), (500., 2.))
        self.assertEqual((a["world_size"], a["classes_per_rank"], a["instances_per_class"]), (2, 16, 2))
        self.assertEqual(a["P"], 0)
        self.assertIsNone(a["proxy_optimizer"])
        self.assertIsNone(a["hier_view_depth"])
        self.assertEqual((b["P"], b["hier_view_depth"], b["lambda_hier_after_warmup"]), (512, 4.6, .5))
        self.assertEqual(b["proxy_optimizer"]["eps"], 1e-8)
        self.assertIn("GSbase1-GShier1", b["canonical_name"])

    def test_cli_rejects_unapproved_budget_or_seed(self):
        for extra in (["--steps-per-epoch", "307"], ["--steps-per-epoch", "200"], ["--seed", "22"],
            ["--epochs", "10"], ["--skip-final-test"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.h["arguments"](["--hier", "off", "--data-dir", "x", "--run-dir", "y", *extra])
        args = self.h["arguments"](["--hier", "on", "--data-dir", "x", "--run-dir", "y",
            "--smoke", "--epochs", "2", "--steps-per-epoch", "2", "--skip-final-test"])
        self.assertTrue(args.smoke)

    def test_private_generators_leave_global_rng_unchanged_and_pair_data_streams(self):
        torch.manual_seed(900)
        before = torch.get_rng_state().clone()
        off = self.h["make_generators"](4780, 1, "cpu", False)
        on = self.h["make_generators"](4780, 1, "cpu", True)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        for _ in range(4):
            torch.rand(200, generator=on["gumbel"])
            for key in ("loader", "crop"):
                torch.testing.assert_close(torch.randint(0, 10000, (31,), generator=on[key]),
                    torch.randint(0, 10000, (31,), generator=off[key]), rtol=0, atol=0)
        self.assertIsNone(off["gumbel"])

    def test_all_private_streams_resume_exactly(self):
        streams = self.h["make_generators"](4780, 0, "cpu", True)
        saved = self.h["capture_generators"](streams)
        expected = {key: torch.rand(31, generator=gen) for key, gen in streams.items()}
        self.h["restore_generators"](streams, saved)
        for key, gen in streams.items():
            torch.testing.assert_close(torch.rand(31, generator=gen), expected[key], rtol=0, atol=0)
        bad = self.h["make_generators"](4780, 0, "cpu", False)
        with self.assertRaises(RuntimeError):
            self.h["restore_generators"](bad, saved)

    def test_fixed_historical_shard_order_is_independent_of_filesystem_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in (0, 3, 2, 4, 1):
                with h5py.File(Path(directory) / ("ply_data_train%d.h5" % number), "w") as file:
                    file["data"] = np.full((2, 4, 3), number, dtype=np.float32)
                    file["label"] = np.full((2, 1), number, dtype=np.int64)
            with mock.patch.dict(os.environ, {}, clear=True):
                points, labels, names = self.h["load_shards"](directory, "train")
            self.assertEqual(names, list(self.h["TRAIN_SHARD_ORDER"]))
            self.assertEqual(labels.tolist(), [1, 1, 4, 4, 3, 3, 2, 2, 0, 0])
            np.testing.assert_array_equal(points[:, 0, 0], labels)
            with mock.patch.dict(os.environ, {"HYCORE_TRAIN_SHARD_ORDER": json.dumps(list(reversed(names)))}):
                with self.assertRaises(ValueError):
                    self.h["load_shards"](directory, "train")

    def test_resume_rejects_mismatched_selected_weights_metrics(self):
        config = self.config()
        ids, test = np.arange(4), np.arange(2)
        saved = {"format": self.h["CHECKPOINT_VERSION"], "training_config": config, "split_sha256": "x",
            "train_ids": ids, "selection_ids": test, "rank_states": [{}, {}], "completed_epochs": 2,
            "best": {"epoch": 1, "test_oa": 90.}, "best_net": {"weight": 1},
            "best_selection_metrics": {"test_oa": 90.}}
        self.h["validate_resume"](saved, config, "x", ids, test)
        saved["best_selection_metrics"]["test_oa"] = 91.
        with self.assertRaises(RuntimeError):
            self.h["validate_resume"](saved, config, "x", ids, test)

    def test_real_training_route_forks_before_base_gs_and_has_no_step_reseed(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        for node in ast.walk(main):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        self.assertNotEqual(child.func.id, "seed_all")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "whole_to_hier"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "values['hier_source_mu']")
        self.assertIn('if enabled:\n            from ..hier_proxy_scratch_v8.hier_loss import HIERLoss', SOURCE.read_text(encoding="utf-8"))
        self.assertIn('"completed_optimizer_steps": epoch * args.steps_per_epoch', SOURCE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
