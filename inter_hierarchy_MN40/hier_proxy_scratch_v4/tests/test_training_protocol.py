"""CPU protocol checks without importing the PointMLP CUDA extension.

Extract the small protocol functions from the real training source, so these
checks exercise its implementations while avoiding unrelated GPU imports.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
import unittest

import numpy as np
import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
HUTIL_PATH = TRAIN_PATH.parent.parent / "hutil.py"


def extracted_functions(path, names, scope):
    source = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("protocol function disappeared: " + str(names))
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)


SCOPE = {"np": np, "torch": torch, "random": random,
         "hashlib": hashlib, "contextmanager": contextmanager}
extracted_functions(HUTIL_PATH, {"knn"}, SCOPE)
extracted_functions(TRAIN_PATH, {"stratified_split", "split_identity", "sample_subcloud",
                                 "part_batch_statistics_only"}, SCOPE)


class TrainingProtocolTests(unittest.TestCase):
    def test_validation_split_is_deterministic_disjoint_and_stratified(self):
        labels = np.concatenate([np.full(40 + label, label, dtype=np.int64)
                                 for label in range(40)])
        split = SCOPE["stratified_split"]
        train, validation = split(labels, seed=22, fraction=0.1)
        again_train, again_validation = split(labels, seed=22, fraction=0.1)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(validation, again_validation)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(len(labels))))
        self.assertTrue(np.all(np.bincount(labels[train], minlength=40) >= 8))
        self.assertTrue(np.all(np.bincount(labels[validation], minlength=40) >= 1))
        identity = SCOPE["split_identity"]
        other_train, other_validation = split(labels, seed=23, fraction=0.1)
        self.assertNotEqual(identity(train, validation), identity(other_train, other_validation))

    def test_sampling_keeps_source_and_whole_unchanged_and_backpropagates(self):
        torch.manual_seed(22)
        random.seed(22)
        source = torch.randn(4, 3, 40, requires_grad=True)
        original = source.detach().clone()
        whole, _ = SCOPE["sample_subcloud"](source, 30, 36)
        saved_whole = whole.detach().clone()
        child, count = SCOPE["sample_subcloud"](whole, 8, 12)
        self.assertTrue(torch.equal(source.detach(), original))
        self.assertTrue(torch.equal(whole.detach(), saved_whole))
        self.assertEqual(child.shape, (4, 3, count))
        self.assertNotEqual(child.data_ptr(), whole.data_ptr())
        child.square().sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(float(source.grad.norm()), 0)

    def test_part_bn_keeps_buffers_and_whole_updates_once(self):
        torch.manual_seed(22)
        model = torch.nn.Sequential(
            torch.nn.Conv1d(3, 8, 1), torch.nn.BatchNorm1d(8), torch.nn.ReLU(),
            torch.nn.Conv1d(8, 4, 1), torch.nn.BatchNorm1d(4), torch.nn.ReLU(),
        ).train()
        before = {name: value.clone() for name, value in model.named_buffers()}
        with SCOPE["part_batch_statistics_only"](model):
            part = model(torch.randn(4, 3, 8))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(layer.track_running_stats for layer in model.modules()
                            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)))
        whole = model(torch.randn(4, 3, 32))
        for name, value in model.named_buffers():
            if name.endswith("num_batches_tracked"):
                self.assertEqual(int(value), int(before[name]) + 1)
        (part.square().mean() + whole.square().mean()).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_part_bn_restores_flags_after_exception(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3)).train()
        layer = model[0]
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with SCOPE["part_batch_statistics_only"](model):
                self.assertFalse(layer.track_running_stats)
                raise RuntimeError("intentional")
        self.assertTrue(layer.track_running_stats)


if __name__ == "__main__":
    unittest.main()
