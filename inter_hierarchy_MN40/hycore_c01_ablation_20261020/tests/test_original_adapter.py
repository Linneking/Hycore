"""CPU contracts for original-source reuse, curvature, cap and data RNG."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
import geoopt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hycore_c01_ablation_20261020 import original_adapter as adapter


class Ball(geoopt.PoincareBall):
    def __init__(self, dim, c=1.0):
        super().__init__(c=c)
        self.register_buffer("dim", torch.tensor(dim))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.manifold = Ball(dim=512)
        self.manifold2 = Ball(dim=256)
        self.emb = torch.nn.Identity()
        self.emb.manifold = self.manifold
        self.classifier = torch.nn.Identity()
        self.classifier.manifold = self.manifold2


class OriginalAdapterTests(unittest.TestCase):
    def test_configuration_uses_same_two_fixed_c01_manifolds(self):
        model = adapter.configure_model(TinyModel(), capped=False)
        adapter.verify_curvature(model)
        self.assertIs(model.emb.manifold, model.manifold)
        self.assertIs(model.classifier.manifold, model.manifold2)
        self.assertEqual(len(model.emb._forward_hooks), 0)
        self.assertTrue(all(not p.requires_grad for p in model.manifold.parameters()))

    def test_cap_is_final_shared_mu_and_has_no_inverse_metric_node(self):
        model = adapter.configure_model(TinyModel(), capped=True)
        self.assertEqual(len(model.emb._forward_hooks), 1)
        tangent = torch.tensor([[.5, .0], [3., .1]], dtype=torch.float64, requires_grad=True)
        raw = model.manifold.expmap0(tangent)
        capped = model.emb(raw)
        depths = model.manifold.dist0(capped)
        self.assertLess(float(depths.max()), 4.60001)
        self.assertGreater(float(depths[1]), 4.5998)
        self.assertTrue(torch.allclose(capped[0], raw[0], atol=1e-10))
        pending, visited, names = [capped.grad_fn], set(), set()
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            names.add(type(node).__name__)
            pending.extend(item[0] for item in node.next_functions)
        self.assertFalse(any("InverseMetric" in name for name in names))
        capped.sum().backward()
        self.assertTrue(bool(torch.isfinite(tangent.grad).all()))

    def test_loss_is_official_function_with_only_ball_curvature_rebound(self):
        source = REPO / "classification_ModelNet40/hutil.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "hype_triplet_losses")
        scope = {"torch": torch, "PoincareBall": Ball}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        module = types.SimpleNamespace(PoincareBall=Ball)
        adapter.install_loss_curvature(module)
        scope["PoincareBall"] = module.PoincareBall
        tangent = torch.tensor([[.3, .0], [.1, .1], [.0, .2]], dtype=torch.float64)
        ball = Ball(2, c=.1)
        whole, part = ball.expmap0(tangent), ball.expmap0(tangent * .6)
        output = scope["hype_triplet_losses"](whole, part, hier_margin=1000/600, contr_margin=4, ball_dim=2)
        positive, negative = ball.dist(whole, part), ball.dist(whole, part.flip(0))
        expected_contr = (positive - negative + 4).clamp_min(0).mean()
        expected_hier = (-ball.dist0(whole) + ball.dist0(part) + 1000/600).clamp_min(0).mean()
        self.assertTrue(torch.allclose(output[4], expected_contr))
        self.assertTrue(torch.allclose(output[5], expected_hier))

    def test_data_reuses_official_augmentation_rng_and_smoke_never_loads_test(self):
        spec = importlib.util.spec_from_file_location("isolated_official_data", REPO / "classification_ModelNet40/data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            points = np.arange(4 * 1024 * 3, dtype=np.float32).reshape(4, 1024, 3) / 1000
            labels = np.arange(4, dtype=np.int64).reshape(-1, 1)
            with h5py.File(root / "ply_data_train0.h5", "w") as handle:
                handle["data"], handle["label"] = points, labels
            loaded = adapter.install_data_root(module, root)
            data = module.ModelNet40(1024, "train")
            np.random.seed(123)
            actual, label = data[2]
            np.random.seed(123)
            expected = module.translate_pointcloud(points[2][:1024])
            np.random.shuffle(expected)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(int(label[0]), 2)
            self.assertEqual(set(loaded), {"train"})
            self.assertEqual(loaded["train"]["file_order"], ["ply_data_train0.h5"])

    def test_limited_loader_produces_exact_two_updates(self):
        self.assertEqual(list(adapter.LimitedLoader(range(100), 2)), [0, 1])

    def test_smoke_arguments_refuse_test_or_partial_production_protocols(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            base = ["--variant", "c01", "--data-dir", str(root), "--run-dir", str(root / "new")]
            args = adapter.argument_parser().parse_args(base)
            adapter.validate_arguments(args)
            args = adapter.argument_parser().parse_args(base + ["--smoke", "--epochs", "2", "--steps-per-epoch", "2"])
            adapter.validate_arguments(args)
            args = adapter.argument_parser().parse_args(base + ["--epochs", "20"])
            with self.assertRaises(ValueError):
                adapter.validate_arguments(args)
            args = adapter.argument_parser().parse_args(base + ["--smoke"])
            with self.assertRaises(ValueError):
                adapter.validate_arguments(args)

    def test_entrypoint_reuses_original_train_and_validate(self):
        tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8-sig"))
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertIn("original.train", calls)
        self.assertIn("original.validate", calls)
        self.assertNotIn("original.main", calls)
        source = Path(adapter.__file__).read_text(encoding="utf-8-sig")
        self.assertNotIn("worker_init_fn=", source)
        self.assertNotIn("generator=", source)
        self.assertNotIn("InverseMetric.apply", source)

    def test_real_original_import_finds_frozen_extension_without_cuda_allocation(self):
        initialized = torch.cuda.is_initialized()
        original = adapter.import_original(REPO)
        self.assertTrue(callable(original.Hype_pointMLP))
        self.assertTrue(callable(original.train))
        self.assertTrue(callable(original.validate))
        self.assertEqual(torch.cuda.is_initialized(), initialized)

    def test_import_preserves_launch_gpu_selection(self):
        class Loader:
            def exec_module(self, module):
                adapter.os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                adapter.os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        fake_spec = types.SimpleNamespace(loader=Loader())
        with mock.patch.dict(adapter.os.environ, {"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}):
            with mock.patch.object(adapter.importlib.util, "spec_from_file_location", return_value=fake_spec):
                with mock.patch.object(adapter.importlib.util, "module_from_spec", return_value=types.SimpleNamespace()):
                    adapter.import_original(REPO)
            self.assertEqual(adapter.os.environ["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(adapter.os.environ["CUDA_DEVICE_ORDER"], "FASTEST_FIRST")


if __name__ == "__main__":
    unittest.main()
