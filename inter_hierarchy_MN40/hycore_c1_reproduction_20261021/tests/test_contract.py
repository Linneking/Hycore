"""Independent CPU gates for untouched official c1 geometry, losses, loader RNG and queue scope."""
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import geoopt
import h5py
import numpy as np
import torch

from .. import train, queue

REPO = Path(__file__).resolve().parents[3]


class Ball(geoopt.PoincareBall):
    def __init__(self, dim, c=1.0):
        super().__init__(c=c)
        self.register_buffer("dim", torch.tensor(dim))


class OfficialC1Contracts(unittest.TestCase):
    def test_actual_native_constructor_and_cpu_identity_are_unchanged(self):
        before_cuda = torch.cuda.is_initialized()
        original = train.import_original(REPO)
        torch.manual_seed(4780)
        model = original.Hype_pointMLP()
        before = train.state_digest(model)
        self.assertIs(train.configure_model(model), model)
        self.assertEqual(before, train.state_digest(model))
        manifolds = (model.manifold, model.manifold2, model.emb.manifold, model.classifier.manifold)
        self.assertTrue(all(abs(float(ball.c) - 1) < 1e-6 for ball in manifolds))
        self.assertTrue(all(not module._forward_hooks and not module._backward_hooks for module in model.modules()))
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)
        path = os.environ.get("HYCORE_C1_CONTRACT_IDENTITY")
        if path:
            train.atomic_json(Path(path), {"initialized_model_sha256": before, "seed": 4780,
                              "native_c": 1, "module_replaced": False, "extra_hooks": 0,
                              "cuda_initialized": torch.cuda.is_initialized()})

    def test_official_loss_function_is_not_replaced_or_wrapped(self):
        original = train.import_original(REPO)
        expected = ast.parse((REPO / "classification_ModelNet40/hutil.py").read_text(encoding="utf-8-sig"))
        source_function = next(node for node in expected.body if isinstance(node, ast.FunctionDef)
                               and node.name == "hype_triplet_losses")
        actual = ast.parse(inspect.getsource(original.hype_triplet_losses)).body[0]
        self.assertEqual(ast.dump(actual), ast.dump(source_function))
        self.assertEqual(Path(inspect.getsourcefile(original.train)).resolve(),
                         (REPO / "classification_ModelNet40/main_pointmlp_hycore.py").resolve())
        self.assertEqual(Path(inspect.getsourcefile(original.validate)).resolve(),
                         (REPO / "classification_ModelNet40/main_pointmlp_hycore.py").resolve())

    def test_native_c1_loss_values_and_parameter_gradients_match_direct_formula(self):
        source = REPO / "classification_ModelNet40/hutil.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "hype_triplet_losses")
        scope = {"torch": torch, "PoincareBall": Ball}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        raw = torch.tensor([[.31, .12], [-.15, .17], [.07, -.23]], dtype=torch.float64, requires_grad=True)
        ball = Ball(2, c=1)
        whole, part = ball.expmap0(raw), ball.expmap0(raw * .6)
        output = scope["hype_triplet_losses"](whole, part, hier_margin=1000/400, contr_margin=4, ball_dim=2)
        contrast = torch.maximum(ball.dist(whole, part) - ball.dist(whole, part.flip(0)) + 4,
                                 torch.zeros(3, dtype=raw.dtype)).mean()
        hierarchy = torch.maximum(-ball.dist0(whole) + ball.dist0(part) + 1000/400,
                                  torch.zeros(3, dtype=raw.dtype)).mean()
        expected = .01 * contrast + .01 * hierarchy
        actual = .01 * output[4] + .01 * output[5]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-13, rtol=1e-13))
        ga, = torch.autograd.grad(actual, raw, retain_graph=True)
        ge, = torch.autograd.grad(expected, raw)
        self.assertTrue(torch.allclose(ga, ge, atol=1e-12, rtol=1e-12))

    def test_original_c1_geometry_can_exceed_hier_depth_4point6(self):
        ball = Ball(2, c=1)
        point = ball.expmap0(torch.tensor([[2.7, 0.]], dtype=torch.float32))
        self.assertGreater(float(ball.dist0(point)[0]), 5.39)
        self.assertLess(float(point.norm()), 1)

    def test_original_data_augmentation_and_explicit_reference_row_order(self):
        spec = importlib.util.spec_from_file_location("isolated_original_data", REPO / "classification_ModelNet40/data.py")
        data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_module)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            points = np.arange(2*1024*3, dtype=np.float32).reshape(2,1024,3)/10000
            for number in (0, 1):
                with h5py.File(directory / ("ply_data_train%d.h5" % number), "w") as handle:
                    handle["data"], handle["label"] = points + number, np.full((2,1), number, dtype=np.int64)
            order = ["ply_data_train1.h5", "ply_data_train0.h5"]
            with mock.patch.dict(os.environ, {"HYCORE_TRAIN_SHARD_ORDER": json.dumps(order)}):
                loaded = train.install_data_root(data_module, directory)
                dataset = data_module.ModelNet40(1024, "train")
            np.random.seed(123)
            actual, label = dataset[0]
            np.random.seed(123)
            expected = data_module.translate_pointcloud(points[0]+1)
            np.random.shuffle(expected)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(int(label[0]), 1)
            self.assertEqual(loaded["train"]["file_order"], order)
            self.assertEqual(set(loaded), {"train"})

    def test_complete_production_config_and_train_only_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            common = ["--data-dir", root, "--run-dir", str(Path(root)/"new")]
            args = train.argument_parser().parse_args(common)
            train.validate_arguments(args)
            c = train.training_config(args)
            self.assertEqual((c["c"],c["clip_r"],c["gradient_hook"],c["radial_gamma"],c["contrastive_margin"]),
                             (1,None,False,1000,4))
            self.assertEqual((c["seed"],c["batch_size"],c["eval_batch_size"],c["workers"],c["epochs"],c["steps_per_epoch"]),
                             (4780,32,16,8,300,307))
            self.assertEqual((c["learning_rate"],c["min_lr"],c["cosine_period"]),(.1,.005,300))
            self.assertEqual(c["proxy_parameter_count"],0)
            args = train.argument_parser().parse_args(common+["--smoke","--epochs","2","--steps-per-epoch","2"])
            train.validate_arguments(args)
            self.assertEqual(train.training_config(args)["selection"], "none_train_only_smoke")
            self.assertEqual(list(train.LimitedLoader(range(50),2)),[0,1])
            args.epochs = 3
            with self.assertRaises(ValueError): train.validate_arguments(args)

    def test_source_route_has_no_curvature_or_gradient_adaptation(self):
        source = Path(train.__file__).read_text()
        tree = ast.parse(source)
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertIn("original.train", calls)
        self.assertIn("original.validate", calls)
        self.assertIn("original.Hype_pointMLP", calls)
        for forbidden in ("register_forward_hook", "InverseMetric", "to_ball", "half_margin", "install_loss_curvature", "autocast", "GradScaler"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("generator=", source)
        self.assertNotIn("worker_init_fn=", source)
        self.assertNotIn("--resume", source)

    def test_queue_pins_only_gpu2_and_ignores_busy_or_unknown_devices(self):
        idle = {"index":2,"uuid":"GPU-test","free_MiB":31000,"utilization":0,"compute_busy":False}
        self.assertEqual(queue.candidate([idle],2), idle)
        self.assertIsNone(queue.candidate([dict(idle,index=1)],2))
        self.assertIsNone(queue.candidate([dict(idle,compute_busy=True)],2))
        self.assertIsNone(queue.candidate([dict(idle,utilization=1)],2))
        with mock.patch.object(queue,"query_gpus", side_effect=subprocess.TimeoutExpired("nvidia-smi",20)):
            self.assertEqual(queue.observe_gpus(),[])
        env = queue.device_environment([idle])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"GPU-test")
        self.assertEqual(env["HYCORE_PHYSICAL_GPUS"],"2")

    def test_smoke_and_production_commands_are_fresh_and_separate(self):
        smoke = queue.command(Path("data"),Path("smoke"),True)
        main = queue.command(Path("data"),Path("main"))
        self.assertIn("--smoke",smoke)
        self.assertNotIn("--smoke",main)
        self.assertNotIn("--resume",main)
        self.assertEqual(main[main.index("--run-dir")+1],"main")

    def test_import_preserves_selected_gpu_uuid(self):
        class Loader:
            def exec_module(self, module):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        spec = types.SimpleNamespace(loader=Loader())
        with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"GPU-example","CUDA_DEVICE_ORDER":"FASTEST_FIRST"}):
            with mock.patch.object(train.importlib.util,"spec_from_file_location",return_value=spec):
                with mock.patch.object(train.importlib.util,"module_from_spec",return_value=types.SimpleNamespace()):
                    train.import_original(REPO)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"],"GPU-example")
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"],"FASTEST_FIRST")


if __name__ == "__main__":
    unittest.main()
