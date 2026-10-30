"""CPU contracts for F/G/H dropout and regularizer tap placement."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from segmentation_ShapeNetPart.hycore_seg_v1.vendor.dgcnn_model import DGCNN_partseg
from segmentation_ShapeNetPart.hycore_seg_v2.native import MobiusLayer
from segmentation_ShapeNetPart.hycore_seg_v3.model import NativePseg3
from segmentation_ShapeNetPart.hycore_seg_v3.monitoring import GeometryMonitor
from segmentation_ShapeNetPart.hycore_seg_v5.model import NativeTail2 as FrozenTail2
from segmentation_ShapeNetPart.hycore_seg_v5.losses import HyCoReSegLoss as FrozenLoss
if __package__:
    from .model import NativeTail2
    from .losses import HyCoReSegLoss
    from . import train as trainer
else:
    import sys
    name = "segmentation_ShapeNetPart.hycore_seg_v6"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("__init__.py"), submodule_search_locations=[str(Path(__file__).parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
    from segmentation_ShapeNetPart.hycore_seg_v6.losses import HyCoReSegLoss
    from segmentation_ShapeNetPart.hycore_seg_v6 import train as trainer
ARMS = ("f", "g", "h")
def fresh(seed=4780, arm="f"):
    torch.manual_seed(seed)
    return NativeTail2(arm, k=4)
def fixture():
    generator = torch.Generator().manual_seed(811)
    return torch.randn(3, 3, 24, generator=generator), torch.eye(16)[:3]
def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    def test_all_topologies_original_conv9_and_two_official_maps(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = fresh(arm=arm)
                self.assertEqual(model.arm, arm)
                self.assertEqual(model.architecture, "pseg3_fgh_v6_e9")
                self.assertEqual(model.feature_dim, 256)
                self.assertTrue(model.uses_category)
                self.assertFalse(model.has_m1)
                self.assertTrue(model.has_euclidean_conv9)
                self.assertEqual(model.regularizer_feature_stage, "after_conv9_bn9_leaky" if arm == "h" else "before_conv9")
                self.assertIsInstance(model.dp1, nn.Dropout if arm == "g" else nn.Identity)
                self.assertEqual(model.dropout_active, arm == "g")
                if arm == "g": self.assertEqual(model.dp1.p, .5)
                self.assertEqual((model.conv9[0].in_channels, model.conv9[0].out_channels), (256, 256))
                self.assertIsNone(model.conv9[0].bias)
                self.assertIs(model.conv9[1], model.bn9)
                self.assertTrue(model.bn9.affine)
                self.assertIsInstance(model.conv9[2], nn.LeakyReLU)
                self.assertEqual(model.conv9[2].negative_slope, .2)
                self.assertEqual(sum(isinstance(m, MobiusLayer) for m in model.modules()), 2)
                self.assertEqual((model.emb2.in_features, model.emb2.out_features), (256, 128))
                self.assertEqual((model.mlr.in_features, model.mlr.out_features), (128, 50))
                for name in ("emb", "conv10", "conv11", "bn10", "dp2"):
                    self.assertFalse(hasattr(model, name), name)
                    self.assertFalse(any(k.startswith(name + ".") for k in model.state_dict()), name)
                self.assertEqual({id(p) for p in model.head_parameters()}, {id(p) for layer in (model.emb2, model.mlr) for p in layer.parameters() if p.requires_grad})
                self.assertEqual(count(model), 1460283)
    def test_all_arms_exact_old_d_initial_state_and_constructor_rng(self):
        torch.manual_seed(4780)
        old = FrozenTail2("d", k=4)
        after_old = torch.get_rng_state().clone()
        for arm in ARMS:
            model = fresh(arm=arm)
            self.assert_exact(torch.get_rng_state(), after_old)
            self.assertEqual(set(model.state_dict()), set(old.state_dict()))
            for name, value in old.state_dict().items(): self.assert_exact(model.state_dict()[name], value)
    def test_conv9_original_surviving_b_parameters_unchanged(self):
        model = fresh()
        torch.manual_seed(4780)
        baseline = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50)
        torch.manual_seed(4780)
        b = NativePseg3("b", k=4)
        ds, bs = model.state_dict(), b.state_dict()
        self.assertEqual(set(ds) - set(bs), {k for k in ds if k.startswith(("conv9.", "bn9."))})
        for name, value in bs.items(): self.assert_exact(ds[name], value)
        for name, value in baseline.state_dict().items():
            if name in ds: self.assert_exact(ds[name], value)
        self.assertEqual(count(model) - count(b), 66048)
        self.assertEqual(count(baseline) - count(model), 78)
    def test_constructor_reproducible_preserves_b_rng(self):
        first = fresh()
        after = torch.rand(17)
        second = fresh()
        self.assert_exact(torch.rand(17), after)
        for name, value in first.state_dict().items(): self.assert_exact(second.state_dict()[name], value)
        torch.manual_seed(4780)
        NativePseg3("b", k=4)
        self.assert_exact(torch.rand(17), after)
    def test_main_frontend_matches_original_with_only_approved_dp1_switch(self):
        points, labels = fixture()
        for arm in ARMS:
            for training in (False, True):
                with self.subTest(arm=arm, training=training):
                    model = fresh(arm=arm).train(training)
                    monitor = GeometryMonitor().begin(True)
                    model.monitor = monitor
                    torch.manual_seed(4780)
                    reference = DGCNN_partseg(SimpleNamespace(k=4, emb_dims=1024, dropout=.5), 50).train(training)
                    if arm != "g": reference.dp1 = nn.Identity()
                    captured = []
                    handle = reference.conv9.register_forward_hook(lambda m, a, out: captured.append(out.detach().clone()))
                    torch.manual_seed(113)
                    with torch.no_grad(): reference(points, labels)
                    handle.remove()
                    torch.manual_seed(113)
                    with torch.no_grad(): model(points, labels)
                    self.assert_exact(monitor.nodes["h"].transpose(1, 2), captured[0])
    def test_regularizer_returns_correct_tap_main_always_uses_conv9(self):
        points, labels = fixture()
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = fresh(arm=arm).train()
                monitor = GeometryMonitor().begin(True)
                model.monitor = monitor
                before, after = [], []
                handles = [model.conv9.register_forward_pre_hook(lambda m, a: before.append(a[0])), model.conv9.register_forward_hook(lambda m, a, out: after.append(out))]
                logits, source = model(points, labels, return_features=True)
                for handle in handles: handle.remove()
                self.assertEqual(len(before), 1)
                self.assertEqual(len(after), 1)
                self.assert_exact(monitor.nodes["pre_conv9"].transpose(1, 2), before[0])
                self.assert_exact(monitor.nodes["post_conv9"].transpose(1, 2), after[0])
                self.assertIs(monitor.nodes["h"], monitor.nodes["post_conv9"])
                self.assertIs(source, monitor.nodes["regularizer_source"])
                self.assertIs(source, monitor.nodes["post_conv9" if arm == "h" else "pre_conv9"])
                self.assertEqual(logits.shape, (3, 50, 24))
                self.assertEqual(source.shape, (3, 24, 256))
                self.assertNotIn("m1", monitor.nodes)
    def test_single_forward_all_fourteen_bn_counters_once(self):
        points, labels = fixture()
        for arm in ARMS:
            model = fresh(arm=arm).train()
            logits, source = model(points, labels, return_features=True)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(source).all())
            counters = [int(m.num_batches_tracked) for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
            self.assertEqual(len(counters), 14)
            self.assertEqual(counters, [1] * 14)
    def test_ce_conv9_maps_regularizers_approved_gradient_routes(self):
        points, labels = fixture()
        categories = torch.tensor([0, 1, 2])
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = fresh(arm=arm).train()
                monitor = GeometryMonitor().begin(True)
                model.monitor = monitor
                criterion = HyCoReSegLoss(.01, .01, part_min=4, part_max=8, seed=4780, feature_dim=256)
                criterion.monitor = monitor
                logits, source = model(points, labels, return_features=True)
                regularizer, diagnostics = criterion(source, categories)
                ce = F.cross_entropy(logits.transpose(1, 2).reshape(-1, 50), torch.tensor([0, 4, 6]).repeat_interleave(24))
                self.assert_exact(monitor.nodes["whole_tangent"], source.mean(dim=1))
                self.assertEqual(diagnostics["negative_legal_rate"], 1.)
                targets = [model.conv9[0].weight, model.bn9.weight, model.bn9.bias, model.conv8[0].weight, *model.head_parameters()]
                for loss_name, loss in (("reg", regularizer), *criterion.last_components.items()):
                    grads = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
                    for grad in grads[:3]: self.assertEqual(grad is not None, arm == "h", (arm, loss_name))
                    self.assertIsNotNone(grads[3])
                    self.assertTrue(all(g is None for g in grads[4:]))
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))
                ce_grads = torch.autograd.grad(ce, targets, retain_graph=True, allow_unused=True)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in ce_grads))
                self.assertGreater(float(ce_grads[0].norm()), 0.)
                surrogate, = torch.autograd.grad(source.square().mean(), [model.conv9[0].weight], retain_graph=True, allow_unused=True)
                self.assertEqual(surrogate is not None, arm == "h")
                before = model.bn9.num_batches_tracked.clone()
                (ce + regularizer).backward()
                self.assert_exact(model.bn9.num_batches_tracked, before)
    def test_head_exact_official_two_map_composition(self):
        model = fresh().eval()
        tangent = torch.randn(3, 11, 256) * .02
        with torch.no_grad():
            expected = model.mlr(model.emb2(model.manifold.expmap0(tangent)))
            actual = model.forward_head(tangent)
        self.assert_exact(actual, expected)
        self.assertLessEqual(float(actual.norm(dim=-1).max()), .996001)
    def test_f_h_main_forward_and_rng_identical_regularizer_taps_differ(self):
        points, labels = fixture()
        outputs = []
        for arm in ("f", "h"):
            model = fresh(arm=arm).train()
            torch.manual_seed(113)
            logits, source = model(points, labels, return_features=True)
            outputs.append((logits, source, torch.get_rng_state().clone(), model.state_dict()))
        self.assert_exact(outputs[0][0], outputs[1][0])
        self.assert_exact(outputs[0][2], outputs[1][2])
        self.assertFalse(torch.equal(outputs[0][1], outputs[1][1]))
        for name, value in outputs[0][3].items(): self.assert_exact(outputs[1][3][name], value)
    def test_g_main_frozen_d_with_regularizer_moved_before_conv9(self):
        points, labels = fixture()
        torch.manual_seed(4780)
        old = FrozenTail2("d", k=4).train()
        new = fresh(arm="g").train()
        outputs = []
        for model in (old, new):
            torch.manual_seed(113)
            logits, source = model(points, labels, return_features=True)
            outputs.append((logits, source, torch.get_rng_state().clone()))
        self.assert_exact(outputs[0][0], outputs[1][0])
        self.assert_exact(outputs[0][2], outputs[1][2])
        self.assertFalse(torch.equal(outputs[0][1], outputs[1][1]))
    def test_regularizer_kernel_and_independent_rng_match_frozen_version(self):
        generator = torch.Generator().manual_seed(431)
        features = (torch.randn(3, 24, 256, generator=generator) * .02).requires_grad_()
        categories = torch.tensor([0, 1, 2])
        values = []
        before = torch.get_rng_state().clone()
        for cls in (FrozenLoss, HyCoReSegLoss):
            criterion = cls(.01, .01, part_min=4, part_max=8, seed=4780, feature_dim=256)
            reg, diagnostics = criterion(features, categories)
            grad, = torch.autograd.grad(reg, features)
            values.append((reg, diagnostics, grad, criterion.generator.get_state(), criterion.negative_generator.get_state()))
        for index in (0, 2, 3, 4): self.assert_exact(values[0][index], values[1][index])
        self.assertEqual(values[0][1], values[1][1])
        self.assert_exact(torch.get_rng_state(), before)
    def test_monitoring_preserves_outputs_gradients_rng(self):
        points, labels = fixture()
        for arm in ARMS:
            results = []
            for enabled in (False, True):
                model = fresh(arm=arm).train()
                model.monitor = GeometryMonitor().begin(enabled)
                torch.manual_seed(725)
                logits, source = model(points, labels, return_features=True)
                gradients = torch.autograd.grad(logits.square().mean() + source.square().mean(), [p for p in model.parameters() if p.requires_grad], allow_unused=True)
                results.append((logits, source, gradients, torch.get_rng_state().clone()))
            for index in (0, 1, 3): self.assert_exact(results[0][index], results[1][index])
            for actual, expected in zip(results[0][2], results[1][2]):
                if expected is None: self.assertIsNone(actual)
                else: self.assert_exact(actual, expected)
    def test_approved_coefficients_and_configs_enforced(self):
        common = ["--data", "unused", "--out", "unused", "--device", "cpu"]
        for arm in ARMS:
            args = trainer.arguments([*common, "--arm", arm])
            self.assertEqual((args.alpha, args.beta), (.01, .01))
            for alpha, beta in ((0., 0.), (.01, 0.), (0., .01), (.02, .02)):
                with self.assertRaises(ValueError): trainer.arguments([*common, "--arm", arm, "--alpha", str(alpha), "--beta", str(beta)])
            config = json.loads((Path(__file__).parent / "configs" / ("fgh_" + arm + ".json")).read_text())
            self.assertEqual(config["arm"], arm)
            self.assertEqual((config["alpha"], config["beta"]), (.01, .01))
            self.assertEqual((config["epochs"], config["lr"], config["min_lr"], config["seed"]), (200, .1, .001, 4780))
    def test_invalid_arm_rejected_before_rng_consumption(self):
        torch.manual_seed(4780)
        before = torch.get_rng_state().clone()
        for arm in ("a", "d", "e", "invalid"):
            with self.assertRaises(ValueError): NativeTail2(arm, k=4)
        self.assert_exact(torch.get_rng_state(), before)
    def test_cpu_contracts_do_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())
if __name__ == "__main__":
    unittest.main(verbosity=2)
