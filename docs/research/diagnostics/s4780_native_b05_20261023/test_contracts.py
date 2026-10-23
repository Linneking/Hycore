"""Source and CPU gates for the fresh S4780 native/B05 Hshell-A1 pair.

Run with the server's existing HyCoRe Python. Geometry and sampler tests need
no PointNet extension. Full namespace/model gates are skipped ONLY when the
local machine lacks Geoopt or the PointNet binary; all must run on the server.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for directory in (REPO, REPO / "inter_hierarchy_MN40", HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch
from torch import nn
import torch.nn.functional as F

from inter_hierarchy_MN40.hier_proxy_scratch_v9 import geometry as native_geometry
from inter_hierarchy_MN40.hier_proxy_cycle_v9c import geometry as b05_geometry
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler

try:
    import geoopt
except ModuleNotFoundError as error:
    if error.name != "geoopt":
        raise
    geoopt = None


def get_runner():
    spec = importlib.util.spec_from_file_location("s4780_pair_runner_contracts", HERE / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_gs_nodes(tensor):
    pending, visited, count = [tensor.grad_fn], set(), 0
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        count += "InverseMetricBackward" in type(node).__name__
        pending.extend(value for value, _ in node.next_functions)
    return count


def namespace(arm):
    try:
        return get_runner().build_namespace(arm=arm)
    except (ModuleNotFoundError, ImportError) as error:
        missing = getattr(error, "name", "") or ""
        message = str(error)
        if missing == "geoopt" or "pointnet2" in missing or "pointnet2" in message or "_ext" in message:
            raise unittest.SkipTest("Local model dependency unavailable; server gate must run: " + message)
        raise


class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = get_runner()
        cls.original_path = REPO / "inter_hierarchy_MN40/hier_proxy_cycle_v9c/train.py"
        cls.original = cls.original_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        cls.transformed = cls.runner.transformed_source(cls.original)

    def test_pinned_source_rejects_unreviewed_change(self):
        with self.assertRaises(RuntimeError):
            self.runner.transformed_source(self.original + "\n# unreviewed edit\n")

    def test_transformed_source_compiles_and_keeps_production_safeguards(self):
        ast.parse(self.transformed)
        for expression in (
            'torch.nn.utils.clip_grad_norm_(model_parameters, 1.)',
            'optimizer.step()', 'proxy_optimizer.step()',
            'sync_parameter_gradients(proxy.parameters())',
            'candidate = select_best(best, selection, selection_prefix, epoch)',
            'rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)',
            'if rank == 0:', 'initialization_e0.pth', 'first_batch_identity.json',
            'actual_training_epoch_', 'checkpoint_epoch_', 'metrics_epoch_',
        ):
            self.assertIn(expression, self.transformed)

    def test_actual_proxy_optimizer_is_a1(self):
        tree = ast.parse(self.transformed)
        optimizers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == "torch.optim.AdamW"]
        self.assertEqual(len(optimizers), 1)
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in optimizers[0].keywords},
                         {"lr": .005, "betas": (.9, .999), "eps": 1e-4, "weight_decay": .01})
        schedulers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == "CosineAnnealingLR"]
        self.assertEqual(len(schedulers), 2)
        proxy_schedule = next(node for node in schedulers if ast.unparse(node.args[0]) == "proxy_optimizer")
        self.assertEqual(ast.unparse(next(kw.value for kw in proxy_schedule.keywords if kw.arg == "T_max")),
                         "args.epochs")
        self.assertEqual(ast.literal_eval(next(kw.value for kw in proxy_schedule.keywords if kw.arg == "eta_min")), .0005)

    def test_actual_sampler_is_not_cycle(self):
        tree = ast.parse(self.transformed)
        construction = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "sampler" for target in node.targets)
                        and isinstance(node.value, ast.Call)]
        self.assertEqual(len(construction), 1)
        self.assertEqual(ast.unparse(construction[0].func), "SourceClassBatchSampler")

    def test_no_per_step_main_rng_reseed(self):
        tree = ast.parse(self.transformed)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        epoch_loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                          and ast.unparse(node.target) == "epoch_index")
        forbidden = {"seed_all", "random.seed", "np.random.seed", "torch.manual_seed", "torch.cuda.manual_seed"}
        actual = {ast.unparse(node.func) for node in ast.walk(epoch_loop) if isinstance(node, ast.Call)}
        self.assertFalse(forbidden.intersection(actual))


class ShellGeometryContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4780)

    def test_each_actual_shell_mapping_has_one_gs_and_exact_angular_jacobian(self):
        for arm, geometry in (("native", native_geometry), ("b05", b05_geometry)):
            with self.subTest(arm=arm):
                source = torch.tensor([[.2, -.35, .4], [.1, .2, -.6]], dtype=torch.float64, requires_grad=True)
                probe = torch.tensor([[.8, -.1, .4], [-.5, .9, .2]], dtype=torch.float64)
                shell = geometry.whole_to_hier(source)
                gradient, = torch.autograd.grad((shell * probe).sum(), source)
                radius = source.detach().norm(dim=-1, keepdim=True)
                direction = source.detach() / radius
                expected = geometry.HIER_SHELL_GS * geometry.HIER_RADIUS / radius * (
                    probe - direction * (direction * probe).sum(-1, keepdim=True))
                torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close((gradient * direction).sum(-1), torch.zeros(2, dtype=torch.float64),
                                           rtol=0, atol=1e-13)
                self.assertEqual(count_gs_nodes(shell), 1)
                torch.testing.assert_close(geometry.hier_depth(shell), torch.full((2,), 4.6, dtype=torch.float64))
                geometry.assert_hier_shell(shell)

    def test_shell_results_are_identical_between_curvature_arms_for_same_direction(self):
        direction = F.normalize(torch.randn(12, 256, dtype=torch.float64), dim=-1)
        native = direction * torch.linspace(.03, .97, 12, dtype=torch.float64)[:, None]
        b05 = direction * torch.linspace(.06, 1.96, 12, dtype=torch.float64)[:, None]
        a = native_geometry.whole_to_hier(native)
        b = b05_geometry.whole_to_hier(b05)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(F.normalize(a, dim=-1), direction, rtol=1e-12, atol=1e-12)

    def test_invalid_shell_direction_fails_without_detach_or_replacement(self):
        for geometry in (native_geometry, b05_geometry):
            for source in (torch.zeros(1, 3), torch.tensor([[float("nan"), .1, .2]]),
                           torch.tensor([[1e-10, 0., 0.]])):
                with self.subTest(geometry=geometry.__name__), self.assertRaises(ValueError):
                    geometry.whole_to_hier(source)
            with self.assertRaises(TypeError):
                geometry.whole_to_hier(torch.ones(1, 3, dtype=torch.float16))


class ReplacementSamplerContracts(unittest.TestCase):
    @staticmethod
    def sampler(rank=0, steps=200):
        return SourceClassBatchSampler([label for label in range(40) for _ in range(3)],
                                       rank=rank, steps=steps, seed=4780)

    def test_global64_rank_disjoint_and_flip_safe_for_multiple_epochs(self):
        left, right = self.sampler(0), self.sampler(1)
        for epoch in (0, 1, 299):
            left.set_epoch(epoch)
            right.set_epoch(epoch)
            for step in (0, 1, 17, 199):
                a, b = left.global_plan(step), right.global_plan(step)
                self.assertEqual(a, b)
                self.assertEqual(len(a.indices), 64)
                self.assertEqual(len(set(a.classes)), 32)
                self.assertEqual(len(a.rank_indices(0)), 32)
                self.assertFalse(set(a.rank_classes(0)).intersection(a.rank_classes(1)))
                self.assertTrue(all(x != y for x, y in zip(a.labels, reversed(a.labels))))

    def test_within_class_replacement_really_allows_duplicate_objects(self):
        sampler = self.sampler()
        duplicates = sum(plan.indices[index] == plan.indices[index + 1]
                         for plan in sampler.iter_global_plans()
                         for index in range(0, 64, 2))
        self.assertGreater(duplicates, 0)

    def test_sampler_resume_reproduces_plans_and_rejects_wrong_identity(self):
        sampler = self.sampler()
        sampler.set_epoch(17)
        state = sampler.state_dict()
        restored = self.sampler()
        restored.load_state_dict(copy.deepcopy(state))
        self.assertEqual(list(sampler), list(restored))
        bad = dict(state, seed=22)
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)

    def test_sampler_private_queries_leave_global_python_rng_untouched(self):
        random.seed(99)
        before = random.getstate()
        list(self.sampler())
        self.assertEqual(random.getstate(), before)


class RuntimeRoutingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespaces = {arm: namespace(arm) for arm in ("native", "b05")}
        torch.set_num_threads(2)

    def test_runtime_configuration_matches_approved_pair(self):
        for arm, scope in self.namespaces.items():
            with self.subTest(arm=arm):
                args = SimpleNamespace(arm=arm, hier="on", seed=4780, epochs=300, steps_per_epoch=200,
                                       workers=8, eval_batch_size=16, train_eval_every=10,
                                       save_every=20, audit_epochs=(1, 5, 20, 40, 300), smoke=False,
                                       smoke_eval_batches=1, resume=None)
                config = scope["fixed_configuration"](args)
                self.assertEqual((config["seed"], config["epochs"], config["steps_per_epoch"]), (4780, 300, 200))
                self.assertEqual((config["world_size"], config["global_batch"], config["P"]), (2, 64, 512))
                self.assertEqual((config["hier_c"], config["hier_view_depth"], config["HIER_backward_hook"]), (.1, 4.6, True))
                expected = (1., None, False, 1000., 4.) if arm == "native" else (.1, 2.3, True, 500., 2.)
                actual = tuple(config[key] for key in ("native_c", "native_extra_cap", "native_GS", "radial_gamma", "contrastive_margin"))
                self.assertEqual(actual, expected)
                self.assertEqual((config["proxy_optimizer"]["lr"], config["proxy_optimizer"]["eps"],
                                  config["proxy_optimizer"]["min_lr"]), (.005, 1e-4, .0005))
                self.assertIs(scope["SourceClassBatchSampler"], SourceClassBatchSampler)

    def test_cli_enforces_fresh_seed4780_approved_budget_and_arm(self):
        import contextlib
        import io
        for arm, scope in self.namespaces.items():
            arguments = ["--arm", arm, "--hier", "on", "--data-dir", "not-read", "--run-dir", "not-created"]
            args = scope["arguments"](arguments)
            self.assertEqual((args.seed, args.epochs, args.steps_per_epoch), (4780, 300, 200))
            for extra in (["--seed", "22"], ["--steps-per-epoch", "300"], ["--epochs", "10"],
                          ["--hier", "off"]):
                with self.subTest(arm=arm, extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        scope["arguments"](arguments + extra)
            with self.assertRaises(RuntimeError):
                scope["arguments"](arguments + ["--resume", "old.pth"])
            with self.assertRaises(RuntimeError):
                scope["arguments"](arguments + ["--arm", "b05" if arm == "native" else "native"])

    def test_real_saved_optimizer_and_scheduler_are_accepted_and_wrong_eps_rejected(self):
        runner = get_runner()
        model = nn.Parameter(torch.tensor([.1, .2]))
        proxy = nn.Parameter(torch.tensor([2.1, -.1]))
        opt = geoopt.optim.RiemannianSGD([model], lr=.1, momentum=.9, weight_decay=2e-4)
        popt = torch.optim.AdamW([proxy], lr=.005, betas=(.9, .999), eps=1e-4, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=.005)
        pschedule = torch.optim.lr_scheduler.CosineAnnealingLR(popt, T_max=300, eta_min=.0005)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            popt.zero_grad(set_to_none=True)
            (model.square().sum() + proxy.square().sum()).backward()
            opt.step()
            popt.step()
            schedule.step()
            pschedule.step()
        payload = {"format": runner.VERSION, "completed_epochs": 2,
                   "optimizer": opt.state_dict(), "proxy_optimizer": popt.state_dict(),
                   "scheduler": schedule.state_dict(), "proxy_scheduler": pschedule.state_dict()}
        accepted = runner.verify_optimizer_payload(payload, 2, 300)
        self.assertEqual((accepted["initial_lr"], accepted["eps"], accepted["T_max"]), (.005, 1e-4, 300))
        altered = copy.deepcopy(payload)
        altered["proxy_optimizer"]["param_groups"][0]["eps"] = 1e-8
        with self.assertRaises(RuntimeError):
            runner.verify_optimizer_payload(altered, 2, 300)

    def test_actual_backbone_initialization_same_before_geometry_and_configuration_consumes_no_rng(self):
        scopes = self.namespaces
        torch.manual_seed(4780)
        a = scopes["native"]["Hype_pointMLP"]()
        torch.manual_seed(4780)
        b = scopes["b05"]["Hype_pointMLP"]()
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        trainable_before = {name: value.detach().clone() for name, value in a.named_parameters()
                            if not name.endswith("isp_c")}
        for arm, model in (("native", a), ("b05", b)):
            rng = torch.get_rng_state().clone()
            scopes[arm]["configure_backbone"](model)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for name, value in model.named_parameters():
                if name in trainable_before:
                    torch.testing.assert_close(value, trainable_before[name], rtol=0, atol=0)
        self.assertAlmostEqual(float(a.manifold.c), 1., places=5)
        self.assertAlmostEqual(float(b.manifold.c), .1, places=5)
        self.assertEqual(len(a.emb._forward_hooks), 0)
        self.assertEqual(len(b.emb._forward_hooks), 1)

    def test_actual_forward_routes_hier_before_b05_base_gs(self):
        if geoopt is None:
            self.skipTest("Local Geoopt unavailable; this gate must run on server")

        class SourceBall(geoopt.PoincareBall):
            def __init__(self, *, c, dim):
                super().__init__(c=c)
                self.dim = dim

        class Embedding(nn.Module):
            def __init__(self, ball):
                super().__init__()
                self.manifold = ball
                self.weight = nn.Parameter(torch.eye(3) * .7)

            def forward(self, points):
                return self.manifold.expmap0(F.linear(self.manifold.logmap0(points), self.weight))

        class Classifier(nn.Linear):
            def __init__(self, ball):
                super().__init__(3, 40)
                self.manifold = ball

            def forward(self, points):
                return super().forward(self.manifold.logmap0(points))

        class ProbeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.manifold = SourceBall(c=1., dim=512)
                self.manifold2 = SourceBall(c=1., dim=256)
                self.emb = Embedding(self.manifold)
                self.classifier = Classifier(self.manifold2)
                self.bn = nn.BatchNorm1d(3)

            def forward(self, points, emb=False):
                features = self.bn(points.mean(-1))
                mu = self.emb(self.manifold.expmap0(features))
                return mu, mu if emb else self.classifier(mu)

        cloud = torch.randn(64, 24, 3, dtype=torch.float64)
        gold = torch.arange(32).repeat_interleave(2)
        for arm, scope in self.namespaces.items():
            with self.subTest(arm=arm):
                torch.manual_seed(4780)
                backbone = scope["configure_backbone"](ProbeBackbone().double())
                wrapper = scope["HyCoReTrainingForward"](backbone, hier_enabled=True)
                output = wrapper(cloud.clone(), 20, 12)
                values = scope["global_base_losses"](output, gold)
                source = values["hier_source_mu"]
                self.assertEqual(count_gs_nodes(values["mu"]), int(arm == "b05"))
                self.assertEqual(count_gs_nodes(source), 0)
                self.assertEqual(count_gs_nodes(scope["whole_to_hier"](source)), 1)
                torch.testing.assert_close(source, values["mu"], rtol=0, atol=0)
                self.assertTrue(source.requires_grad)
                self.assertEqual(int(backbone.bn.num_batches_tracked), 2)
                self.assertTrue(all(output["input_stats"][key] for key in
                                    ("child_aliases_whole", "whole_aliases_source", "whole_changed_by_child")))
                if arm == "b05":
                    self.assertIsNone(backbone.v9c_geometry.pre_gs)
                    self.assertFalse(backbone.v9c_geometry.capture_enabled)

    def test_private_rng_streams_match_between_arms_and_resume_exactly(self):
        a = self.namespaces["native"]["make_generators"](4780, 0, "cpu", True)
        b = self.namespaces["b05"]["make_generators"](4780, 0, "cpu", True)
        for key in ("loader", "crop", "gumbel"):
            torch.testing.assert_close(torch.rand(20, generator=a[key]), torch.rand(20, generator=b[key]),
                                       rtol=0, atol=0)
        scope = self.namespaces["native"]
        state = scope["capture_generators"](a)
        expected = {key: torch.rand(20, generator=generator) for key, generator in a.items()}
        scope["restore_generators"](a, state)
        for key, generator in a.items():
            torch.testing.assert_close(torch.rand(20, generator=generator), expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
