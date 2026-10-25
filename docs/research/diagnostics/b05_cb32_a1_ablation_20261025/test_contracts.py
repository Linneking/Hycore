"""Verify each executable change against the preserved 10/25 CB32 A1 baseline."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO), str(REPO / "inter_hierarchy_MN40")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("ablation_contract_runner", HERE / "runner.py")
baseline = load("preserved_cb32_runner", HERE.parent / "b05_singlecard_a0_a1_20261025/runner.py")


class AblationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = runner.ORIGINAL.read_text(encoding="utf-8")

    def test_executable_diff_is_only_requested_K_or_weight(self):
        reference = baseline.transformed_source(self.original, "a1")
        reference = reference.replace(baseline.VERSION, runner.VERSION)
        reference = reference.replace('choices=("a0", "a1")', 'choices=("k5p10", "w01")')
        for variant in ("k5p10", "w01"):
            actual = runner.transformed_source(self.original, variant)
            if variant == "k5p10":
                actual = actual.replace('topk=5, proxy_topk=10', 'topk=10, proxy_topk=20')
                actual = actual.replace('whole_count, child_count, topk=5, seed=',
                                        'whole_count, child_count, topk=10, seed=')
            else:
                actual = actual.replace('"lambda_hier_after_warmup": .1 if on else 0.',
                                        '"lambda_hier_after_warmup": .5 if on else 0.')
                actual = actual.replace('lambda_hier=.1)', 'lambda_hier=.5)')
                actual = actual.replace('loss = values["base"] + .1 * hier_loss',
                                        'loss = values["base"] + .5 * hier_loss')
                actual = actual.replace('"lambda_hier": .1 if enabled else 0.',
                                        '"lambda_hier": .5 if enabled else 0.')
            self.assertEqual(ast.dump(ast.parse(reference)), ast.dump(ast.parse(actual)))

    def test_configuration_and_actual_loss_calls_agree(self):
        for variant, ks, kp, weight in (("k5p10", 5, 10, .5), ("w01", 10, 20, .1)):
            tree = ast.parse(runner.transformed_source(self.original, variant))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in ("arguments", "parent_configuration")]
            import argparse
            scope = dict(argparse=argparse, Path=Path, __doc__="test", TRAIN_SHARD_ORDER=())
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<isolated>", "exec"), scope)
            args = scope["arguments"](["--variant", variant, "--hier", "on", "--run-dir", "unused", "--data-dir", "unused"])
            config = runner.successor_configuration(args, scope["parent_configuration"](args))
            self.assertEqual((config["sample_K"], config["proxy_K"], config["lambda_hier_after_warmup"]), (ks, kp, weight))
            self.assertEqual((args.seed, args.epochs, args.steps_per_epoch), (4780, 300, 200))
            self.assertEqual((config["global_batch"], config["world_size"], config["base_c"], config["native_GS"]), (32, 1, .1, True))
            self.assertEqual((config["proxy_optimizer"]["lr"], config["proxy_optimizer"]["eps"]), (.005, 1e-4))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "proxy"]
            self.assertEqual(len(calls), 1)
            values = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg in ("topk", "proxy_topk")}
            self.assertEqual(values, dict(topk=ks, proxy_topk=kp))
            audit = runner.transformed_audit_source(runner.AUDIT_SOURCE.read_text(encoding="utf-8"), variant)
            self.assertIn(f'proxy_topk={kp}', audit)

    def test_live_geometry_gradients_and_weight_scaling(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier
        torch.manual_seed(18)
        points = to_ball(torch.randn(32, 8) * .35, hook=False).detach().requires_grad_()
        shell = whole_to_hier(points)
        labels = torch.arange(16).repeat_interleave(2)
        proxy = HIERLoss(num_proxies=512, dim=8, seed=4780)
        for variant in ("k5p10", "w01"):
            arm = runner.settings(variant)
            loss, stats = proxy(shell, labels, topk=arm["sample_K"], proxy_topk=arm["proxy_K"],
                                t_per_anchor=50, seed=4780, generator=torch.Generator().manual_seed(99), monitor_structure=False)
            self.assertEqual(stats["sample"]["topk_including_self"], arm["sample_K"])
            self.assertEqual(stats["proxy"]["topk_including_self"], arm["proxy_K"])
            for graph in ("sample", "proxy"):
                self.assertGreater(stats[graph]["triplets"], 0)
                self.assertEqual(stats[graph]["self_k_triplets"], 0)
            gradients = torch.autograd.grad(arm["lambda_hier"] * loss, (points, proxy.tangent_proxies), retain_graph=True)
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 0)
            if variant == "w01":
                reference = torch.autograd.grad(.5 * loss, (points, proxy.tangent_proxies), retain_graph=True)
                for actual, old in zip(gradients, reference):
                    torch.testing.assert_close(actual, .2 * old, rtol=2e-4, atol=1e-6)

    def test_parent_hash_guards_and_fresh_cli(self):
        with self.assertRaises(RuntimeError):
            runner.transformed_source(self.original + "#changed")
        with self.assertRaises(RuntimeError):
            runner.transformed_audit_source(runner.AUDIT_SOURCE.read_text(encoding="utf-8") + "#changed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
