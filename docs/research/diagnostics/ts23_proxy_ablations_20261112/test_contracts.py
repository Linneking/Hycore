"""Contracts for the exact TS23 base and each isolated proxy change."""
import argparse
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for path in (REPO, REPO / "inter_hierarchy_MN40"):
    sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location("ts23_ablation_test_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Contracts(unittest.TestCase):
    def cli(self, source):
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in ("arguments", "parent_configuration")]
        ns = dict(argparse=argparse, Path=Path, __doc__="TS23 CLI test", TRAIN_SHARD_ORDER=())
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<cli>", "exec"), ns)
        return ns

    def parse(self, namespace, arm, extra=()):
        return namespace["arguments"]([
            "--variant", arm, "--hier", "on", "--data-dir", "unused", "--run-dir", "unused", *extra
        ])

    def test_matches_ts23_training_config_except_the_assigned_proxy_variable(self):
        raw = runner.ORIGINAL.read_text(encoding="utf-8")
        baseline_ns = self.cli(runner.parent.transformed_source(raw, runner.PARENT_VARIANT))
        baseline_args = self.parse(baseline_ns, runner.PARENT_VARIANT)
        baseline = runner.parent.successor_configuration(baseline_args, baseline_ns["parent_configuration"](baseline_args))
        changed_fields = (
            "variant", "canonical_name", "display_name", "implementation_variant",
            "proxy_initialization", "comparison_scope", "training_source", "P", "proxy_tangent_cap",
        )
        for arm, (count, scale, cap) in {
            "p256": (256, .9, 2.3), "init05": (512, .5, 2.3), "cap29": (512, .9, 2.9),
        }.items():
            with self.subTest(arm=arm):
                ns = self.cli(runner.transformed_source(raw, arm))
                args = self.parse(ns, arm)
                config = runner.successor_configuration(args, ns["parent_configuration"](args))
                self.assertEqual((args.seed, args.epochs, args.steps_per_epoch), (4780, 300, 200))
                self.assertEqual(
                    (config["P"], config["proxy_initialization_coefficient"],
                     config["proxy_tangent_cap"], config["proxy_depth_cap"]),
                    (count, scale, cap, 2 * cap),
                )
                for key in baseline.keys() - set(changed_fields):
                    self.assertEqual(baseline[key], config[key], key)
                self.assertEqual((config["native_extra_cap"], config["native_GS"],
                                  config["hier_view_depth"], config["sample_K"], config["proxy_K"]),
                                 (2.3, True, 4.6, 10, 10))

    def test_smoke_is_training_only_and_resume_is_rejected(self):
        ns = self.cli(runner.transformed_source(runner.ORIGINAL.read_text(encoding="utf-8"), "cap29"))
        args = self.parse(ns, "cap29", ["--smoke", "--epochs", "2", "--steps-per-epoch", "2",
                                        "--skip-final-test"])
        self.assertTrue(args.smoke and args.skip_final_test)
        with self.assertRaises(RuntimeError):
            runner.require_fresh_args(self.parse(ns, "cap29", ["--resume", "unused"]), "cap29")

    def test_original_ts23_body_is_reused(self):
        source = runner.ORIGINAL.read_text(encoding="utf-8")
        baseline = runner.parent.transformed_source(source, runner.PARENT_VARIANT)
        transformed = runner.transformed_source(source, "p256")
        restored = transformed.replace(runner.VERSION, runner.parent.VERSION)
        restored = restored.replace(f"choices={tuple(runner.ARMS)!r}", 'choices=("k10p10w01", "k10p20w005")')
        restored = restored.replace(
            'proxy = ABLATION_HIER_CLASS(seed=args.seed).to(device)',
            'from ..hier_proxy_scratch_v8.hier_loss import HIERLoss\n'
            '            proxy = HIERLoss(seed=args.seed).to(device)',
        ).replace('proxy.tangent_proxies.numel() if enabled else 0', '512 * 256 if enabled else 0')
        restored = restored.replace(
            ',\n            "proxy_forward_tangent_cap": config["proxy_tangent_cap"],\n'
            '            "proxy_forward_depth_cap": config["proxy_depth_cap"]', "")
        self.assertEqual(ast.dump(ast.parse(restored)), ast.dump(ast.parse(baseline)))
        with self.assertRaises(RuntimeError):
            runner.transformed_source(source + "\n# altered source", "p256")

    def test_actual_initializers_match_ts23_prefix_and_fixed_scale(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss

        state = torch.get_rng_state().clone()
        baseline = HIERLoss(seed=4780).tangent_proxies
        classes = {arm: runner.proxy_class(arm)(seed=4780) for arm in runner.ARMS}
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(classes["p256"].tangent_proxies, baseline[:256]))
        self.assertTrue(torch.equal(classes["cap29"].tangent_proxies, baseline))
        generator = torch.Generator().manual_seed(4780)
        expected = torch.randn(512, 256, generator=generator) / 16 * 2.3 * .5
        self.assertTrue(torch.equal(classes["init05"].tangent_proxies, expected))

    def test_cap29_geometry_and_metrics_do_not_move_the_sample_shell(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import depth, to_ball

        proxy = runner.proxy_class("cap29")(seed=4780)
        with torch.no_grad():
            proxy.tangent_proxies[0].zero_()
            proxy.tangent_proxies[1].zero_()
            proxy.tangent_proxies[0, 0] = 2.6
            proxy.tangent_proxies[1, 0] = 3.0
        proxy.assert_geometry()
        torch.testing.assert_close(depth(proxy.proxies()[:2]), torch.tensor([5.2, 5.8]), atol=5e-5, rtol=0)
        self.assertEqual(proxy.project_after_step(None)["forward_clip_count"], 1)
        self.assertGreater(runner.cap29_proxy_update_stats(torch.tensor([[2.4]]), torch.tensor([[2.8]]))
                           ["mapped_ball_radius_change_mean"], 0.)
        sample = to_ball(torch.full((2, 256), 2.8 / 16), clip_r=2.9)
        with self.assertRaises(RuntimeError):
            proxy(sample, torch.tensor([0, 1]))

    def test_real_hier_loss_and_backward_accept_all_three_arms(self):
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier

        points = torch.randn(32, 256, generator=torch.Generator().manual_seed(42)) * .05
        points.requires_grad_(True)
        labels = torch.arange(16).repeat_interleave(2)
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                proxy = runner.proxy_class(arm)(seed=4780)
                if arm == "cap29":
                    with torch.no_grad():
                        proxy.tangent_proxies[0].zero_()
                        proxy.tangent_proxies[0, 0] = 3.0
                loss, stats = proxy(whole_to_hier(points), labels, topk=10, proxy_topk=10,
                                    generator=torch.Generator().manual_seed(43))
                gradients = torch.autograd.grad(loss, (points, proxy.tangent_proxies))
                for gradient in gradients:
                    self.assertTrue(bool(gradient.isfinite().all()))
                    self.assertGreater(float(gradient.norm()), 0.)
                count = runner.settings(arm)["P"]
                self.assertEqual(stats["geometry"]["proxy"]["points"], count)
                self.assertEqual(stats["structure"]["sample"]["proxy_count"], count)
                self.assertEqual(stats["structure"]["proxy"]["proxy_count"], count)
                self.assertEqual(stats["sample"]["self_k_triplets"], 0)
                self.assertEqual(stats["proxy"]["self_k_triplets"], 0)
                if arm == "cap29":
                    self.assertEqual(stats["geometry"]["proxy"]["cap_fraction"], 1 / count)
                    self.assertAlmostEqual(stats["geometry"]["whole"]["depth_max"], 4.6, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
