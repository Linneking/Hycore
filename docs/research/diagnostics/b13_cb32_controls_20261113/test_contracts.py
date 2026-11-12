"""Validate the configured and executed B13 control boundaries on CPU."""
import argparse
import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_queue
import runner


class Contracts(unittest.TestCase):
    def test_actual_cli_and_configuration(self):
        source = runner.parent.ORIGINAL.read_text(encoding="utf-8")
        for arm, budget in runner.STEPS.items():
            with self.subTest(arm=arm):
                tree = ast.parse(runner.transformed_source(source, arm))
                definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name in ("arguments", "parent_configuration")]
                scope = dict(argparse=argparse, Path=Path, TRAIN_SHARD_ORDER=(1, 4, 3, 2, 0))
                exec(compile(ast.Module(body=definitions, type_ignores=[]), "<test>", "exec"), scope)
                settings = runner.settings(arm)
                cli = ["--variant", arm, "--hier", settings["hier"], "--run-dir", "out", "--data-dir", "data"]
                args = scope["arguments"](cli)
                cfg = runner.configuration(args, scope["parent_configuration"](args))
                enabled = settings["hier"] == "on"
                self.assertEqual((args.seed, args.epochs, args.steps_per_epoch), (4780, 300, budget))
                self.assertEqual((cfg["world_size"], cfg["global_batch"], cfg["classes_per_rank"],
                                  cfg["instances_per_class"]), (1, 32, 16, 2))
                self.assertEqual((cfg["base_c"], cfg["native_extra_cap"], cfg["native_GS"],
                                  cfg["radial_gamma"], cfg["contrastive_margin"]), (.1, 2.3, True, 500., 2.))
                self.assertEqual((cfg["hier_enabled"], cfg["HIER_calls_per_step"], cfg["P"]),
                                 (enabled, int(enabled), 512 if enabled else 0))
                self.assertEqual((cfg["sample_K"], cfg["proxy_K"], cfg["lambda_hier_after_warmup"]),
                                 (settings["sample_K"], settings["proxy_K"], settings["lambda_hier"]))
                if enabled:
                    self.assertEqual((cfg["proxy_optimizer"]["lr"], cfg["proxy_optimizer"]["eps"],
                                      cfg["hier_view_depth"]), (.005, 1e-4, 4.6))
                else:
                    self.assertIsNone(cfg["proxy_optimizer"])
                    self.assertIsNone(cfg["hier_view_depth"])
                smoke = scope["arguments"](cli + ["--smoke", "--skip-final-test", "--epochs", "2",
                                                  "--steps-per-epoch", "2"])
                self.assertTrue(smoke.smoke and smoke.skip_final_test)

    def test_loss_and_audit_both_use_k20(self):
        raw = runner.parent.ORIGINAL.read_text(encoding="utf-8")
        source = runner.transformed_source(raw, "a1_200")
        self.assertIn('proxy(hier_mu, values["gold"], topk=20, proxy_topk=20', source)
        self.assertIn('whole_count, child_count, topk=20, seed=', source)
        self.assertIn('loss = values["base"] + .5 * hier_loss', source)
        audit = runner.parent.transformed_audit_source(runner.parent.AUDIT_SOURCE.read_text(encoding="utf-8"))
        self.assertIn('topk=topk, proxy_topk=20', audit)
        self.assertIn('else:\n                    loss = values["base"]', source)

    def test_launcher_routes_all_three_controls(self):
        source = launch_queue.manager_source()
        compile(source, "<launcher>", "exec")
        self.assertEqual(launch_queue.GPU_BY_VARIANT, {"h0_200": 0, "h0_300": 1, "a1_200": 2})
        self.assertIn('runner.settings(args.variant)["hier"]', source)
        self.assertIn('if smoke and enabled:', source)
        self.assertIn('(300, arm["steps"])', source)
