"""CPU gates for approved B05 single-rank K10/20 A0/A1 successors."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO), str(REPO / "inter_hierarchy_MN40")]
spec = importlib.util.spec_from_file_location("singlecard_runner_contract", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier, hier_depth
from inter_hierarchy_MN40.hier_proxy_cycle_v9c.base_protocol import global_base_losses, smoothed_ce, intra_losses


class SingleCardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = runner.ORIGINAL.read_text(encoding="utf-8")
        cls.sources = {v: runner.transformed_source(cls.original, v) for v in ("a0", "a1")}

    def test_pins_reject_changed_parent_and_audit(self):
        with self.assertRaises(RuntimeError):
            runner.transformed_source(self.original + "\n# modified", "a1")
        with self.assertRaises(RuntimeError):
            runner.transformed_audit_source(runner.AUDIT_SOURCE.read_text(encoding="utf-8") + "\n# modified")

    def test_runtime_collectives_sampler_and_k_are_singlecard(self):
        for source in self.sources.values():
            ast.parse(source)
            for expression in ('devices = [None]', 'rank_peaks = [None]', 'rank_states = [None]',
                               '"rank_order": [0]', 'int(os.environ["WORLD_SIZE"]) != 1',
                               'world_size=1, classes_per_rank=16, instances_per_class=2',
                               'topk=10, proxy_topk=20', 'whole_count, child_count, topk=10',
                               'torch.nn.utils.clip_grad_norm_(model_parameters, 1.)'):
                self.assertIn(expression, source)
            self.assertNotIn('[None, None]', source)
            audit = runner.transformed_audit_source(runner.AUDIT_SOURCE.read_text(encoding="utf-8"))
            self.assertIn('topk=topk, proxy_topk=20', audit)

    def test_a0_a1_only_actual_optimizer_arguments_differ(self):
        normalized = []
        for variant, source in self.sources.items():
            tree = ast.parse(source)
            optimizers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and ast.unparse(n.func) == "torch.optim.AdamW"]
            self.assertEqual(len(optimizers), 1)
            values = {kw.arg: ast.literal_eval(kw.value) for kw in optimizers[0].keywords}
            self.assertEqual(values, {**runner.recipe(variant), "betas": (.9, .999), "weight_decay": .01})
            for kw in optimizers[0].keywords:
                if kw.arg in ("lr", "eps"):
                    kw.value = ast.Constant(value=0.)
            normalized.append(ast.dump(tree))
        self.assertEqual(normalized[0], normalized[1])

    def test_config_and_cli_budget_fresh_constraints_without_model_import(self):
        for variant, source in self.sources.items():
            tree = ast.parse(source)
            selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name in ("arguments", "parent_configuration")]
            import argparse
            scope = {"argparse": argparse, "Path": Path, "__doc__": "test",
                     "TRAIN_SHARD_ORDER": ("ply_data_train1.h5", "ply_data_train4.h5", "ply_data_train3.h5", "ply_data_train2.h5", "ply_data_train0.h5")}
            exec(compile(ast.Module(body=selected, type_ignores=[]), "<isolated-cli>", "exec"), scope)
            args = scope["arguments"](["--variant", variant, "--hier", "on", "--data-dir", "unused", "--run-dir", "unused"])
            cfg = runner.successor_configuration(args, scope["parent_configuration"](args))
            self.assertEqual((args.seed, args.epochs, args.steps_per_epoch), (4780, 300, 200))
            self.assertEqual((cfg["global_batch"], cfg["world_size"], cfg["sample_K"], cfg["proxy_K"], cfg["T"], cfg["P"]), (32, 1, 10, 20, 50, 512))
            self.assertEqual((cfg["native_c"], cfg["native_GS"], cfg["native_extra_cap"], cfg["hier_view_depth"]), (.1, True, 2.3, 4.6))

    def test_singlecard_sampler_16classes_two_examples_flip_safe_rng_preserved(self):
        labels = [v for v in range(40) for _ in range(3)]
        sampler = SourceClassBatchSampler(labels, world_size=1, classes_per_rank=16,
                                         instances_per_class=2, steps=200, seed=4780, rank=0)
        random.seed(78)
        before = random.getstate()
        for epoch in (0, 1, 299):
            sampler.set_epoch(epoch)
            plans = list(sampler.iter_global_plans())
            self.assertEqual(len(plans), 200)
            for plan in plans:
                self.assertEqual((len(plan.indices), len(set(plan.classes))), (32, 16))
                self.assertEqual(plan.rank_indices(0), plan.indices)
                self.assertTrue(all(a != b for a, b in zip(plan.labels, reversed(plan.labels))))
        self.assertEqual(before, random.getstate())
        state = sampler.state_dict()
        restored = SourceClassBatchSampler(labels, world_size=1, steps=200, seed=4780)
        restored.load_state_dict(state)
        self.assertEqual(list(restored), list(sampler))

    def test_global32_objective_gradients_match_direct_mean(self):
        torch.manual_seed(20)
        labels = torch.arange(16).repeat_interleave(2)
        mu = (.2 * torch.randn(32, 5, dtype=torch.float64)).requires_grad_()
        nu = (.1 * torch.randn(32, 5, dtype=torch.float64)).requires_grad_()
        logits = torch.randn(32, 40, dtype=torch.float64, requires_grad=True)
        values = global_base_losses({"mu": mu, "nu": nu, "logits": logits,
                                     "whole_count": 800, "child_count": 400,
                                     "hier_source_mu": mu}, labels, expected_global_batch=32)
        intra = intra_losses(mu, nu, 400)
        reference = smoothed_ce(logits, labels) + .01 * (intra["intra_contrastive"] + intra["intra_radial"])
        actual_grad = torch.autograd.grad(values["base"], (mu, nu, logits), retain_graph=True)
        expected_grad = torch.autograd.grad(reference, (mu, nu, logits))
        for a, b in zip(actual_grad, expected_grad):
            # Independent regrouping of the two weighted means differs only
            # at FP64 roundoff; this still rejects any batch-size scaling.
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-15)
        with self.assertRaises(ValueError):
            global_base_losses({"mu": mu, "nu": nu, "logits": logits, "child_count": 400}, labels)

    def test_real_hier_uses_separate_k_and_produces_finite_sample_proxy_gradients(self):
        torch.manual_seed(18)
        points = to_ball(torch.randn(32, 8) * .35, hook=False).detach().requires_grad_()
        shell = whole_to_hier(points)
        labels = torch.arange(16).repeat_interleave(2)
        proxy = HIERLoss(num_proxies=512, dim=8, seed=4780)
        loss, stats = proxy(shell, labels, topk=10, proxy_topk=20, t_per_anchor=50, seed=4780,
                            generator=torch.Generator().manual_seed(99), monitor_structure=False)
        self.assertEqual(stats["sample"]["topk_including_self"], 10)
        self.assertEqual(stats["proxy"]["topk_including_self"], 20)
        self.assertEqual((stats["sample"]["batch_size"], stats["proxy"]["batch_size"]), (32, 512))
        self.assertGreater(stats["sample"]["triplets"], 0)
        self.assertGreater(stats["proxy"]["triplets"], 0)
        self.assertEqual(stats["sample"]["self_k_triplets"], 0)
        self.assertEqual(stats["proxy"]["self_k_triplets"], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(proxy.tangent_proxies.grad).all())
        self.assertGreater(float(points.grad.norm()), 0)
        self.assertGreater(float(proxy.tangent_proxies.grad.norm()), 0)
        torch.testing.assert_close(hier_depth(shell), torch.full((32,), 4.6), rtol=1e-5, atol=1e-5)

    def test_saved_actual_adamw_schedule_accepts_recipe_and_rejects_other(self):
        import geoopt
        for variant in ("a0", "a1"):
            model, proxy = nn.Parameter(torch.tensor([.1, .2])), nn.Parameter(torch.tensor([2.1, -.1]))
            opt = geoopt.optim.RiemannianSGD([model], lr=.1, momentum=.9, weight_decay=2e-4)
            popt = torch.optim.AdamW([proxy], **runner.recipe(variant), betas=(.9, .999), weight_decay=.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=.005)
            psch = torch.optim.lr_scheduler.CosineAnnealingLR(popt, T_max=2, eta_min=.0005)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                popt.zero_grad(set_to_none=True)
                (model.square().sum() + proxy.square().sum()).backward()
                opt.step(); popt.step(); sch.step(); psch.step()
            payload = {"format": runner.VERSION, "completed_epochs": 2,
                       "optimizer": opt.state_dict(), "proxy_optimizer": popt.state_dict(),
                       "scheduler": sch.state_dict(), "proxy_scheduler": psch.state_dict()}
            runner.verify_optimizer_payload(payload, 2, 2, variant)
            with self.assertRaises(RuntimeError):
                runner.verify_optimizer_payload(payload, 2, 2, "a0" if variant == "a1" else "a1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
