"""Independent CPU source contracts for the two-value V9 optimizer change."""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import tempfile
import unittest

import runner


class OptimizerSuccessorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = runner.ORIGINAL.read_text(encoding="utf-8")
        cls.parent = runner.BASE_TRANSFORM(cls.original)
        cls.source = runner.transformed_source(cls.original)
        cls.tree = ast.parse(cls.source)
        cls.main = next(node for node in cls.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_exact_generated_source_delta(self):
        recovered = self.source.replace(runner.OPTIMIZER_NEW, runner.OPTIMIZER_OLD, 1)
        recovered = recovered.replace(runner.VERSION_NEW, runner.VERSION_OLD, 1)
        self.assertEqual(recovered, self.parent)
        self.assertEqual(self.source.count(runner.OPTIMIZER_NEW), 1)
        self.assertNotIn(runner.OPTIMIZER_OLD, self.source)

    def test_actual_optimizer_arguments(self):
        assignments = [node for node in ast.walk(self.main) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "proxy_optimizer" for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        call = assignments[0].value
        self.assertEqual(ast.unparse(call.func), "torch.optim.AdamW")
        values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        self.assertEqual(values, {"lr": .005, "betas": (.9, .999), "eps": 1e-4, "weight_decay": .01})

    def test_manifest_matches_actual_optimizer(self):
        args = runner.diagnostic_arguments(["--data-dir", "unused", "--run-dir", "unused"])
        original_tree = ast.parse(self.original)
        function = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "fixed_configuration")
        isolated = {}
        exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]), "<configuration-only>", "exec"), isolated)
        original = isolated["fixed_configuration"](args)
        parent = runner.BASE_CONFIGURATION(args, copy.deepcopy(original))
        successor = runner.diagnostic_configuration(args, copy.deepcopy(original))
        expected_optimizer = dict(parent["proxy_optimizer"], lr=.005, eps=1e-4)
        self.assertEqual(successor["proxy_optimizer"], expected_optimizer)
        for key in parent:
            if key not in ("display_name", "proxy_optimizer"):
                self.assertEqual(successor[key], parent[key], key)

    def test_no_evaluation_and_only_train_shards(self):
        calls = [node for node in ast.walk(self.main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertFalse({"evaluate_clean", "rank0_evaluation", "select_best"} & {c.func.id for c in calls})
        loads = [c for c in calls if c.func.id == "load_shards"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(ast.literal_eval(loads[0].args[1]), "train")
        self.assertIn('selection, clean_train, improved, best = None, None, False, None', self.source)

    def test_original_cosine_horizon_and_minimum(self):
        args = runner.diagnostic_arguments(["--data-dir", "unused", "--run-dir", "unused"])
        self.assertEqual((args.seed, args.epochs, args.diagnostic_epochs, args.steps_per_epoch), (22, 300, 10, 200))
        self.assertIn("range(start_epoch, args.diagnostic_epochs)", self.source)
        self.assertEqual(self.source.count("T_max=args.epochs"), 2)
        self.assertIn("proxy_scheduler = CosineAnnealingLR(proxy_optimizer, T_max=args.epochs, eta_min=.0005)", self.source)
        lr_after10 = .0005 + (.005-.0005) * (1 + math.cos(math.pi * 10/300)) / 2
        self.assertGreater(lr_after10, .00498)

    def test_sampling_and_same_graph_observer_preserved(self):
        for text in (
            "shared_crop_counts(args.seed, epoch_index, step)",
            "local_out = model(cloud, whole_count, child_count)",
            "values = global_base_losses(local_out, local_gold)",
            'loss = values["base"] + lambda_hier * hier_loss',
            "sync_parameter_gradients(proxy.parameters())",
            "observer.before_step(proxy, proxy_optimizer)",
            "observer.after_step(proxy_observation, proxy, proxy_optimizer)",
            "initial_hash != DIAGNOSTIC_EXPECTED_INITIAL_MODEL",
        ):
            self.assertEqual(self.source.count(text), self.parent.count(text), text)
        self.assertEqual(runner.EXPECTED_INITIAL_MODEL, runner.base.EXPECTED_INITIAL_MODEL)
        self.assertEqual(runner.SELECTED_STEPS, (1, 2, 5, 10, 20, 50, 100, 200))

    def test_smoke_is_fresh_and_keeps_schedule(self):
        args = runner.diagnostic_arguments(["--data-dir", "unused", "--run-dir", "unused", "--smoke",
            "--diagnostic-epochs", "2", "--steps-per-epoch", "2"])
        self.assertEqual((args.epochs, args.diagnostic_epochs, args.steps_per_epoch), (300, 2, 2))
        self.assertIsNone(args.resume)
        self.assertTrue(args.skip_final_test)

    def test_source_mutation_fails_closed(self):
        with self.assertRaises(RuntimeError):
            runner.transformed_source(self.original.replace("lr=.01, betas=", "lr=.02, betas="))

    def test_repository_resolution(self):
        self.assertEqual(runner.REPO, Path(__file__).resolve().parents[4])
        self.assertNotIn('if __name__ == "__main__":', self.source)

    def test_saved_optimizer_acceptance_uses_actual_adam_and_schedule(self):
        import torch
        queue_tree = ast.parse((runner.HERE / "launch_queue.py").read_text(encoding="utf-8"))
        function = next(node for node in queue_tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_saved_optimizer")
        namespace = {"math": math}
        exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]), "<saved-optimizer-acceptance>", "exec"), namespace)
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=.005, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=.0005)
        with tempfile.TemporaryDirectory(dir=runner.HERE, prefix=".optimizer_acceptance_") as temporary:
            directory = Path(temporary)
            for epoch in range(1, 11):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                scheduler.step()
                if epoch not in (2, 10):
                    continue
                payload = {"proxy_optimizer": optimizer.state_dict(), "proxy_scheduler": scheduler.state_dict(), "completed_epochs": epoch}
                torch.save(payload, directory / "last.pth")
                accepted = namespace["verify_saved_optimizer"](directory, epoch)
                self.assertEqual(accepted["initial_lr"], .005)
                payload["proxy_optimizer"]["param_groups"][0]["eps"] = 1e-8
                torch.save(payload, directory / "last.pth")
                with self.assertRaises(RuntimeError):
                    namespace["verify_saved_optimizer"](directory, epoch)


if __name__ == "__main__":
    unittest.main()
