"""Source, true AdamW and saved-state contracts for the full V9 successor."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import runner


class FullV9OptimizerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = runner.ORIGINAL.read_text(encoding="utf-8").replace("\r\n", "\n")
        cls.source = runner.transformed_source(cls.original)
        cls.tree = ast.parse(cls.source)
        cls.main = next(node for node in cls.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_exact_source_delta_has_only_optimizer_version_and_wrapper_hooks(self):
        restored = self.source
        for old, new in reversed(runner.REPLACEMENTS):
            self.assertEqual(restored.count(new), 1)
            restored = restored.replace(new, old, 1)
        self.assertEqual(restored, self.original)

    def test_changed_original_source_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "reviewed SHA"):
            runner.transformed_source(self.original + "\n# changed\n")

    def test_actual_adamw_values(self):
        assignment = [node for node in ast.walk(self.main) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "proxy_optimizer" for t in node.targets)]
        self.assertEqual(len(assignment), 1)
        call = assignment[0].value
        self.assertEqual(ast.unparse(call.func), "torch.optim.AdamW")
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords},
            {"lr": .005, "betas": (.9, .999), "eps": 1e-4, "weight_decay": .01})

    def configuration(self):
        function = next(n for n in ast.parse(self.original).body if isinstance(n, ast.FunctionDef) and n.name == "fixed_configuration")
        isolated = {}
        exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]), "<config-only>", "exec"), isolated)
        args = SimpleNamespace(seed=22, epochs=300, steps_per_epoch=200, train_eval_every=10,
            eval_batch_size=32, workers=2, audit_epochs=(1,5,20,40,99,100,160,200,240,300), smoke=False)
        return args, isolated["fixed_configuration"](args)

    def test_manifest_delta_is_only_two_optimizer_values_and_identity(self):
        args, original = self.configuration()
        modified = runner.successor_configuration(args, original)
        self.assertEqual(modified["proxy_optimizer"], dict(original["proxy_optimizer"], lr=.005, eps=1e-4))
        for key in original:
            if key not in ("proxy_optimizer", "display_name"):
                self.assertEqual(modified[key], original[key], key)
        self.assertEqual(modified["epochs"], 300)
        self.assertEqual(modified["proxy_optimizer"]["min_lr"], .0005)
        self.assertEqual(modified["training_source"]["original_normalized_sha256"], runner.EXPECTED_SOURCE)

    def test_production_evaluation_and_training_loop_are_retained(self):
        for original in (
            "for epoch_index in range(start_epoch, args.epochs):",
            'selection_points, selection_labels, selection_shards = load_shards(args.data_dir, "test")',
            'selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)',
            'candidate = select_best(best, selection, selection_prefix, epoch)',
            'local_out = model(cloud, whole_count, child_count)',
            'loss = values["base"] + lambda_hier * hier_loss',
            'audit = audit_parameter_gradients(model, proxy, cloud, local_gold, local_ids,',
            'if epoch % args.save_every == 0 or epoch in (1, 5, 99):',
            'f"actual_training_epoch_{epoch:03d}.pth"',
            'proxy_scheduler = CosineAnnealingLR(proxy_optimizer, T_max=args.epochs, eta_min=.0005)',
        ):
            self.assertEqual(self.source.count(original), self.original.count(original), original)
            self.assertIn(original, self.source)
        self.assertNotIn("observer", self.source)
        self.assertNotIn("v9_lr_eps_20261022", Path(runner.__file__).read_text(encoding="utf-8"))

    def test_true_optimizer_step_and_checkpoint_acceptance(self):
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        parameter = torch.nn.Parameter(torch.tensor([[2.1, -.1], [.4, .5]], dtype=torch.float32))
        proxy = torch.nn.ParameterList([parameter])
        namespace = {"torch": torch, "proxy": proxy}
        exec(runner.OPTIMIZER_NEW, namespace)
        optimizer = namespace["proxy_optimizer"]
        scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=.0005)
        for epoch in range(2):
            optimizer.zero_grad(set_to_none=True)
            parameter.square().sum().backward()
            optimizer.step()
            scheduler.step()
        payload = {"format": runner.VERSION, "completed_epochs": 2,
            "proxy_optimizer": optimizer.state_dict(), "proxy_scheduler": scheduler.state_dict()}
        accepted = runner.verify_optimizer_payload(payload, 2, 300)
        self.assertEqual((accepted["initial_lr"], accepted["eps"], accepted["T_max"]), (.005, 1e-4, 300))
        altered = copy.deepcopy(payload)
        altered["proxy_optimizer"]["param_groups"][0]["eps"] = 1e-8
        with self.assertRaises(RuntimeError):
            runner.verify_optimizer_payload(altered, 2, 300)
        with self.assertRaises(RuntimeError):
            runner.verify_optimizer_payload(payload, 2, 2)

    def test_queue_retains_idle_locks_and_uses_only_fresh_successor(self):
        import launch_queue
        original = launch_queue.ORIGINAL.read_text(encoding="utf-8")
        source = launch_queue.transformed_queue_source(original)
        for text in ("safe_query(state, path)", "fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "time.sleep(10)", "idle_after", "production = args.root / 'main'",
            "Unknown resources are never treated as idle", "native_initial_identity(snapshot, env, args.root)"):
            self.assertIn(text, source)
        self.assertIn("str(snapshot / SUCCESSOR_RELATIVE / 'runner.py')", source)
        self.assertIn("verify_successor_smoke(smoke, manifest)", source)
        self.assertIn("verify_successor_completion(production)", source)
        self.assertIn("'--smoke', '--skip-final-test'", source)
        self.assertNotIn("--resume", source)

    def test_queue_rejects_changed_original(self):
        import launch_queue
        with self.assertRaisesRegex(RuntimeError, "reviewed source SHA"):
            launch_queue.transformed_queue_source(launch_queue.ORIGINAL.read_text(encoding="utf-8") + "\n# changed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
