"""CPU-only source/CLI contracts. No training dependencies or data are opened."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import runner


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = runner.ORIGINAL.read_text(encoding="utf-8")
        cls.source = runner.transformed_source(cls.original)
        cls.tree = ast.parse(cls.source)
        cls.main = next(node for node in cls.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_main_never_loads_test(self):
        loads = [node for node in ast.walk(self.main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "load_shards"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0].args[1].value, "train")

    def test_no_evaluation_or_selection_call(self):
        names = [node.func.id for node in ast.walk(self.main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)]
        for name in ("evaluate_clean", "rank0_evaluation", "select_best"):
            self.assertNotIn(name, names)

    def test_original_scheduler_horizon_and_bounded_loop(self):
        self.assertEqual(self.source.count("T_max=args.epochs"), 2)
        self.assertIn("range(start_epoch, args.diagnostic_epochs)", self.source)
        a = runner.diagnostic_arguments(["--data-dir", "train", "--run-dir", "new"])
        self.assertEqual((a.seed, a.epochs, a.steps_per_epoch, a.diagnostic_epochs), (22, 300, 200, 10))
        self.assertEqual(a.audit_epochs, ())
        self.assertIsNone(a.resume)

    def test_smoke_keeps_original_cosine_horizon(self):
        a = runner.diagnostic_arguments(["--data-dir", "train", "--run-dir", "new", "--smoke",
                                         "--diagnostic-epochs", "2", "--steps-per-epoch", "2"])
        self.assertEqual((a.epochs, a.diagnostic_epochs, a.steps_per_epoch), (300, 2, 2))

    def test_no_score_or_selection_claim(self):
        self.assertIn('selection, clean_train, improved, best = None, None, False, None', self.source)
        a = runner.diagnostic_arguments(["--data-dir", "train", "--run-dir", "new"])
        config = runner.diagnostic_configuration(a, {"epochs": 300})
        self.assertEqual(config["selection"], "none")
        self.assertFalse(config["test_read"])
        self.assertFalse(config["test_used_for_selection"])

    def test_bad_bounds_fail_closed(self):
        base = ["--data-dir", "train", "--run-dir", "new"]
        for extra in (["--diagnostic-epochs", "0"], ["--diagnostic-epochs", "11"],
                      ["--steps-per-epoch", "2"], ["--workers", "3"],
                      ["--smoke", "--diagnostic-epochs", "3", "--steps-per-epoch", "2"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                runner.diagnostic_arguments(base + extra)

    def test_unchanged_update_operations(self):
        for operation in (
            'local_out = model(cloud, whole_count, child_count)',
            'values = global_base_losses(local_out, local_gold)',
            'loss = values["base"] + lambda_hier * hier_loss',
            'loss.backward()', 'sync_parameter_gradients(proxy.parameters())',
            'torch.nn.utils.clip_grad_norm_(model_parameters, 1.)',
            'optimizer.step()', 'proxy_optimizer.step()',
            'step_seed = args.seed + epoch_index * 10000019 + step * 1009 + rank * 1000003',
            'relation_seed = args.seed + epoch_index * 10000019 + step * 10007',
            'generator = torch.Generator(device=device).manual_seed(relation_seed)',
        ):
            self.assertEqual(self.source.count(operation), self.original.count(operation))

    def test_source_change_breaks_contract(self):
        with self.assertRaises(RuntimeError):
            runner.transformed_source(self.original.replace('    args = arguments()\n', '    args = altered()\n'))
        with self.assertRaises(RuntimeError):
            runner.transformed_source(self.original + '\n    args = arguments()\n')

    def test_training_main_not_invoked_on_compile(self):
        self.assertNotIn('if __name__ == "__main__":', self.source)
        self.assertEqual(runner.REPO, Path(__file__).resolve().parents[4])


if __name__ == "__main__":
    unittest.main()
