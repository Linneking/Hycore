"""CPU checks for the new author's argument and proxy seed wiring."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from worker import SEED, HierAddition, author, author_args, support


class SeedContracts(unittest.TestCase):
    def test_explicit_seed_reaches_author_parser(self):
        source = ast.parse(support.source_text())
        parser = next(node for node in source.body
                      if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
        context = {"argparse": __import__("argparse")}
        exec(compile(ast.Module(body=[parser], type_ignores=[]), str(author.SOURCE), "exec"), context)
        for smoke, epochs in ((False, 300), (True, 2)):
            with self.subTest(smoke=smoke), patch.object(sys, "argv", [str(author.SOURCE), *author_args(smoke)]):
                args = context["parse_args"]()
                self.assertEqual((args.seed, args.epoch, args.batch_size, args.num_points, args.workers),
                                 (4780, epochs, 32, 1024, 8))
                self.assertEqual((args.learning_rate, args.weight_decay, args.msg),
                                 (.1, 1e-4, "Offv_pointmlp_hycore"))

    def test_proxy_branch_uses_new_seed(self):
        import torch

        seen_seeds = []
        with TemporaryDirectory() as directory, patch.object(support, "SEED", 12), \
                patch.object(support, "Hierarchy") as make_branch:
            branch = make_branch.return_value
            make_branch.side_effect = lambda *args, **kwargs: seen_seeds.append(support.SEED) or branch
            addon = HierAddition(Path(directory))
            for _ in range(2):
                self.assertIs(addon.loss(1, 2, 3, 4), branch.add_loss.return_value)
            self.assertEqual(seen_seeds, [4780])
            self.assertEqual(support.SEED, SEED)
            make_branch.assert_called_once_with(True, Path(directory) / "hier", torch, epochs=300)
            self.assertIsNone(branch.ids)
            self.assertEqual(branch.add_loss.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
