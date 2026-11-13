import ast
import unittest

import runner


class B13Contract(unittest.TestCase):
    def test_weights_k_and_recipe_reach_live_training_and_audit(self):
        raw = runner.parent.ORIGINAL.read_text(encoding="utf-8")
        for variant, weight, lr, eps in (("ts33", .5, .01, 1e-8), ("ts34", 1., .005, 1e-4)):
            with self.subTest(variant=variant):
                source = runner.transformed_source(raw, variant)
                ast.parse(source)
                self.assertIn(f'values["base"] + {weight!r} * hier_loss', source)
                self.assertIn(f'topk=20, seed=args.seed + epoch_index * 10000019, lambda_hier={weight!r}', source)
                self.assertIn('proxy(hier_mu, values["gold"], topk=20, proxy_topk=20', source)
                self.assertIn(f'lr={lr!r}, betas=(.9, .999), eps={eps!r}', source)


if __name__ == "__main__":
    unittest.main()
