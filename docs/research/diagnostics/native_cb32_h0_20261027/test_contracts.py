"""CPU source boundaries plus a CB32 loss/gradient comparison to released code."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_cb32_runner', HERE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SourceContracts(unittest.TestCase):
    def test_pinned_parent_and_single_rank_collectives(self):
        source = runner.transformed_source(runner.ORIGINAL.read_text(encoding='utf-8'))
        ast.parse(source)
        for name in ('devices', 'rank_peaks', 'rank_states'):
            self.assertIn(name + ' = [None]', source)
        self.assertNotIn('= [None, None]', source)
        self.assertIn('choices=("off",)', source)
        self.assertIn('world_size=1, classes_per_rank=16, instances_per_class=2', source)

    def test_configuration_is_native_h0(self):
        args = SimpleNamespace(hier='off')
        config = runner.configuration(args, {})
        self.assertEqual((config['base_c'], config['global_batch'], config['native_GS']), (1., 32, False))
        self.assertIsNone(config['native_extra_cap'])
        self.assertEqual((config['radial_gamma'], config['contrastive_margin']), (1000., 4.))

    def test_parent_drift_is_rejected(self):
        with self.assertRaises(RuntimeError):
            runner.transformed_source(runner.ORIGINAL.read_text(encoding='utf-8') + '\n# drift')


class RuntimeContracts(unittest.TestCase):
    def test_cb32_matches_released_loss_and_gradient(self):
        import torch
        from inter_hierarchy_MN40.hier_proxy_scratch_v9 import base_protocol
        from inter_hierarchy_MN40.hier_proxy_scratch_v9.tests.test_base_protocol import released_functions
        reference = released_functions()
        gen = torch.Generator().manual_seed(4780)
        labels = torch.arange(16).repeat_interleave(2)
        for count in (200, 400, 600):
            inputs = [torch.randn(32, d, generator=gen).mul(.02).requires_grad_() for d in (256, 256, 40)]
            mu, nu, logits = inputs
            actual = base_protocol.global_base_losses(dict(mu=mu, nu=nu, logits=logits,
                child_count=count, whole_count=900), labels, expected_global_batch=32)['base']
            clones = [v.detach().clone().requires_grad_() for v in inputs]
            components = reference['hype_triplet_losses'](clones[0], clones[1],
                hier_margin=1000./count, contr_margin=4, ball_dim=256)
            expected = reference['cal_loss'](clones[2], labels) + .01*components[4] + .01*components[5]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(torch.autograd.grad(actual, inputs), torch.autograd.grad(expected, clones)):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-10)

    def test_cb32_class_balance_and_roundtrip(self):
        from collections import Counter
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        labels = [c for c in range(40) for _ in range(c+1)]
        sampler = SourceClassBatchSampler(labels, world_size=1, rank=0, classes_per_rank=16,
                                          instances_per_class=2, steps=200, seed=4780)
        for step in range(200):
            plan = sampler.global_plan(step)
            self.assertEqual(sorted(Counter(plan.labels).values()), [2]*16)
            self.assertTrue(all(a != b for a, b in zip(plan.labels, reversed(plan.labels))))
        sampler.set_epoch(3)
        state = sampler.state_dict()
        sampler.set_epoch(0)
        sampler.load_state_dict(state)
        self.assertEqual(sampler.epoch, 3)

    def test_actual_namespace_has_no_extra_geometry_and_original_init(self):
        ns = runner.build_namespace()
        ns['seed_all'](4780)
        model = ns['configure_backbone'](ns['Hype_pointMLP']())
        self.assertEqual(ns['state_digest'](model), runner.INITIAL_DIGEST)
        self.assertEqual(ns['state_hash'](model), runner.INITIAL_HASH)
        self.assertFalse(model.emb._forward_hooks)
        args = ns['arguments'](['--hier','off','--data-dir','.','--run-dir','unused'])
        config = ns['fixed_configuration'](args)
        self.assertFalse(config['hier_enabled'])
        self.assertIsNone(config['proxy_optimizer'])
        self.assertEqual((args.epochs, args.steps_per_epoch), (300, 200))


if __name__ == '__main__':
    unittest.main(verbosity=2)
