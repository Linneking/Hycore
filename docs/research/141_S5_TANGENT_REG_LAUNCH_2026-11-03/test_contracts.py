"""CPU-only contracts for restoring both original R terms to S5-TAN.

Reuses the established v6 train/val fixture. No real data, production checkpoint,
test split, CUDA context, or GPU smoke run is accessed.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash
from segmentation_ShapeNetPart.hycore_seg_v6.cpu_entrypoint_check import write_fixture
from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2 as Reference
from segmentation_ShapeNetPart.hycore_seg_v11 import train as trainer

EVIDENCE = {}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


class TangentRegularizerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError('These contracts must be CPU-only')

    def test_cli_requires_original_both_regularizers_and_scale5(self):
        base = ['--data', 'unused', '--out', 'unused', '--arm', 'h', '--device', 'cpu']
        args = trainer.arguments(base + ['--regularizers'])
        self.assertEqual((args.alpha, args.beta, args.supervision_scale), (.01, .01, 5.))
        with self.assertRaises(ValueError):
            trainer.arguments(base)
        for extra in (['--alpha', '0'], ['--beta', '0'], ['--supervision-scale', '1']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                trainer.arguments(base + ['--regularizers', *extra])
        EVIDENCE['cli_requires_scale5_and_both_R001'] = True

    def test_model_initialization_and_rng_exactly_match_R0(self):
        torch.manual_seed(4780)
        reference = Reference('h', k=4)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(4780)
        candidate = trainer.NativeTail2('h', k=4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(reference.state_dict()), list(candidate.state_dict()))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        self.assertTrue(torch.equal(reference.mlr.weight, candidate.mlr.weight))
        EVIDENCE['all_model_tensors_and_constructor_RNG_match_R0'] = True

    def test_active_regularizer_gradient_route_and_paired_sampling(self):
        torch.manual_seed(4780)
        # Eval BN at initialization gives a well-conditioned small synthetic U.
        # Training-mode BN/updates are tested by the complete entrypoint below.
        model = trainer.NativeTail2('h', k=4).eval()
        monitor = trainer.GeometryMonitor().begin(True, 0, 1)
        model.monitor = monitor
        generator = torch.Generator().manual_seed(849)
        points = torch.randn(4, 3, 32, generator=generator) * .1
        categories = torch.tensor([0, 1, 2, 3])
        targets = torch.arange(128).reshape(4, 32) % 50
        logits, source = model(points, F.one_hot(categories, 16).float(), return_features=True)
        self.assertIs(source, monitor.nodes['post_conv9'])
        self.assertIs(source, monitor.nodes['regularizer_source'])
        self.assertIs(source, monitor.nodes['h'])
        q = logits.transpose(1, 2).contiguous()
        ce = trainer.plain_ce(5 * q, targets) / 5
        q_gradient, = torch.autograd.grad(ce, q, retain_graph=True)
        torch.testing.assert_close(q_gradient, ((5 * q).softmax(-1) - F.one_hot(targets, 50)) / targets.numel())

        active = trainer.HyCoReSegLoss(.01, .01, part_min=4, part_max=8, seed=4780)
        active.monitor = monitor
        rng = torch.get_rng_state().clone()
        reg, diagnostic = active(source, categories)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(reg.requires_grad)
        self.assertTrue(diagnostic['regularizers_active'])
        self.assertFalse(diagnostic['regularizer_diagnostics_only'])
        torch.testing.assert_close(reg, active.last_components['contr'] + active.last_components['hier'])
        route = (source, model.conv9[0].weight, model.conv8[0].weight,
                 model.emb2._weight, model.emb2._bias, model.mlr._weight, model.mlr._bias)
        for name, loss in active.last_components.items():
            self.assertTrue(loss.requires_grad)
            gradients = torch.autograd.grad(loss, route, retain_graph=True, allow_unused=True)
            for g in gradients[:3]:
                self.assertIsNotNone(g)
                self.assertTrue(torch.isfinite(g).all())
                self.assertGreater(float(g.norm()), 0., name)
            self.assertTrue(all(g is None for g in gradients[3:]), name)

        off = trainer.HyCoReSegLoss(0., 0., part_min=4, part_max=8, seed=4780)
        no_reg, no_diagnostic = off(source, categories)
        self.assertFalse(no_reg.requires_grad)
        for key in ('part_indices_sha256', 'part_centers', 'part_points', 'negative_indices'):
            self.assertEqual(diagnostic[key], no_diagnostic[key], key)
        monitor.capture_targets(targets)
        observed = monitor.summarize()['geometry']['final_tangent']
        self.assertIn('scaled_correct_class_probability', observed)
        for key in ('near_boundary_fraction', 'projection_radius', 'depth'):
            self.assertNotIn(key, observed)
        self.assertTrue(torch.isfinite(ce + reg))
        EVIDENCE['active_R_reaches_U_conv9_conv8_bypasses_both_M'] = True
        EVIDENCE['CE5_over5_gradient_and_R_sum_correct'] = True
        EVIDENCE['R_on_off_pairing_equal_and_no_global_RNG_consumption'] = True
        EVIDENCE['q_keeps_real_score_monitoring'] = True

    def test_complete_CPU_two_epoch_entrypoint(self):
        with tempfile.TemporaryDirectory(prefix='hycore_tan_r_cpu_') as temporary:
            root = Path(temporary)
            write_fixture(root / 'data')
            out = root / 'run'
            out.mkdir()
            args = trainer.arguments([
                '--data', str(root / 'data'), '--out', str(out), '--arm', 'h',
                '--regularizers', '--protocol', 'validation', '--epochs', '2',
                '--schedule-epochs', '200', '--batch-size', '4', '--eval-batch-size', '4',
                '--num-points', '32', '--k', '4', '--workers', '0', '--cpu-threads', '2',
                '--device', 'cpu', '--part-min', '4', '--part-max', '8', '--max-steps', '2',
                '--monitor-every', '1', '--gradient-every', '1', '--no-final-test', '--allow-dirty-cpu'])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer.train(args)
            manifest, acceptance = (read_json(out / name) for name in ('manifest.json', 'acceptance.json'))
            self.assertTrue(acceptance['complete'])
            self.assertEqual((acceptance['epochs'], acceptance['updates']), (2, 4))
            self.assertEqual(manifest['format'], 'hycore_seg_v11')
            self.assertEqual(manifest['display_name'], 'S5-TAN-R')
            self.assertTrue(manifest['regularizers_active'])
            self.assertFalse(manifest['regularizer_diagnostics_only'])
            self.assertEqual(manifest['regularizer_diagnostics'], 'same_v3_kernel_active_autograd')
            self.assertEqual(manifest['supervision_scale'], 5.)
            self.assertEqual(manifest['regularizer_feature_location'], 'after_conv9_bn9_leaky')
            self.assertFalse(manifest['dropout_active'])
            self.assertFalse(manifest['gradient_clipping'])
            rows, observations = read_lines(out / 'metrics.jsonl'), read_lines(out / 'geometry.jsonl')
            self.assertEqual([r['steps'] for r in rows], [2, 2])
            self.assertEqual(len(observations), 4)
            for r in rows:
                self.assertEqual((r['objects_seen'], r['val']['objects']), (8, 8))
                self.assertEqual(len(r['train']['parts']), 50)
                self.assertEqual(len(r['val']['parts']), 50)
                self.assertGreater(r['train_reg'], 0.)
                self.assertGreater(r['train_loss'], r['train_ce'])
                self.assertEqual(set(r['bn_counts'].values()), {r['updates']})
            for r in observations:
                self.assertEqual(set(r['gradients']), {'ce', 'contr', 'hier'})
                self.assertTrue(r['diagnostics']['regularizers_active'])
                self.assertFalse(r['diagnostics']['regularizer_diagnostics_only'])
                self.assertNotIn('m3', r['geometry'])
                self.assertNotIn('near_boundary_fraction', r['geometry']['final_tangent'])
                self.assertIn('scaled_correct_class_probability', r['geometry']['final_tangent'])
                for component in ('contr', 'hier'):
                    d = r['gradients'][component]
                    self.assertTrue(d['loss_requires_grad'])
                    self.assertNotEqual(d['nodes']['h']['status'], 'none')
                    self.assertEqual(d['nodes']['final_tangent']['status'], 'none')
                    self.assertEqual(d['parameter_groups']['m2']['status'], 'none')
                    self.assertEqual(d['parameter_groups']['m3']['status'], 'none')
            checkpoints = {name: torch.load(out / (name + '.pth'), map_location='cpu', weights_only=False)
                           for name in ('e0', 'best', 'last')}
            initial, last = checkpoints['e0'], checkpoints['last']
            self.assertEqual(last['format'], 'hycore_seg_v11_complete')
            self.assertEqual((last['epoch'], last['updates']), (2, 4))
            self.assertEqual(last['scheduler']['T_max'], 200)
            self.assertTrue(trainer.finite_tensor_state(last['model']))
            self.assertTrue(trainer.finite_tensor_state(last['optimizer']))
            self.assertEqual({int(v) for k, v in last['model'].items() if k.endswith('num_batches_tracked')}, {4})
            restored = trainer.NativeTail2('h', k=4)
            restored.load_state_dict(last['model'], strict=True)
            self.assertEqual(state_hash(restored.state_dict()), state_hash(last['model']))
            optimizer = trainer.geoopt.optim.RiemannianSGD(restored.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
            optimizer.load_state_dict(last['optimizer'])
            self.assertTrue(trainer.finite_tensor_state(optimizer.state_dict()))
            self.assertFalse(initial['rng']['cuda'])
            self.assertFalse(last['rng']['cuda'])
            self.assertFalse((root / 'data/test.h5').exists())
            self.assertFalse((out / 'final_test.json').exists())
            self.assertFalse(torch.cuda.is_initialized())
            EVIDENCE['complete_CPU_entrypoint'] = dict(epochs=2, updates=4, BN_updates=4,
                all50parts=True, model_optimizer_restorable=True, R_active=True,
                R_bypasses_both_M=True, test_accessed=False, scheduler_horizon=200,
                initial_model_sha256=manifest['initial_model_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence-out')
    extra, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if extra.evidence_out:
        evidence = dict(complete=result.result.wasSuccessful(), tests_run=result.result.testsRun,
                        cuda_initialized=torch.cuda.is_initialized(), scope='synthetic_CPU_only', **EVIDENCE)
        Path(extra.evidence_out).write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    sys.exit(0 if result.result.wasSuccessful() else 1)
