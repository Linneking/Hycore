"""CPU-only contracts for matched TAN-C and TAN-HIER component ablations.

Only synthetic train/val fixtures are used. No real dataset/checkpoint, test split,
CUDA context, or GPU smoke run is accessed.
"""
import argparse
import contextlib
import io
import json
import math
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
from segmentation_ShapeNetPart.hycore_seg_v5.losses import HyCoReSegLoss as FullRegularizer
from segmentation_ShapeNetPart.hycore_seg_v6.cpu_entrypoint_check import write_fixture
from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2 as Reference
from segmentation_ShapeNetPart.hycore_seg_v12 import train as trainer
from segmentation_ShapeNetPart.hycore_seg_v12.monitoring import pair_summary

EVIDENCE = {}
COEFFICIENTS = {'contr': (.01, 0.), 'hier': (0., .01)}
DISPLAY_NAMES = {'contr': 'S5-TAN-C', 'hier': 'S5-TAN-HIER'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


class TangentComponentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError('These contracts must be CPU-only')

    def test_cli_component_coefficients_and_production_configs(self):
        base = ['--data', 'unused', '--out', 'unused', '--arm', 'h', '--device', 'cpu']
        package = Path(trainer.__file__).parent
        for component, coefficients in COEFFICIENTS.items():
            with self.subTest(component=component):
                args = trainer.arguments(base + ['--component', component, '--regularizers'])
                self.assertEqual((args.alpha, args.beta, args.supervision_scale), (*coefficients, 5.))
                with self.assertRaises(ValueError):
                    trainer.arguments(base + ['--component', component])
                wrong = [['--alpha', '0', '--beta', '0'], ['--alpha', '.01', '--beta', '.01'],
                         ['--supervision-scale', '1']]
                for extra in wrong:
                    with self.subTest(extra=extra), self.assertRaises(ValueError):
                        trainer.arguments(base + ['--component', component, '--regularizers', *extra])
                filename = 's5_tan_c.json' if component == 'contr' else 's5_tan_hier.json'
                config = trainer.arguments(base + ['--component', component, '--config', str(package / 'configs' / filename)])
                self.assertEqual((config.alpha, config.beta), coefficients)
                self.assertTrue(config.regularizers)
                self.assertEqual((config.seed, config.epochs, config.schedule_epochs, config.supervision_scale),
                                 (4780, 200, 200, 5.))
        for coefficients in ((0., 0.), (.01, .01), (.02, 0.), (0., .02)):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                trainer.HyCoReSegLoss(*coefficients)
        EVIDENCE['CLI_and_configs_match_selected_single_component'] = True

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

    def test_selected_components_raw_values_pairing_and_gradient_routes(self):
        torch.manual_seed(4780)
        model = trainer.NativeTail2('h', k=4).eval()
        monitor = trainer.GeometryMonitor().begin(True, 0, 1)
        model.monitor = monitor
        generator = torch.Generator().manual_seed(849)
        points = torch.randn(4, 3, 32, generator=generator) * .1
        categories = torch.tensor([0, 1, 2, 3])
        targets = torch.arange(128).reshape(4, 32) % 50
        logits, source = model(points, F.one_hot(categories, 16).float(), return_features=True)
        for node in ('post_conv9', 'regularizer_source', 'h'):
            self.assertIs(source, monitor.nodes[node])
        q = logits.transpose(1, 2).contiguous()
        ce = trainer.plain_ce(5 * q, targets) / 5
        q_gradient, = torch.autograd.grad(ce, q, retain_graph=True)
        torch.testing.assert_close(q_gradient, ((5 * q).softmax(-1) - F.one_hot(targets, 50)) / targets.numel())
        reference = FullRegularizer(.01, .01, part_min=4, part_max=8, seed=4780)
        candidates = {name: trainer.HyCoReSegLoss(*coefficients, part_min=4, part_max=8, seed=4780)
                      for name, coefficients in COEFFICIENTS.items()}
        route = (source, model.conv9[0].weight, model.conv8[0].weight,
                 model.emb2._weight, model.emb2._bias, model.mlr._weight, model.mlr._bias)
        # A second call also checks preservation of the independent sampling streams.
        for call in range(2):
            rng = torch.get_rng_state().clone()
            _, ref_diagnostic = reference(source, categories)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for component, candidate in candidates.items():
                with self.subTest(component=component, call=call):
                    reg, diagnostic = candidate(source, categories)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    inactive = 'hier' if component == 'contr' else 'contr'
                    self.assertTrue(reg.requires_grad)
                    self.assertIs(reg, candidate.last_components[component])
                    self.assertTrue(candidate.last_components[component].requires_grad)
                    self.assertFalse(candidate.last_components[inactive].requires_grad)
                    self.assertEqual(float(candidate.last_components[inactive]), 0.)
                    self.assertEqual(diagnostic['active_regularizer_component'], component)
                    self.assertTrue(diagnostic[component + '_training_active'])
                    self.assertFalse(diagnostic[inactive + '_training_active'])
                    self.assertEqual(diagnostic['active_weighted_' + inactive], 0.)
                    for key in ('part_indices_sha256', 'part_centers', 'part_points', 'negative_indices',
                                'hier', 'contr', 'gamma', 'delta'):
                        self.assertEqual(diagnostic[key], ref_diagnostic[key], key)
                    self.assertTrue(torch.equal(candidate.generator.get_state(), reference.generator.get_state()))
                    self.assertTrue(torch.equal(candidate.negative_generator.get_state(), reference.negative_generator.get_state()))
                    self.assertEqual(candidate.kernel.last_components, {})
                    torch.testing.assert_close(reg, reference.last_components[component], rtol=0., atol=0.)
                    actual = torch.autograd.grad(reg, route, retain_graph=True, allow_unused=True)
                    expected = torch.autograd.grad(reference.last_components[component], route,
                                                   retain_graph=True, allow_unused=True)
                    for a, b in zip(actual[:3], expected[:3]):
                        self.assertIsNotNone(a)
                        self.assertTrue(torch.isfinite(a).all())
                        self.assertGreater(float(a.norm()), 0., component)
                        torch.testing.assert_close(a, b, rtol=0., atol=0.)
                    self.assertTrue(all(g is None for g in actual[3:]), component)
                    total = ce + reg
                    total_route = torch.autograd.grad(total, route, retain_graph=True, allow_unused=True)
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in total_route))
                    if call == 0:
                        # Compare the actual complete network backward on this
                        # same graph with/without the observational monitor.
                        parameters = tuple(p for p in model.parameters() if p.requires_grad)
                        expected_all = torch.autograd.grad(total, parameters, retain_graph=True, allow_unused=True)
                        self.assertTrue(all(p.grad is None for p in parameters))
                        part_rng = candidate.generator.get_state().clone()
                        negative_rng = candidate.negative_generator.get_state().clone()
                        model_hash = state_hash(model.state_dict())
                        monitor.gradient_summary({'ce': ce, **candidate.last_components},
                                                 trainer.monitored_parameter_groups(model))
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertTrue(torch.equal(part_rng, candidate.generator.get_state()))
                        self.assertTrue(torch.equal(negative_rng, candidate.negative_generator.get_state()))
                        self.assertTrue(all(p.grad is None for p in parameters))
                        self.assertEqual(model_hash, state_hash(model.state_dict()))
                        total.backward(retain_graph=True)
                        for p, expected_gradient in zip(parameters, expected_all):
                            if expected_gradient is None:
                                self.assertIsNone(p.grad)
                            else:
                                torch.testing.assert_close(p.grad, expected_gradient, rtol=0., atol=0.)
                        model.zero_grad(set_to_none=True)
        monitor.capture_targets(targets)
        observed = monitor.summarize()['geometry']['final_tangent']
        self.assertIn('scaled_correct_class_probability', observed)
        for key in ('near_boundary_fraction', 'projection_radius', 'depth'):
            self.assertNotIn(key, observed)
        EVIDENCE['both_single_components_match_full_R_values_gradients_and_sampling'] = True
        EVIDENCE['R_reaches_U_conv9_conv8_bypasses_both_M'] = True
        EVIDENCE['inactive_component_no_training_graph'] = True
        EVIDENCE['CE5_over5_gradient_correct_and_no_global_RNG_consumption'] = True
        EVIDENCE['monitor_preserves_all_network_gradients_model_and_sampling_RNG'] = True
        EVIDENCE['q_keeps_real_score_monitoring'] = True

    def test_pair_summary_known_angle_and_virtual_direction_sign(self):
        a, b = torch.tensor([1., 2.]), torch.tensor([-3., 4.])
        pair = pair_summary(a, b)
        self.assertEqual(pair['status'], 'nonzero')
        self.assertAlmostEqual(pair['dot'], 5.)
        self.assertAlmostEqual(pair['cosine'], 1. / math.sqrt(5.))
        self.assertAlmostEqual(pair['u_virtual_negative_first_directional_derivative_of_second'], -5.)
        opposed = pair_summary(a, -a)
        self.assertAlmostEqual(opposed['cosine'], -1.)
        self.assertAlmostEqual(opposed['u_virtual_negative_first_directional_derivative_of_second'], 5.)
        self.assertEqual(pair_summary(a, torch.tensor([2., -1.]))['dot'], 0.)
        self.assertEqual(pair_summary(a, torch.zeros_like(a))['status'], 'zero_norm')
        self.assertEqual(pair_summary(a, None)['status'], 'inactive_or_unused')
        with self.assertRaises(ValueError):
            pair_summary(a, torch.zeros(3))
        EVIDENCE['known_gradient_angles_and_virtual_U_direction_sign_correct'] = True

    def test_gradient_monitor_is_observational(self):
        node = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
        unrelated = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        ce = node[0, 0] + 2 * node[0, 1]
        reg = -3 * node[0, 0] + 4 * node[0, 1]
        inactive = node.new_zeros(())
        expected, = torch.autograd.grad(ce + reg, node, retain_graph=True)
        monitor = trainer.GeometryMonitor().begin(True, 0, 1)
        monitor.capture('h', node)
        rng = torch.get_rng_state().clone()
        result = monitor.gradient_summary({'ce': ce, 'contr': reg, 'hier': inactive})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(node.grad)
        pair = result['gradient_pairs_at_u']['pairs']['ce__contr']
        self.assertAlmostEqual(pair['cosine'], 1. / math.sqrt(5.))
        self.assertEqual(pair['u_virtual_negative_first_directional_derivative_of_second'], -5.)
        self.assertEqual(result['gradient_pairs_at_u']['pairs']['ce__hier']['status'], 'inactive_or_unused')
        # A differentiable loss unrelated to U must also remain explicitly unused.
        unused = monitor.gradient_summary({'ce': ce, 'unrelated': unrelated.square()})
        self.assertEqual(unused['gradient_pairs_at_u']['pairs']['ce__unrelated']['status'], 'inactive_or_unused')
        self.assertIsNone(node.grad)
        self.assertIsNone(unrelated.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        (ce + reg).backward()
        torch.testing.assert_close(node.grad, expected, rtol=0., atol=0.)
        EVIDENCE['gradient_monitor_preserves_RNG_grad_fields_and_total_backward'] = True

    def test_complete_CPU_two_epoch_entrypoints(self):
        entries = {}
        with tempfile.TemporaryDirectory(prefix='hycore_tan_components_cpu_') as temporary:
            root = Path(temporary)
            write_fixture(root / 'data')
            for component in COEFFICIENTS:
                with self.subTest(component=component):
                    out = root / component
                    out.mkdir()
                    args = trainer.arguments([
                        '--data', str(root / 'data'), '--out', str(out), '--arm', 'h',
                        '--component', component, '--regularizers', '--protocol', 'validation', '--epochs', '2',
                        '--schedule-epochs', '200', '--batch-size', '4', '--eval-batch-size', '4',
                        '--num-points', '32', '--k', '4', '--workers', '0', '--cpu-threads', '2',
                        '--device', 'cpu', '--part-min', '4', '--part-max', '8', '--max-steps', '2',
                        '--monitor-every', '1', '--gradient-every', '1', '--no-final-test', '--allow-dirty-cpu'])
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        trainer.train(args)
                    manifest, acceptance = (read_json(out / name) for name in ('manifest.json', 'acceptance.json'))
                    self.assertTrue(acceptance['complete'])
                    self.assertEqual((acceptance['epochs'], acceptance['updates']), (2, 4))
                    self.assertEqual(manifest['format'], 'hycore_seg_v12')
                    self.assertEqual(manifest['display_name'], DISPLAY_NAMES[component])
                    self.assertEqual(manifest['active_regularizer_component'], component)
                    self.assertTrue(manifest['regularizers_active'])
                    self.assertFalse(manifest['regularizer_diagnostics_only'])
                    self.assertEqual(manifest['regularizer_diagnostics'],
                                     'same_v3_kernel_single_active_component; inactive_raw_diagnostics_only')
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
                    inactive = 'hier' if component == 'contr' else 'contr'
                    for r in observations:
                        self.assertEqual(set(r['gradients']), {'ce', 'contr', 'hier'})
                        self.assertTrue(r['diagnostics']['regularizers_active'])
                        self.assertEqual(r['diagnostics']['active_regularizer_component'], component)
                        self.assertEqual(r['diagnostics']['active_weighted_' + inactive], 0.)
                        self.assertNotIn('m3', r['geometry'])
                        self.assertNotIn('near_boundary_fraction', r['geometry']['final_tangent'])
                        self.assertIn('scaled_correct_class_probability', r['geometry']['final_tangent'])
                        active = r['gradients'][component]
                        self.assertTrue(active['loss_requires_grad'])
                        self.assertNotEqual(active['nodes']['h']['status'], 'none')
                        self.assertEqual(active['nodes']['final_tangent']['status'], 'none')
                        self.assertEqual(active['parameter_groups']['m2']['status'], 'none')
                        self.assertEqual(active['parameter_groups']['m3']['status'], 'none')
                        self.assertFalse(r['gradients'][inactive]['loss_requires_grad'])
                        self.assertEqual(r['gradients'][inactive]['nodes']['h']['status'], 'none')
                        pairs = r['gradient_pairs_at_u']
                        self.assertEqual(pairs['coordinate_space'], 'ambient_euclidean_U_post_conv9')
                        pair = pairs['pairs']['ce__' + component]
                        self.assertIn(pair['status'], ('nonzero', 'zero_norm'))
                        if pair['status'] == 'nonzero':
                            self.assertTrue(math.isfinite(pair['cosine']))
                            self.assertAlmostEqual(pair['u_virtual_negative_first_directional_derivative_of_second'],
                                                   -pair['dot'])
                        self.assertEqual(pairs['pairs']['ce__' + inactive]['status'], 'inactive_or_unused')
                    checkpoints = {name: torch.load(out / (name + '.pth'), map_location='cpu', weights_only=False)
                                   for name in ('e0', 'best', 'last')}
                    initial, last = checkpoints['e0'], checkpoints['last']
                    self.assertEqual(last['format'], 'hycore_seg_v12_complete')
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
                    entries[component] = dict(epochs=2, updates=4, BN_updates=4,
                        all50parts=True, model_optimizer_restorable=True, R_active=True,
                        inactive_R_has_no_training_graph=True, R_bypasses_both_M=True,
                        gradient_pairs_recorded=True, test_accessed=False, scheduler_horizon=200,
                        initial_model_sha256=manifest['initial_model_sha256'])
        self.assertEqual(entries['contr']['initial_model_sha256'], entries['hier']['initial_model_sha256'])
        EVIDENCE['complete_CPU_entrypoints'] = entries


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
