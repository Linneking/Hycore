"""CPU contracts for the two declared no-R TAN tail interventions.

Only synthetic train/val fixtures are used. No production checkpoint, real data,
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
import numpy as np
from torch.nn import functional as F

from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash
from segmentation_ShapeNetPart.hycore_seg_v6.cpu_entrypoint_check import write_fixture
from segmentation_ShapeNetPart.hycore_seg_v10.model import NativeTail2 as Reference
from segmentation_ShapeNetPart.hycore_seg_v10.losses import HyCoReSegLoss as ReferenceLoss
from segmentation_ShapeNetPart.hycore_seg_v13 import train as trainer
from segmentation_ShapeNetPart.hycore_seg_v9.metrics import SegMetrics as ReferenceMetrics

EVIDENCE = {}
DISPLAYS = {'bnleaky': 'TAN-BNLEAKY-R0', 'bias': 'TAN-BIAS-R0'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


class TailContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU-only contracts must not initialize CUDA')

    def test_cli_and_production_configs(self):
        base = ['--data', 'unused', '--out', 'unused', '--arm', 'h', '--device', 'cpu']
        for variant, filename in [('bnleaky', 'tan_bnleaky_r0.json'), ('bias', 'tan_bias_r0.json')]:
            args = trainer.arguments(base + ['--variant', variant])
            self.assertEqual((args.alpha, args.beta, args.supervision_scale), (0., 0., 5.))
            self.assertFalse(args.regularizers)
            for extra in (['--regularizers'], ['--alpha', '.01'], ['--beta', '.01'],
                          ['--supervision-scale', '1']):
                with self.subTest(variant=variant, extra=extra), self.assertRaises(ValueError):
                    trainer.arguments(base + ['--variant', variant, *extra])
            config = trainer.arguments(base + ['--variant', variant, '--config',
                str(Path(trainer.__file__).parent / 'configs' / filename)])
            self.assertEqual((config.seed, config.epochs, config.schedule_epochs), (4780, 200, 200))
            self.assertEqual((config.alpha, config.beta, config.regularizers), (0., 0., False))
        EVIDENCE['declared_configs_Roff_scale5_final_bnleaky_and_bias'] = True

    def test_shared_initialization_and_rng_match_R0(self):
        torch.manual_seed(4780)
        reference = Reference('h', k=4)
        rng = torch.get_rng_state().clone()
        reference_frozen = {name for name, p in reference.named_parameters() if not p.requires_grad}
        for variant in DISPLAYS:
            torch.manual_seed(4780)
            model = trainer.NativeTail2('h', k=4, variant=variant)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            extras = set(model.state_dict()) - set(reference.state_dict())
            frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
            if variant == 'bnleaky':
                self.assertEqual(extras, {'tail_bn.weight', 'tail_bn.bias', 'tail_bn.running_mean',
                                         'tail_bn.running_var', 'tail_bn.num_batches_tracked'})
                self.assertTrue(torch.equal(model.tail_bn.weight, torch.ones(128)))
                self.assertTrue(torch.equal(model.tail_bn.bias, torch.zeros(128)))
                self.assertEqual(set(frozen), reference_frozen)
                head_ids = {id(p) for p in model.head_parameters()}
                self.assertTrue(all(id(p) in head_ids for p in model.tail_bn.parameters()))
                self.assertIn('tail_bn', trainer.monitored_parameter_groups(model))
            else:
                self.assertEqual(extras, {'score_bias'})
                self.assertEqual(set(frozen), reference_frozen)
                self.assertTrue(torch.equal(model.score_bias, torch.zeros(50)))
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 sum(p.numel() for p in reference.parameters()) + 50)
                self.assertIn(id(model.score_bias), {id(p) for p in model.head_parameters()})
                self.assertIn('readout_bias', trainer.monitored_parameter_groups(model))
                self.assertIn('_bias', trainer.monitored_parameter_groups(model)['m3'])
        EVIDENCE['all_shared_model_tensors_and_constructor_RNG_match_R0'] = True
        EVIDENCE['BN_default_affine_initialization_and_head_membership'] = True
        EVIDENCE['bias_adds_exactly50_parameters_and_keeps_transport_trainable'] = True

    def test_forward_formulas_and_zero_bias_exact_R0_equivalence(self):
        generator = torch.Generator().manual_seed(731)
        source = torch.randn(4, 32, 256, generator=generator) * .02
        torch.manual_seed(4780)
        reference = Reference('h', k=4).eval()
        for variant in DISPLAYS:
            torch.manual_seed(4780)
            model = trainer.NativeTail2('h', k=4, variant=variant).eval()
            with torch.no_grad():
                tangent = model.manifold.logmap0(model.emb2(model.manifold.expmap0(source)))
                if variant == 'bnleaky':
                    transformed = F.leaky_relu(model.tail_bn(tangent.transpose(1, 2)).transpose(1, 2).contiguous(), .2)
                    expected = F.linear(transformed, model.mlr.weight)
                else:
                    expected = F.linear(tangent, model.mlr.weight) + model.score_bias
                actual = model.forward_head(source)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual.shape, (4, 32, 50))
                if variant == 'bias':
                    torch.testing.assert_close(actual, reference.forward_head(source), rtol=0., atol=0.)
                    before_labels = actual.argmax(-1)
                    self.assertFalse(bool((before_labels == 17).all()))
                    model.score_bias[17] = 50.
                    changed = model.forward_head(source)
                    torch.testing.assert_close(changed, actual + model.score_bias, rtol=0., atol=0.)
                    self.assertTrue(bool((changed.argmax(-1) == 17).all()))
        EVIDENCE['BNLeaky_and_affine_readout_forward_formulas_exact'] = True
        EVIDENCE['zero_bias_forward_exact_R0_and_nonzero_bias_changes_predictions'] = True

    def test_Roff_sampling_matches_and_has_no_training_graph(self):
        generator = torch.Generator().manual_seed(637)
        source = torch.randn(4, 32, 256, generator=generator, requires_grad=True) * .02
        categories = torch.arange(4)
        candidate = trainer.HyCoReSegLoss(0., 0., part_min=4, part_max=8, seed=4780)
        reference = ReferenceLoss(0., 0., part_min=4, part_max=8, seed=4780)
        for _ in range(2):
            rng = torch.get_rng_state().clone()
            value, actual = candidate(source, categories)
            expected, diagnostic = reference(source, categories)
            self.assertFalse(value.requires_grad)
            self.assertEqual(float(value), 0.)
            for component in candidate.last_components.values():
                self.assertFalse(component.requires_grad)
                self.assertEqual(float(component), 0.)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key in ('part_centers', 'part_points', 'negative_indices', 'part_indices_sha256', 'hier', 'contr'):
                self.assertEqual(actual[key], diagnostic[key])
            self.assertTrue(torch.equal(candidate.generator.get_state(), reference.generator.get_state()))
            self.assertTrue(torch.equal(candidate.negative_generator.get_state(), reference.negative_generator.get_state()))
        EVIDENCE['Roff_no_training_graph_and_reference_sampling_preserved'] = True

    def test_absence_metrics_preserve_original_IoU_and_separate_present_only(self):
        # Part49 is absent in two shapes, false-positive in one, and present in
        # the third with IoU .5. Present-only=.5; absent-both reward stays separate.
        target = np.array([[47, 47, 48, 48], [47, 47, 48, 48], [47, 48, 49, 49]])
        prediction = np.array([[47, 49, 48, 48], [47, 47, 48, 48], [47, 48, 49, 48]])
        categories = np.array([15, 15, 15])
        candidate, reference = trainer.SegMetrics(), ReferenceMetrics()
        actual_shape_iou = candidate.update(prediction, target, categories)
        expected_shape_iou = reference.update(prediction, target, categories)
        np.testing.assert_array_equal(actual_shape_iou, expected_shape_iou)
        actual, expected = candidate.result(), reference.result()
        for key, value in expected.items():
            if key != 'parts':
                self.assertEqual(actual[key], value)
            else:
                for part, fields in value.items():
                    self.assertEqual({field: actual['parts'][part][field] for field in fields}, fields)
        part49 = actual['parts']['49']
        self.assertEqual(part49['absent_objects'], 2)
        self.assertEqual(part49['false_positive_absent_objects'], 1)
        self.assertEqual(part49['present_objects'], 1)
        self.assertEqual(part49['mean_iou_gt_present'], .5)
        self.assertEqual(part49['mean_iou_all_objects_absent_both_is_one'], .5)
        self.assertIsNone(actual['parts']['0']['mean_iou_gt_present'])
        self.assertIsNone(actual['parts']['0']['mean_iou_all_objects_absent_both_is_one'])
        EVIDENCE['GT_absence_FP_and_present_only_IoU_observations_preserve_original_metrics'] = True

    def test_backward_and_monitor_are_observational(self):
        generator = torch.Generator().manual_seed(849)
        points = torch.randn(4, 3, 32, generator=generator) * .1
        categories = torch.arange(4)
        targets = torch.arange(128).reshape(4, 32) % 50
        for variant in DISPLAYS:
            torch.manual_seed(4780)
            model = trainer.NativeTail2('h', k=4, variant=variant).eval()
            monitor = trainer.GeometryMonitor().begin(True, 0, 1)
            model.monitor = monitor
            output, source = model(points, F.one_hot(categories, 16).float(), return_features=True)
            q = output.transpose(1, 2).contiguous()
            ce = trainer.plain_ce(5 * q, targets) / 5
            q_grad, = torch.autograd.grad(ce, q, retain_graph=True)
            torch.testing.assert_close(q_grad, ((5 * q).softmax(-1) - F.one_hot(targets, 50)) / targets.numel())
            loss = trainer.HyCoReSegLoss(0., 0., part_min=4, part_max=8, seed=4780)
            loss.monitor = monitor
            reg, _ = loss(source, categories)
            parameters = tuple(p for p in model.parameters() if p.requires_grad)
            expected = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in expected))
            before, rng = state_hash(model.state_dict()), torch.get_rng_state().clone()
            monitor.capture_targets(targets)
            observed = monitor.summarize()
            gradients = monitor.gradient_summary({'ce': ce, **loss.last_components}, trainer.monitored_parameter_groups(model))
            self.assertEqual(before, state_hash(model.state_dict()))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            q_summary = observed['geometry']['final_tangent']
            self.assertIn('scaled_correct_class_probability', q_summary)
            self.assertNotIn('near_boundary_fraction', q_summary)
            self.assertNotIn('m3', observed['geometry'])
            for name in ('contr', 'hier'):
                self.assertFalse(gradients['gradients'][name]['loss_requires_grad'])
                self.assertEqual(gradients['gradients'][name]['nodes']['h']['status'], 'none')
            (ce + reg).backward()
            for parameter, wanted in zip(parameters, expected):
                torch.testing.assert_close(parameter.grad, wanted, rtol=0., atol=0.)
            if variant == 'bias':
                self.assertGreater(float(model.score_bias.grad.norm()), 0.)
                self.assertGreater(float(model.mlr._bias.grad.norm()), 0.)
            else:
                self.assertGreater(float(model.tail_bn.weight.grad.norm()), 0.)
                self.assertGreater(float(model.mlr._bias.grad.norm()), 0.)
        EVIDENCE['CE5_over5_backward_correct_and_finite_for_all_trainable_parameters'] = True
        EVIDENCE['monitor_preserves_all_final_parameter_gradients_model_state_and_RNG'] = True
        EVIDENCE['new_BN_and_retained_transport_receive_CE_gradients'] = True

    def test_complete_CPU_two_epoch_entrypoints(self):
        entries = {}
        with tempfile.TemporaryDirectory(prefix='hycore_tan_tail_cpu_') as temporary:
            root = Path(temporary)
            write_fixture(root / 'data')
            for variant in DISPLAYS:
                out = root / variant
                out.mkdir()
                args = trainer.arguments([
                    '--data', str(root / 'data'), '--out', str(out), '--arm', 'h', '--variant', variant,
                    '--protocol', 'validation', '--epochs', '2', '--schedule-epochs', '200',
                    '--batch-size', '4', '--eval-batch-size', '4', '--num-points', '32', '--k', '4',
                    '--workers', '0', '--cpu-threads', '2', '--device', 'cpu', '--part-min', '4',
                    '--part-max', '8', '--max-steps', '2', '--monitor-every', '1', '--gradient-every', '1',
                    '--no-final-test', '--allow-dirty-cpu'])
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    trainer.train(args)
                manifest = read_json(out / 'manifest.json')
                acceptance = read_json(out / 'acceptance.json')
                self.assertTrue(acceptance['complete'])
                self.assertEqual((acceptance['epochs'], acceptance['updates']), (2, 4))
                self.assertEqual(manifest['format'], 'hycore_seg_v13')
                self.assertEqual(manifest['display_name'], DISPLAYS[variant])
                self.assertEqual(manifest['tail_variant'], variant)
                self.assertNotIn('cosine_effective_softmax_scale', manifest)
                fixed_curvature = {'manifold.isp_c', 'manifold2.isp_c', 'manifold3.isp_c'}
                self.assertEqual(set(manifest['frozen_parameter_names']),
                                 fixed_curvature)
                self.assertFalse(manifest['regularizers_active'])
                self.assertTrue(manifest['regularizer_diagnostics_only'])
                self.assertEqual(manifest['regularizer_feature_location'], 'after_conv9_bn9_leaky')
                self.assertFalse(manifest['dropout_active'])
                self.assertFalse(manifest['gradient_clipping'])
                rows, observations = read_lines(out / 'metrics.jsonl'), read_lines(out / 'geometry.jsonl')
                self.assertEqual([r['steps'] for r in rows], [2, 2])
                for row in rows:
                    self.assertEqual((row['objects_seen'], row['val']['objects']), (8, 8))
                    self.assertEqual(len(row['train']['parts']), 50)
                    self.assertEqual(len(row['val']['parts']), 50)
                    for split in ('train', 'val'):
                        for part in row[split]['parts'].values():
                            self.assertTrue({'absent_objects', 'false_positive_absent_objects',
                                             'mean_iou_gt_present', 'mean_iou_all_objects_absent_both_is_one'} <= set(part))
                    self.assertEqual(row['train_reg'], 0.)
                    self.assertEqual(row['train_loss'], row['train_ce'])
                    self.assertEqual(set(row['bn_counts'].values()), {row['updates']})
                    self.assertEqual('tail_bn' in row['bn_counts'], variant == 'bnleaky')
                for observed in observations:
                    self.assertEqual(observed['diagnostics']['active_weighted_contr'], 0.)
                    self.assertEqual(observed['diagnostics']['active_weighted_hier'], 0.)
                    self.assertIn('tail_tangent', observed['geometry'])
                    self.assertEqual('tail_bn' in observed['geometry'], variant == 'bnleaky')
                    if variant == 'bias':
                        self.assertIn('readout_bias', observed['gradients']['ce']['parameter_groups'])
                    for name in ('contr', 'hier'):
                        self.assertFalse(observed['gradients'][name]['loss_requires_grad'])
                        self.assertEqual(observed['gradients'][name]['nodes']['h']['status'], 'none')
                initial = torch.load(out / 'e0.pth', map_location='cpu', weights_only=False)
                last = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
                self.assertEqual(last['format'], 'hycore_seg_v13_complete')
                self.assertEqual((last['epoch'], last['updates']), (2, 4))
                self.assertEqual(last['scheduler']['T_max'], 200)
                self.assertTrue(trainer.finite_tensor_state(last['model']))
                self.assertTrue(trainer.finite_tensor_state(last['optimizer']))
                self.assertEqual({int(v) for k, v in last['model'].items() if k.endswith('num_batches_tracked')}, {4})
                restored = trainer.NativeTail2('h', k=4, variant=variant)
                restored.load_state_dict(last['model'], strict=True)
                self.assertEqual(state_hash(restored.state_dict()), state_hash(last['model']))
                optimizer = trainer.geoopt.optim.RiemannianSGD(restored.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
                optimizer.load_state_dict(last['optimizer'])
                self.assertTrue(trainer.finite_tensor_state(optimizer.state_dict()))
                if variant == 'bias':
                    self.assertTrue(restored.mlr._bias.requires_grad)
                    self.assertTrue(restored.score_bias.requires_grad)
                    self.assertFalse(torch.equal(last['model']['score_bias'], initial['model']['score_bias']))
                self.assertFalse(initial['rng']['cuda'])
                self.assertFalse(last['rng']['cuda'])
                self.assertFalse((root / 'data/test.h5').exists())
                self.assertFalse((out / 'final_test.json').exists())
                self.assertFalse(torch.cuda.is_initialized())
                entries[variant] = dict(epochs=2, updates=4, BN_updates=4, all50parts=True,
                    model_optimizer_restorable=True, R_training_graph=False, test_accessed=False,
                    scheduler_horizon=200, initial_model_sha256=manifest['initial_model_sha256'],
                    initial_shared_head_sha256=manifest['initial_shared_head_sha256'],
                    initial_shared_model_sha256=manifest['initial_shared_model_sha256'])
        self.assertEqual(entries['bnleaky']['initial_shared_head_sha256'], entries['bias']['initial_shared_head_sha256'])
        self.assertEqual(entries['bnleaky']['initial_shared_model_sha256'], entries['bias']['initial_shared_model_sha256'])
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
