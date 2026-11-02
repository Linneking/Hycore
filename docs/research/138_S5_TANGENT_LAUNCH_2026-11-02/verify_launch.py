"""CPU-only acceptance of an existing production run; no model forward."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = read(args.run / 'manifest.json')
    reference = read(args.reference / 'manifest.json')
    rows = [json.loads(s) for s in (args.run / 'metrics.jsonl').read_text().splitlines() if s]
    ref_first = json.loads((args.reference / 'metrics.jsonl').read_text().splitlines()[0])
    first = rows[0]
    e0 = torch.load(args.run / 'e0.pth', map_location='cpu', weights_only=False)
    ref0 = torch.load(args.reference / 'e0.pth', map_location='cpu', weights_only=False)
    e1 = torch.load(args.run / 'epoch_001.pth', map_location='cpu', weights_only=False)
    ignored = {'config', 'out', 'physical_gpu'}
    checks = {
        'fresh_initial_hash_matches_s5': manifest['initial_model_sha256'] == reference['initial_model_sha256'],
        'all_e0_model_tensors_exactly_equal': e0['model'].keys() == ref0['model'].keys()
            and all(torch.equal(t, ref0['model'][k]) for k, t in e0['model'].items()),
        'initial_cpu_rng_exactly_equal': torch.equal(e0['rng']['torch'], ref0['rng']['torch']),
        'data_manifest_equal': manifest['data_manifest_sha256'] == reference['data_manifest_sha256'],
        'reference_config_equal_except_paths_gpu':
            {k:v for k,v in manifest['config'].items() if k not in ignored}
            == {k:v for k,v in reference['config'].items() if k not in ignored},
        'first_input_exactly_equal': first['first_input_sha256'] == ref_first['first_input_sha256'],
        'first_pairing_exactly_equal': first['first_pairing'] == ref_first['first_pairing'],
        'first_full_epoch437': first['epoch'] == 1 and first['steps'] == 437 and first['updates'] == 437,
        'bn437': all(v == 437 for v in first['bn_counts'].values()),
        'test2874': first['test']['objects'] == 2874,
        'all50parts': len(first['test']['parts']) == 50,
        'train_regularizer_zero': first['train_reg'] == 0,
        'manifest_regularizer_off': not manifest['regularizers_active'],
        'scale5': manifest['supervision_scale'] == 5,
        'real_output_metadata': manifest['final_output_space'] == 'real_tangent_scores',
        'model_and_optimizer_finite': finite(e1['model']) and finite(e1['optimizer']),
        'no_failure_file': not (args.run / 'failure.json').exists(),
    }
    with (args.run / 'geometry.jsonl').open() as stream:
        geometry = json.loads(next(line for line in stream if line.strip()))
    q = geometry['geometry']['final_tangent']
    checks['q_has_no_ball_radius_or_boundary'] = not any(k in q for k in
        ('near_boundary_fraction', 'radius', 'depth', 'projection_radius'))
    checks['q_finite'] = q['finite']
    checks['no_m3_ball_node'] = 'm3' not in geometry['geometry']
    checks['zero_r_gradients'] = all(
        not d['loss_requires_grad'] for name, d in geometry['gradients'].items() if name != 'ce')
    out = {'checks': checks, 'all_passed': all(checks.values()),
           'commit': manifest['source']['commit'], 'display_name': manifest['display_name'],
           'first_epoch_instance_miou': first['test']['instance_miou'],
           'first_epoch_class_miou': first['test']['class_miou'],
           'first_epoch_seconds': first['seconds'], 'first_input_sha256': first['first_input_sha256'],
           'initial_model_sha256': manifest['initial_model_sha256'],
           'geometry_nodes': list(geometry['geometry']), 'q_summary': q,
           'current_status': read(args.run / 'status.json')}
    # Compact status omits sampled object/part IDs.
    out['current_status'] = {k:v for k,v in out['current_status'].items()
                             if k in ('state','epoch','step','updates','at')}
    print(json.dumps(out))
    if not out['all_passed']:
        raise AssertionError([k for k,v in checks.items() if not v])


if __name__ == '__main__':
    main()
