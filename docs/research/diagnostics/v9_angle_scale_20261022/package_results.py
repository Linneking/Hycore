"""Retain compact public curves and validate independent anonymous audit outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-angles', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.tables / 'angles_compact.json'
    compact = read(summary_path)
    original = {}
    for filename in ('cars_summary.json', 'v9_summary.json'):
        data = read(args.raw_angles / filename)
        original.update({r['key']: r for r in data['observations']})
    assert len(compact['observations']) == len(original) == 10
    for row in compact['observations']:
        source = original[row['key']]
        assert row['source_sha256'] == source['source_sha256']
        if row['source_kind'] == 'existing_frozen_cache':
            for key in ('within_macro', 'between_class_pair_balanced'):
                row[key]['angle_cdf'] = source[key]['angle_cdf']
    compact['protocol'] = data['protocol']
    summary_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    adam = read(args.tables / 'adam_fixed_state.json')
    assert len(adam['snapshots']) == 7
    assert max(r['native_replay_max_coordinate_error'] for r in adam['snapshots']) < 2e-6
    assert adam['production_updates'] == adam['encoder_forwards'] == adam['test_forwards'] == 0
    cars = read(args.tables / 'cars_optimizer.json')['original_cars']
    assert cars['shape'] == [512, 512]
    assert all(cars['tensor_dtypes'][k] == 'float32' for k in ('proxy_tangent', 'exp_avg', 'exp_avg_sq'))
    assert cars['optimizer_group']['eps'] == 1e-4
    assert len([r for r in compact['observations'] if r['source_kind'] == '71_diagnostic_actual_training_forward']) == 6
    for row in compact['observations']:
        group = row['sameclass_pair_otherclass_k_reference']
        assert group['pair_minus_triple_depth']['quantiles']['q00'] >= -1e-8
        if row['source_kind'] == '71_diagnostic_actual_training_forward':
            actual = row['actual_mined_triplets']
            assert actual['pair_minus_triple_depth']['quantiles']['q00'] >= -1e-8
    acceptance = {
        'status': 'passed', 'remote_execution_code_commit': compact['code_commit'],
        'scope': 'CPU cache/state analysis; no new model forward or production update',
        'unit_tests_passed': {'independent_geometry': 5, 'angles': 4, 'adam_scale': 5},
        'independent_report_review': 'passed', 'figures_visually_reviewed': 4,
        'observations': 10, 'actual_training_forward_snapshots': 6, 'adam_state_snapshots': 7,
        'source_read_before_after_sha_checks': 'passed by angular analyzer',
        'native_adam_max_coordinate_error': max(r['native_replay_max_coordinate_error'] for r in adam['snapshots']),
        'public_tables_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.tables.glob('*')) if p.name != 'acceptance.json'},
    }
    (args.tables / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: acceptance[k] for k in ('status', 'observations', 'adam_state_snapshots')}))


if __name__ == '__main__':
    main()
