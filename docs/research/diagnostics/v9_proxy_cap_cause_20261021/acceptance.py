"""Export anonymous completion/identity checks for the bounded V9 observation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.run_dir / 'manifest.json')
    metrics = [read(p) for p in sorted(args.run_dir.glob('metrics_epoch_*.json'))]
    summaries = list(rows(args.run_dir / 'proxy_cause_steps.jsonl'))
    ordinary = list(rows(args.run_dir / 'steps.jsonl'))
    snapshots = list(rows(args.run_dir / 'proxy_cause_snapshots.jsonl'))
    checks = {
        'completed_ten_epochs': manifest['status'] == 'completed' and manifest['completed_epochs'] == 10,
        'two_thousand_updates': len(summaries) == len(ordinary) == 2000,
        'original_scheduler_horizon': manifest['training_config']['epochs'] == 300,
        'training_only': not manifest['official_test_read'] and manifest['selection_size'] == 0,
        'no_model_selection': not manifest['test_used_for_selection'] and manifest['best'] is None,
        'proxy_replicas_equal': all(r['proxy_replica_max_difference'] == 0 for r in ordinary),
        'shell_depth_fixed': all(abs(r['hier_whole_depth'] - 4.6) < 1e-4 for r in ordinary),
        'epoch_lengths': len(metrics) == 10 and all(sum(v['epoch'] == i for v in summaries) == 200 for i in range(1, 11)),
        'eighty_sparse_snapshots': len(snapshots) == 80,
        'snapshot_hashes': all(digest(args.run_dir / r['file']) == r['sha256'] for r in snapshots),
        'last_checkpoint_identity': digest(args.run_dir / 'last.pth') == metrics[-1]['checkpoint']['last_sha256'],
    }
    output = {
        'checks': checks, 'all_passed': all(checks.values()),
        'production_source_commit': manifest['commit'], 'branch': manifest['branch'],
        'started_utc': manifest['started_utc'], 'finished_utc': manifest['finished_utc'],
        'wall_seconds': manifest['wall_seconds'], 'seed': manifest['args']['seed'],
        'initialized_model_sha256': manifest['initialized_model_sha256'],
        'initialization': manifest['initialization'], 'training_size': manifest['train_size'],
        'training_labels_sha256': manifest['train_labels_sha256'],
        'runtime': manifest['runtime'], 'physical_gpu_indices': manifest['physical_gpus'],
        'gpu_models': [r['name'] for r in manifest['rank_devices']],
        'diagnostic_source': manifest['diagnostic_source'],
        'last_checkpoint_sha256': metrics[-1]['checkpoint']['last_sha256'],
        'sparse_snapshot_count': len(snapshots),
        'accounting_max_coordinate_error': max(r['all']['prediction_error_max_coordinate']['max'] for r in summaries),
        'component_gradient_sum_max_error': None,
        'first_step_crop': {k: ordinary[0][k] for k in ('whole_count', 'child_count')},
        'first_step_ids_sha256': hashlib.sha256(json.dumps(ordinary[0]['ids'], separators=(',', ':')).encode()).hexdigest(),
        'historical_replay_claim': False,
        'scope': 'Fresh bounded mechanism reproduction; no official test/evaluation/model selection; not bitwise historical replay',
        'diagnostic_code_limit': 'Parameter-gradient audit forwards disabled; same-graph proxy observation enabled',
    }
    output['diagnostic_source'].pop('original_relative_path', None)
    if not output['all_passed']:
        raise RuntimeError(json.dumps(checks))
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'all_passed': True, 'checks': len(checks), 'epochs': len(metrics), 'updates': len(summaries)}))


if __name__ == '__main__':
    main()
