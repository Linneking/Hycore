"""Read existing segmentation results; never load models or modify a run.

Usage: python summarize_runs.py NAME=RUN_DIRECTORY [...]
Outputs only derived metrics and provenance hashes, without server paths/IDs.
"""
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metric(m, parts=False):
    keys = ('instance_miou', 'class_miou', 'point_accuracy', 'objects',
            'classes_present', 'ce')
    out = {key: m[key] for key in keys if key in m}
    out['per_class'] = m['per_class']
    if parts and 'parts' in m:
        out['parts'] = m['parts']
        out['zero_recall_parts'] = [int(k) for k, v in m['parts'].items()
                                    if v['gt'] and v['tp'] == 0]
    if 'masked_secondary' in m:
        out['masked_secondary'] = {key: m['masked_secondary'][key]
                                    for key in keys if key in m['masked_secondary']}
    return out


def summarize(path, verify_checkpoint=False):
    rows = [json.loads(s) for s in (path / 'metrics.jsonl').read_text().splitlines()
            if s.strip()]
    manifest = json.loads((path / 'manifest.json').read_text())
    status = json.loads((path / 'status.json').read_text())
    best = json.loads((path / 'best_selection.json').read_text())
    selected = int(best['selected_epoch'])
    row = next(r for r in rows if r['epoch'] == selected)
    final = json.loads((path / 'final_test.json').read_text())
    tail = rows[-20:]
    keep_config = ('arm', 'protocol', 'epochs', 'schedule_epochs', 'batch_size',
                   'num_points', 'k', 'lr', 'min_lr', 'momentum', 'weight_decay',
                   'seed', 'alpha', 'beta', 'supervision_scale', 'regularizers')
    out = {
        'status': status, 'source_commit': manifest.get('source', {}).get('commit'),
        'config': {k: v for k, v in manifest['config'].items() if k in keep_config},
        'initial_model_sha256': manifest.get('initial_model_sha256'),
        'data_manifest_sha256': manifest.get('data_manifest_sha256'),
        'metrics_sha256': digest(path / 'metrics.jsonl'),
        'selection': manifest.get('selection'),
        'best_epoch': selected, 'best': metric(row['test'], True),
        'last_epoch': rows[-1]['epoch'], 'last': metric(rows[-1]['test'], True),
        'checks': {
            'epochs_1_to_200': [r['epoch'] for r in rows] == list(range(1, 201)),
            'all_437_updates': all(r['steps'] == 437 for r in rows),
            'last_updates_87400': rows[-1]['updates'] == 87400,
            'all_test_objects_2874': all(r['test']['objects'] == 2874 for r in rows),
            'all_model_finite': all(r.get('model_finite', True) for r in rows),
            'all_optimizer_finite': all(r.get('optimizer_finite', True) for r in rows),
            'best_selection_matches_epoch': best['metrics'] == row['test'],
            'final_test_matches_best': final['selected_epoch'] == selected
                and final['metrics'] == row['test'],
            'all_last_bn_counts_87400': all(v == 87400 for v in rows[-1].get('bn_counts', {}).values()),
        },
        'window181_200': {
            k: {'mean': statistics.mean(r['test'][k] for r in tail),
                'std_population': statistics.pstdev(r['test'][k] for r in tail),
                'min': min(r['test'][k] for r in tail),
                'max': max(r['test'][k] for r in tail)}
            for k in ('class_miou', 'instance_miou')},
        'window_per_class': {
            name: statistics.mean(r['test']['per_class'][name]['miou'] for r in tail)
            for name in rows[-1]['test']['per_class']},
        'curve': [{k: r[k] for k in ('epoch', 'train_loss', 'train_ce', 'train_reg')
                   if k in r} | {k: r['test'][k] for k in ('class_miou', 'instance_miou', 'ce')}
                  for r in rows],
        'selected_part_snapshots': {
            str(r['epoch']): {k: v for k, v in r['test'].get('parts', {}).items()
                              if int(k) in (4, 18, 31, 33, 34, 42, 43, 46)}
            for r in rows if r['epoch'] in (10, 40, 100, 150, 180, 200)},
    }
    if verify_checkpoint:
        out['best_checkpoint_sha256'] = digest(path / 'best.pth')
        out['last_checkpoint_sha256'] = digest(path / 'last.pth')
        out['checks']['best_checkpoint_matches_final_identity'] = (
            out['best_checkpoint_sha256'] == final['checkpoint_sha256'])
    return out


def main(arguments):
    out = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'models': {}}
    for arg in arguments:
        name, path = arg.split('=', 1)
        out['models'][name] = summarize(Path(path), name.startswith('S5'))
    print(json.dumps(out, ensure_ascii=True))


if __name__ == '__main__':
    main(sys.argv[1:])
