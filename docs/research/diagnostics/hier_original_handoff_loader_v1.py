"""Validate supplied HIER best checkpoints/caches and normalize private audit inputs.

This is a CPU tensor reader, not an image forward reproduction of the old runtime.
Never modifies input files. Complete identities and converted arrays stay private.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


EXPECTED = {'cub': {'train': 5864, 'eval': 5924, 'classes': 100, 'epoch': 17},
            'cars': {'train': 8054, 'eval': 8131, 'classes': 98, 'epoch': 45},
            'sop': {'train': 59551, 'eval': 60502, 'classes': 11318, 'epoch': 96}}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def scalar_json(value):
    if isinstance(value, dict):
        return {str(k): scalar_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_checkpoint(path):
    # Only known data containers observed in the supplied tensor checkpoints.
    allowed = [argparse.Namespace, np.core.multiarray.scalar, np.dtype,
               type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def v7_map(tangent, c):
    tangent = np.asarray(tangent, dtype=np.float64)
    norm = np.linalg.norm(tangent, axis=1)
    unit = tangent / np.maximum(norm[:, None], 1e-15)
    rho = np.minimum(np.tanh(np.sqrt(c) * norm), .999)
    return unit * (rho / np.sqrt(c))[:, None]


def validate_array(points, c, cap=None):
    points = np.asarray(points)
    assert points.ndim == 2 and np.isfinite(points).all()
    rho = np.sqrt(c) * np.linalg.norm(points.astype(np.float64), axis=1)
    assert np.all((rho > 0) & (rho < 1)), 'Invalid ball coordinate/direction.'
    if cap is not None:
        assert rho.max() < np.tanh(np.sqrt(c) * cap) + 2e-6, 'Declared cap violated.'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--handoff', required=True)
    ap.add_argument('--v7-root', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    root, output, v7root = map(Path, (args.handoff, args.output, args.v7_root))
    if output.exists():
        raise FileExistsError('Use a fresh conversion output directory.')
    output.mkdir(parents=True)
    converted = output / 'converted'
    converted.mkdir()
    handed = root / 'proxy_eval/output/handoff'
    cache_hashes = dict((line.split()[1], line.split()[0]) for line in
                        (handed / 'sha256_caches.txt').read_text().splitlines() if line.strip())
    checkpoint_hashes = dict((line.split()[1], line.split()[0]) for line in
                             (handed / 'sha256_checkpoints.txt').read_text().splitlines() if line.strip())
    checkpoint_records, sources, model_records, augmentation_groups = {}, [], [], []
    for ds, expected in EXPECTED.items():
        relative = f'logs/run_{ds}_r50/checkpoint_best.pth'
        cp = root / relative
        cp_hash = digest(cp)
        assert cp_hash == checkpoint_hashes[relative], f'Checkpoint SHA mismatch: {ds}'
        ck = safe_checkpoint(cp)
        lcas = ck['cluster_loss']['lcas'].detach().cpu().numpy()
        state = ck['stduent']
        effective = scalar_json(vars(ck['args']))
        assert lcas.shape == (512, 512)
        assert int(ck['epoch']) == expected['epoch']
        assert effective['bn_freeze'] is True
        bn = sum(k.endswith('running_mean') for k in state)
        assert bn == 105 and len(state) == 632
        checkpoint_records[ds] = {'sha256': cp_hash, 'epoch': int(ck['epoch']),
                                  'effective_args': effective, 'bn_running_mean_count': bn,
                                  'student_state_tensors': len(state),
                                  'hierarchy_proxy_tangent_shape': list(lcas.shape),
                                  'PA_classification_proxy_saved': False}
        group = []
        for variant in ('train', 'eval', 'train_aug1', 'train_aug2'):
            name = f'{ds}_{variant}'
            path = root / f'proxy_eval/cache/{name}.pt'
            before = digest(path)
            assert before == cache_hashes[path.name], f'Cache SHA mismatch: {name}'
            cache = torch.load(path, map_location='cpu', weights_only=True)
            mu = cache['embeddings'].numpy()
            proxy = cache['hierarchy_proxies'].numpy()
            ids, labels, views = (cache[k].numpy() for k in ('sample_id', 'label', 'view_id'))
            split = 'eval' if variant == 'eval' else 'train'
            n = expected[split]
            assert mu.shape == (n, 512) and proxy.shape == (512, 512)
            assert np.unique(ids).size == n and np.array_equal(np.sort(ids), np.arange(n))
            assert len(labels) == len(views) == len(cache['image_paths']) == n
            assert float(cache['c']) == .1 and cache['D'] == 512 and cache['P'] == 512
            assert cache['checkpoint_hash'] == cp_hash and cache['checkpoint_epoch'] == expected['epoch']
            assert cache['code_commit'] == '3986a744a1a54fd357e307d1cb3f2e81910b9ffc'
            assert cache['mapping_config']['clip_r'] == 2.3
            validate_array(mu, .1, 2.3)
            validate_array(proxy, .1, 2.3)
            mapping = list(csv.DictReader((handed / f'sample_id_map_{ds}_{split}.csv')
                                          .open(encoding='utf-8-sig', newline='')))
            by_id = {int(r['sample_id']): r for r in mapping}
            assert len(by_id) == n
            for i, sid in enumerate(ids):
                row = by_id[int(sid)]
                assert int(row['label']) == int(labels[i]) and row['image_path'] == cache['image_paths'][i]
            meta = {k: scalar_json(cache[k]) for k in cache if k not in
                    ('sample_id', 'label', 'view_id', 'embeddings', 'hierarchy_proxies', 'image_paths')}
            # Suppress private server path from downstream report metadata.
            meta.pop('checkpoint_path', None)
            meta.update(source_cache_sha256=before, effective_bn_freeze=True,
                        exported_mode='eval', image_pool_complete=True,
                        audit_forward_reproduced=False, variant=variant,
                        checkpoint_source_sha256=cp_hash)
            target = converted / f'{name}.npz'
            np.savez(target, mu=mu, proxy_ball=proxy, proxy_tangent=lcas,
                     labels=labels, sample_ids=ids, view_ids=views, c=np.array(.1),
                     metadata_json=np.array(json.dumps(meta, ensure_ascii=False)))
            assert digest(path) == before, 'Source modified during conversion.'
            record = {'name': f'HIER_{name}', 'kind': 'original_HIER', 'cache': str(target),
                      'original_dataset': ds, 'split': split,
                      'old_proxy_stats_csv': str(root / f'proxy_eval/output/{ds}_{split}/proxy_stats.csv')}
            if variant in ('train', 'eval'):
                model_records.append(record)
            if variant != 'eval':
                group.append({'variant': variant, 'cache': str(target)})
            sources.append({'name': name, 'sha256': before, 'N': n, 'D': 512, 'P': 512,
                            'classes': int(np.unique(labels).size), 'converted': str(target),
                            'identity_rows_verified': n, 'views': np.unique(views).tolist()})
            print(json.dumps({'converted': name, 'N': n, 'SHA': 'verified'}, ensure_ascii=False), flush=True)
            del cache, mu, proxy
        augmentation_groups.append({'dataset': ds, 'caches': group})
        assert digest(cp) == cp_hash, 'Checkpoint changed during conversion.'
        del ck, state, lcas
    for epoch in (5, 187, 300):
        src = v7root / f'data/V7/e{epoch:03d}_whole_cache.npz'
        before = digest(src)
        with np.load(src, allow_pickle=False) as z:
            mu, tangent = z['mu'], z['proxy_tangent']
            c = float(np.asarray(z['c']).ravel()[0])
            meta = json.loads(str(np.asarray(z['metadata_json']).item()))
            ids, labels = z['sample_ids'], z['labels']
            proxy = v7_map(tangent, c)
            target = converted / f'V7_e{epoch:03d}.npz'
            np.savez(target, mu=mu, proxy_ball=proxy, proxy_tangent=tangent,
                     labels=labels, sample_ids=ids, view_ids=np.zeros(len(mu), np.int8), c=np.array(c),
                     metadata_json=np.array(json.dumps({'source_cache_sha256': before, 'variant': f'e{epoch}',
                                                        'source_metadata': meta,
                                                        'proxy_mapping': 'expmap0; numeric_rho_limit=.999; no forward tangent cap',
                                                        'proxy_post_step_tangent_cap': 3., 'epoch': epoch,
                                                        'exported_mode': 'eval', 'split': 'train'}, ensure_ascii=False)))
        assert digest(src) == before
        model_records.append({'name': f'V7_e{epoch:03d}', 'kind': 'HyCoRe_V7', 'split': 'train', 'cache': str(target)})
        sources.append({'name': f'V7_e{epoch:03d}', 'sha256': before, 'N': len(mu), 'D': mu.shape[1], 'P': len(proxy)})
    spec = {'models': model_records, 'augmentation_groups': augmentation_groups,
            'protocol': 'Frozen CPU tensor geometry; original best only; no independent image forward.'}
    (output / 'spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'runtime': {'python': __import__('sys').version, 'torch': torch.__version__, 'numpy': np.__version__,
                             'device': 'CPU'}, 'checkpoints': checkpoint_records, 'sources': sources,
                'verified_source_files': len(sources), 'complete_original_caches': 12,
                'dataset_download_required': False,
                'limitations': ['Only three supplied best checkpoints; no training trajectory.',
                                'Image datasets absent; cache export model forward not independently re-run.',
                                'Supplied exporter differs from working generator (out referenced before creation).',
                                'SOP CLI False is effective True under original argparse type=bool.',
                                'Frozen ancestor draws are not actual training usage.']}
    (output / 'loader_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
