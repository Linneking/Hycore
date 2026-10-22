"""Read-only CPU angular audit of existing caches and saved training forwards.

No encoder construction, optimizer update, dataset access, or test forward.
Input paths remain in the private run spec; outputs contain anonymous aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry_math import pair_midpoint_depth, triple_center_from_gram

C, H = 0.1, 4.6
QUANTILES = np.array([0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 1e-14):
        raise ValueError('Undefined direction at zero norm')
    return x / norm


def pair_depth(cosine, c=C, h=H):
    return pair_midpoint_depth(cosine, c=c, depth=h)


def triple_depth(directions, c=C, h=H):
    """Exact equal-shell hyperbolic minimax center, via convex-hull projection.

    Directions shape [N,3,D]. The closest point to the origin on their Euclidean
    convex hull determines the spherical cap and the absolute hyperbolic1center.
    Enumerate vertices/edges/full-simplex stationary point; singular full faces
    already have an optimal representation on an edge (affine dependence).
    """
    n = np.asarray(directions, np.float64)
    gram = np.einsum('nid,njd->nij', n, n)
    certificate = triple_center_from_gram(gram, c=c, depth=h)
    if np.max(certificate['projection_residual'], initial=0) > 1e-7:
        raise RuntimeError('Triple center failed projection optimality certificate')
    return certificate['depth']


def stats(values, weights=None):
    values = np.asarray(values, np.float64).ravel()
    if not len(values):
        return {'count': 0}
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, np.float64)
    weights /= weights.sum()
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    quantiles = np.interp(QUANTILES, np.cumsum(sorted_weights), sorted_values)
    return {'count': len(values), 'mean': float(weights @ values),
            'std': float(np.sqrt(weights @ ((values - weights @ values) ** 2))),
            'quantiles': {f'q{round(q*100):02d}': float(v) for q, v in zip(QUANTILES, quantiles)}}


def cosine_summary(cosines, weights=None):
    cosines = np.clip(cosines, -1, 1)
    angles = np.degrees(np.arccos(cosines))
    depths = pair_depth(cosines)
    if weights is None:
        weights = np.ones(len(cosines))
    weights = weights / weights.sum()
    grid = np.linspace(0, 180, 181)
    return {'cosine': stats(cosines, weights), 'angle_deg': stats(angles, weights),
            'pair_midpoint_depth_shell': stats(depths, weights),
            'pair_midpoint_deeper_than_init4_14_fraction': float(weights @ (depths > 4.14)),
            'angle_cdf': {'x_deg': grid.tolist(),
                          'y': [float(weights @ (angles <= v)) for v in grid]}}


def class_geometry(mu, labels, seed, car_class=None):
    directions, labels = unit(mu), np.asarray(labels).ravel()
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    by_class, within, within_weight, centroids = [], [], [], []
    for label in classes:
        subset = directions[labels == label]
        if len(subset) < 2:
            continue
        gram = subset @ subset.T
        cosines = gram[np.triu_indices(len(subset), 1)]
        within.append(cosines)
        within_weight.append(np.full(len(cosines), 1 / len(cosines)))
        result = cosine_summary(cosines)
        result.update(class_label=int(label), objects=len(subset))
        by_class.append(result)
        centroids.append(unit(subset.mean(axis=0)))
    all_cos = np.concatenate(within)
    macro_weights = np.concatenate(within_weight)
    between = []
    # Equal32 samples per unordered class pair; every class pair has equal mass.
    for i, first in enumerate(classes):
        ids_i = np.flatnonzero(labels == first)
        for second in classes[i + 1:]:
            ids_j = np.flatnonzero(labels == second)
            pick_i = rng.choice(ids_i, 32)
            pick_j = rng.choice(ids_j, 32)
            between.append(np.einsum('nd,nd->n', directions[pick_i], directions[pick_j]))
    centroid_gram = np.stack(centroids) @ np.stack(centroids).T
    centroid_cos = centroid_gram[np.triu_indices(len(centroids), 1)]
    result = {'objects': len(labels), 'classes': len(classes), 'dimension': directions.shape[1],
              'within_macro': cosine_summary(all_cos, macro_weights),
              'within_micro': cosine_summary(all_cos),
              'between_class_pair_balanced': cosine_summary(np.concatenate(between)),
              'between_class_centroids': cosine_summary(centroid_cos),
              'per_class': by_class}
    if car_class is not None:
        result['modelnet40_car_class'] = next((r for r in by_class if r['class_label'] == car_class), None)
    # Fixed supervised reference, independently sampled from actual reciprocal mining.
    triples = []
    for label in classes:
        ids = np.flatnonzero(labels == label)
        if len(ids) < 2:
            continue
        for _ in range(64):
            i, j = rng.choice(ids, 2, replace=False)
            other = rng.choice(classes[classes != label])
            k = rng.choice(np.flatnonzero(labels == other))
            triples.append((i, j, k))
    result['sameclass_pair_otherclass_k_reference'] = triplet_geometry(
        directions, labels, np.asarray(triples, np.int64))
    return result


def triplet_geometry(directions, labels, triplets, proxy_ball=None, details=None):
    triplets = np.asarray(triplets, np.int64)
    if not len(triplets):
        return {'count': 0}
    n = directions[triplets]
    cos_ij = np.einsum('nd,nd->n', n[:, 0], n[:, 1])
    cos_ik = np.einsum('nd,nd->n', n[:, 0], n[:, 2])
    cos_jk = np.einsum('nd,nd->n', n[:, 1], n[:, 2])
    pair, triple = pair_depth(cos_ij), triple_depth(n)
    y = np.asarray(labels)[triplets]
    result = {'count': len(triplets), 'ij_angle_deg': stats(np.degrees(np.arccos(np.clip(cos_ij,-1,1)))),
              'ik_angle_deg': stats(np.degrees(np.arccos(np.clip(cos_ik,-1,1)))),
              'jk_angle_deg': stats(np.degrees(np.arccos(np.clip(cos_jk,-1,1)))),
              'pair_midpoint_depth_shell': stats(pair), 'triple_absolute_minimax_depth_shell': stats(triple),
              'pair_minus_triple_depth': stats(pair - triple),
              'ij_same_class_fraction': float(np.mean(y[:,0] == y[:,1])),
              'ij_same_k_other_fraction': float(np.mean((y[:,0] == y[:,1]) & (y[:,0] != y[:,2]))),
              'all_distinct_classes_fraction': float(np.mean((y[:,0] != y[:,1]) & (y[:,0] != y[:,2]) & (y[:,1] != y[:,2]))),
              'pair_midpoint_deeper_than_init4_14_fraction': float(np.mean(pair > 4.14)),
              'triple_center_deeper_than_init4_14_fraction': float(np.mean(triple > 4.14))}
    if proxy_ball is not None:
        norm = np.linalg.norm(proxy_ball, axis=-1)
        depth = 2 * np.arctanh(np.sqrt(C) * norm) / np.sqrt(C)
        a, b = np.asarray(details['pair_proxy_idx']), np.asarray(details['triple_proxy_idx'])
        result.update(selected_pair_proxy_depth=stats(depth[a]), selected_triple_proxy_depth=stats(depth[b]),
                      same_pair_triple_proxy_fraction=float(np.mean(a == b)))
    return result


def analyze_npz(record, seed):
    path = Path(record['path'])
    before = sha(path)
    with np.load(path, allow_pickle=False) as saved:
        feature_key = record.get('feature_key', 'mu')
        data = {key: saved[key] for key in saved.files}
    labels = data['labels']
    result = class_geometry(data[feature_key], labels, seed, record.get('car_class'))
    result.update(key=record['key'], source_sha256=before, source_kind='existing_frozen_cache',
                  mode=record['mode'], feature_key=feature_key,
                  label_min=int(labels.min()), label_max=int(labels.max()),
                  shell_comparison={'c': C, 'whole_depth': H, 'initial_proxy_depth': 4.14})
    if 'metadata_json' in data:
        meta = json.loads(str(data['metadata_json']))
        result['provenance'] = {k: meta[k] for k in ('checkpoint_hash', 'checkpoint_epoch', 'split',
                                                  'classes_in_split', 'mapping_config', 'source_cache_sha256') if k in meta}
    if sha(path) != before:
        raise RuntimeError('Source cache changed')
    return result


def analyze_snapshot(record, seed):
    import torch
    torch.set_num_threads(2)
    path = Path(record['path'])
    before = sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    forward = payload['same_forward']
    mu, labels = forward['hier_mu'].numpy(), forward['labels'].numpy()
    result = class_geometry(mu, labels, seed, record.get('car_class'))
    triplets = forward['sample_ghhc']['triplets'].numpy()
    details = {k: v.numpy() for k, v in forward['sample_ghhc']['details'].items()}
    result.update(key=record['key'], source_sha256=before, source_kind='71_diagnostic_actual_training_forward',
                  mode=forward['stage'], epoch=int(payload['epoch']), step=int(payload['step']),
                  actual_mined_triplets=triplet_geometry(unit(mu), labels, triplets,
                    forward['proxy_ball_before'].numpy(), details),
                  global_batch=len(mu), whole_points=int(forward['whole_count']),
                  shell_comparison={'c': C, 'whole_depth': H, 'initial_proxy_depth': 4.14})
    # Actual reciprocal sets are often crossclass. Keep requested subset explicit.
    y = labels[triplets]
    mask = (y[:,0] == y[:,1]) & (y[:,0] != y[:,2])
    result['actual_mined_ij_same_k_other'] = triplet_geometry(unit(mu), labels, triplets[mask])
    if sha(path) != before:
        raise RuntimeError('Source snapshot changed')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    observations = []
    for record in spec['observations']:
        analyze = analyze_snapshot if record.get('kind') == 'snapshot' else analyze_npz
        result = analyze(record, spec.get('seed', 20261022))
        observations.append(result)
        (out / (record['key'] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'key': result['key'], 'objects': result['objects'], 'classes': result['classes'],
                          'within_angle_median': result['within_macro']['angle_deg']['quantiles']['q50'],
                          'within_midpoint_median': result['within_macro']['pair_midpoint_depth_shell']['quantiles']['q50']}, ensure_ascii=False), flush=True)
    summary = {'protocol': {'seed': spec.get('seed', 20261022), 'class_balanced_within': 'all pairs within each class; equal total mass per class',
                'between': '32 random sample pairs for every unordered class pair; equal mass per class pair',
                'triple_reference': '64 sameclass nonself pairs per class with uniformly sampled other class k',
                'triple_center': 'absolute hyperbolic minimax1center on common c=.1, depth4.6 shell; not a HIER-loss optimum',
                'encoder_or_optimizer_updates': 0, 'dataset_or_test_forward': 0},
               'observations': observations}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
