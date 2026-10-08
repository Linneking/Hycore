"""CPU-only frozen whole/proxy geometry with explicit curvature and pool identity.

Input NPZ fields: mu, proxy_ball, labels, sample_ids, c, metadata_json; optional
proxy_tangent/view_ids/proxy_ids. --spec is {"models": [{"name", "cache", ...}]}.
Outputs contain complete internal sample/proxy mappings and must stay private.
No model forward, training reconstruction, dimensional reduction, or GPU use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    good = values[np.isfinite(values)]
    if not good.size:
        return {'n': int(values.size), 'finite_n': 0}
    q = np.percentile(good, QUANTILES)
    result = {'n': int(values.size), 'finite_n': int(good.size),
              'mean': float(good.mean()), 'std_population': float(good.std()),
              'variance_population': float(good.var()), 'iqr': float(q[5] - q[3])}
    result.update({f'p{p:02d}': float(v) for p, v in zip(QUANTILES, q)})
    return result


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    return x / np.maximum(norms[:, None], 1e-15), norms


def radial_coordinates(x, c):
    if not np.isfinite(c) or c <= 0:
        raise ValueError('This audit requires a finite positive curvature magnitude c.')
    r = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)
    rho = math.sqrt(c) * r
    if not np.all(np.isfinite(rho)) or np.any(rho >= 1):
        raise ValueError('Non-finite coordinates or points outside the open Poincare ball.')
    q = 2 * np.arctanh(rho)
    return {'r': r, 'rho': rho, 'd0': q / math.sqrt(c), 'q': q}


def stable_distance_matrix(p, x, c):
    """Float64 exact geodesic distance; accepts arbitrary positive c."""
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    p2 = np.einsum('ij,ij->i', p, p)
    x2 = np.einsum('ij,ij->i', x, x)
    delta2 = np.maximum(p2[:, None] + x2[None, :] - 2 * (p @ x.T), 0)
    denom = (1 - c * p2[:, None]) * (1 - c * x2[None, :])
    if np.any(denom <= 0):
        raise ValueError('Distance input outside ball.')
    score = 2 * c * delta2 / denom
    # arcosh(1+t) = 2 asinh(sqrt(t/2)), avoiding cancellation near t=0.
    return 2 * np.arcsinh(np.sqrt(score / 2)) / math.sqrt(c)


def stable_topk(values, ids, k):
    """Increasing value, then increasing stable sample ID; handles cutoff ties."""
    values = np.asarray(values)
    ids = np.asarray(ids)
    if values.ndim != 2 or values.shape[1] != len(ids):
        raise ValueError('Top-k shape mismatch.')
    k = min(int(k), len(ids))
    partial = np.argpartition(values, k - 1, axis=1)[:, :k]
    result = np.empty((len(values), k), dtype=np.int64)
    for row, candidate in enumerate(partial):
        threshold = values[row, candidate].max()
        # All tied cutoff values must participate; argpartition alone is unstable.
        candidate = np.flatnonzero(values[row] <= threshold)
        order = np.lexsort((ids[candidate], values[row, candidate]))
        result[row] = candidate[order[:k]]
    return result


def direction_resultant(u):
    u = np.asarray(u, dtype=np.float64)
    n = len(u)
    r = float(np.linalg.norm(u.mean(axis=0)))
    # Exact average over distinct unordered pairs of unit vectors.
    pair_cos = (n * r * r - 1) / (n - 1) if n > 1 else None
    return {'n': n, 'R': r, 'mean_pairwise_cosine_excluding_self': pair_cos}


def class_geometry(u, labels, whole_radial, ids, loo_max_classes=200):
    """Macro exact cosine statistics without C^2 or N-by-C storage for SOP."""
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    sums = np.add.reduceat(u[order], starts, axis=0)
    means = sums / counts[:, None]
    centers, resultant = unit_rows(means)
    norms2 = np.einsum('ij,ij->i', means, means)
    within = np.full(len(classes), np.nan)
    valid = counts > 1
    within[valid] = (counts[valid] * norms2[valid] - 1) / (counts[valid] - 1)
    sum_means = means.sum(axis=0)
    between = ((float(sum_means @ sum_means) - float(norms2.sum())) /
               (len(classes) * (len(classes) - 1))) if len(classes) > 1 else None
    own_cos = np.empty(len(u))
    for start in range(0, len(u), 4096):
        stop = min(start + 4096, len(u))
        own_cos[start:stop] = np.einsum('ij,ij->i', u[start:stop], centers[inverse[start:stop]])
    own_angle = np.degrees(np.arccos(np.clip(own_cos, -1, 1)))
    own_angle[resultant[inverse] < 1e-12] = np.nan
    rows = []
    cone90 = np.empty(len(classes))
    radial05 = np.empty(len(classes))
    radial95 = np.empty(len(classes))
    radial50 = np.empty(len(classes))
    for ci, (label, start, n) in enumerate(zip(classes, starts, counts)):
        ix = order[start:start + n]
        angles = own_angle[ix]
        depth = whole_radial['d0'][ix]
        cone90[ci] = np.nanpercentile(angles, 90) if np.isfinite(angles).any() else np.nan
        radial05[ci], radial50[ci], radial95[ci] = np.percentile(depth, [5, 50, 95])
        rows.append({'class_label': int(label), 'n': int(n), 'direction_R': resultant[ci],
                     'within_cosine_excluding_self': within[ci],
                     'mean_own_center_angle_deg': float(np.nanmean(angles)) if np.isfinite(angles).any() else None,
                     'direction_cone90_deg': cone90[ci], 'd0_min': float(depth.min()),
                     'd0_p05': radial05[ci], 'd0_median': radial50[ci],
                     'd0_p95': radial95[ci], 'd0_max': float(depth.max()),
                     'd0_std': float(depth.std())})
    summary = {'class_count': int(len(classes)), 'within_class_cosine_macro': float(np.nanmean(within))
               if np.isfinite(within).any() else None, 'between_class_cosine_macro_pairs': between,
               'class_R': stats(resultant), 'class_cone90_deg': stats(cone90),
               'common_direction_sample_weighted': direction_resultant(u),
               'common_direction_class_balanced_R': float(np.linalg.norm(means.mean(axis=0)))}
    loo_pred = np.full(len(u), -1, dtype=np.int64)
    if len(classes) <= loo_max_classes and np.all(counts > 1) and np.all(resultant > 1e-12):
        for start in range(0, len(u), 1024):
            stop = min(start + 1024, len(u))
            ub = u[start:stop]
            inv = inverse[start:stop]
            score = ub @ centers.T
            own = sums[inv] - ub
            own /= np.maximum(np.linalg.norm(own, axis=1)[:, None], 1e-15)
            score[np.arange(len(ub)), inv] = np.einsum('ij,ij->i', ub, own)
            loo_pred[start:stop] = classes[np.argmax(score, axis=1)]
        correct = loo_pred == labels
        class_acc = np.bincount(inverse, weights=correct, minlength=len(classes)) / counts
        summary['loo_direction_accuracy_micro'] = float(correct.mean())
        summary['loo_direction_accuracy_macro'] = float(class_acc.mean())
        summary['loo_status'] = 'computed; own object excluded from its class center'
    else:
        summary['loo_status'] = 'not computed: class-count limit or singleton/undefined-center class'
    return summary, rows, {'classes': classes, 'inverse': inverse, 'counts': counts,
                           'means': means, 'centers': centers, 'R': resultant, 'cone90': cone90,
                           'd0_p05': radial05, 'd0_p95': radial95, 'd0_median': radial50,
                           'whole_own_center_angle_deg': own_angle, 'whole_loo_prediction': loo_pred}


def proxy_affinity(v, proxy_radial, cg, random_seeds=(1001, 1002, 1003, 1004, 1005)):
    centers = cg['centers']
    if not np.any(cg['R'] >= 1e-12):
        raise ValueError('All class direction centers are undefined.')
    scores = v @ centers.T
    scores[:, cg['R'] < 1e-12] = -np.inf
    closest = np.argmax(scores, axis=1)
    first = scores[np.arange(len(v)), closest]
    second = np.partition(scores, -2, axis=1)[:, -2] if scores.shape[1] > 1 else np.full(len(v), np.nan)
    angle = np.degrees(np.arccos(np.clip(first, -1, 1)))
    second_angle = np.degrees(np.arccos(np.clip(second, -1, 1)))
    cone = angle <= cg['cone90'][closest]
    radial = ((proxy_radial['d0'] >= cg['d0_p05'][closest]) &
              (proxy_radial['d0'] <= cg['d0_p95'][closest]))
    random = []
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        ru, _ = unit_rows(rng.normal(size=v.shape))
        rs = ru @ centers.T
        rs[:, cg['R'] < 1e-12] = -np.inf
        best = rs.max(axis=1)
        random.append({'seed': seed, 'angle_deg': stats(np.degrees(np.arccos(np.clip(best, -1, 1))))})
    summary = {'definition': 'argmax unit proxy dot normalized mean of whole unit directions; not a training label',
               'nearest_center_angle_deg': stats(angle), 'top2_cosine_margin': stats(first - second),
               'top2_angle_margin_deg': stats(second_angle - angle),
               'proxy_count_in_nearest_class_direction_cone90': int(cone.sum()),
               'proxy_count_in_nearest_class_d0_p05_p95': int(radial.sum()),
               'proxy_count_in_both_ranges': int((cone & radial).sum()),
               'relative_nearest_class_counts': {str(int(k)): int(n) for k, n in
                                                 zip(*np.unique(cg['classes'][closest], return_counts=True))},
               'random_direction_reference': random,
               'random_note': 'same D/P and fixed whole centers; angles depend on D and class count'}
    arrays = {'proxy_nearest_class': cg['classes'][closest], 'proxy_nearest_class_index': closest,
              'proxy_nearest_center_angle_deg': angle, 'proxy_nearest_center_cosine': first,
              'proxy_top2_cosine_margin': first - second, 'proxy_inside_direction_cone90': cone,
              'proxy_inside_class_radial90': radial}
    return summary, arrays


def retrieval(p, x, u, v, c, ids, labels, depths, k=16, proxy_chunk=32):
    """Full-pool exact NN; retain only top-k and fixed-P whole occupancy."""
    k = min(k, len(x))
    raw = np.empty((len(p), k), dtype=np.int64)
    directional = np.empty_like(raw)
    raw_values = np.empty((len(p), k))
    dir_values = np.empty_like(raw_values)
    sample_min_raw = np.full(len(x), np.inf)
    sample_min_dir = np.full(len(x), np.inf)
    assign_raw = np.full(len(x), -1, dtype=np.int64)
    assign_dir = np.full(len(x), -1, dtype=np.int64)
    for start in range(0, len(p), proxy_chunk):
        stop = min(start + proxy_chunk, len(p))
        distance = stable_distance_matrix(p[start:stop], x, c)
        rr = stable_topk(distance, ids, k)
        raw[start:stop] = rr
        raw_values[start:stop] = np.take_along_axis(distance, rr, axis=1)
        query = distance.argmin(axis=0)
        best = distance[query, np.arange(len(x))]
        mask = best < sample_min_raw
        sample_min_raw[mask] = best[mask]
        assign_raw[mask] = start + query[mask]
        direction = 1 - v[start:stop] @ u.T
        dr = stable_topk(direction, ids, k)
        directional[start:stop] = dr
        dir_values[start:stop] = np.take_along_axis(direction, dr, axis=1)
        query = direction.argmin(axis=0)
        best = direction[query, np.arange(len(x))]
        mask = best < sample_min_dir
        sample_min_dir[mask] = best[mask]
        assign_dir[mask] = start + query[mask]
    result = {}
    arrays = {'raw_top_index': raw, 'direction_top_index': directional,
              'raw_top_distance': raw_values, 'direction_top_cosine': 1 - dir_values,
              'whole_nearest_raw_proxy': assign_raw, 'whole_nearest_direction_proxy': assign_dir,
              'whole_nearest_raw_proxy_distance': sample_min_raw}
    low = np.lexsort((ids, depths))[:max(1, math.ceil(len(x) * .01))]
    low_mask = np.zeros(len(x), bool)
    low_mask[low] = True
    for name, rows, assigned in [('raw', raw, assign_raw), ('direction', directional, assign_dir)]:
        top4 = rows[:, :min(4, k)]
        slots = np.bincount(top4.ravel(), minlength=len(x))
        hottest_order = np.lexsort((ids, -slots))
        quartets = Counter(tuple(sorted(ids[r].tolist())) for r in top4)
        occupancy = np.bincount(assigned, minlength=len(p))
        frequency = occupancy / occupancy.sum()
        nz = frequency > 0
        entropy = -float(np.sum(frequency[nz] * np.log(frequency[nz])))
        purity = {}
        for requested_k in (4, 16):
            kk = min(requested_k, k)
            fractions = [np.unique(labels[r[:kk]], return_counts=True)[1].max() / kk for r in rows]
            arrays[f'{name}_proxy_purity{requested_k}'] = np.asarray(fractions)
            purity[f'purity{requested_k}'] = {**stats(fractions), 'actual_k': kk}
        result[name] = {'different_objects_top4': int((slots > 0).sum()),
                        'coverage_fraction_of_available_slots': float((slots > 0).sum() / top4.size),
                        'coverage_fraction_of_whole_pool': float((slots > 0).sum() / len(x)),
                        'different_quartet_sets': len(quartets), 'maximum_repeated_quartet': max(quartets.values()),
                        'top4_hot_object_slot_share': float(slots[hottest_order[:4]].sum() / top4.size),
                        'bottom_depth_1pct_slot_share': float(slots[low_mask].sum() / top4.size),
                        'slots': int(top4.size), 'pool_n': len(x), 'proxies': len(p), **purity,
                        'occupancy': {'definition': 'whole to nearest proxy; not training activation',
                                      'fixed_P': len(p), 'occupied_proxy_count': int((occupancy > 0).sum()),
                                      'zero_occupancy_fraction': float((occupancy == 0).mean()),
                                      'entropy': entropy, 'effective_number': math.exp(entropy),
                                      'largest_proxy_sample_share': float(frequency.max()), 'counts': stats(occupancy)}}
        arrays[f'{name}_whole_top4_slots'] = slots
        arrays[f'{name}_proxy_occupancy'] = occupancy
    # Exact geodesic NN ordering after whole-only uniform positive depth is cosine.
    result['whole_only_equal_depth'] = {
        'operation': 'keep proxy coordinates and whole directions; assign all whole the original median d0',
        'common_d0': float(np.median(depths)),
        'ranking_equals_direction_exactly': True,
        'note': 'analytical strict-geodesic equivalence for nonzero proxy/query; not an independent replicate'}
    return result, arrays


def compare_old_csv(path, raw_rows, ids):
    if not path:
        return {'status': 'not provided'}
    with open(path, encoding='utf-8-sig', newline='') as f:
        old = list(csv.DictReader(f))
    compare = []
    for row in old:
        proxy_id = int(row['proxy_id'])
        if not (0 <= proxy_id < len(raw_rows)):
            raise ValueError('Old CSV proxy ID out of range.')
        legacy = [int(row[f'top{i}_id']) for i in range(4)]
        current = ids[raw_rows[proxy_id, :4]].tolist()
        compare.append((legacy == current, set(legacy) == set(current), len(set(legacy) & set(current)) / 4))
    return {'status': 'computed', 'legacy_distance': 'released FP32 hyptorch including epsilon/clamp',
            'current_distance': 'Float64 strict geodesic', 'rows': len(compare),
            'ordered_top4_agreement': float(np.mean([v[0] for v in compare])),
            'set_top4_agreement': float(np.mean([v[1] for v in compare])),
            'mean_top4_member_retention': float(np.mean([v[2] for v in compare])),
            'legacy_csv_sha256': sha256(path),
            'tie_note': 'strict ties use sample ID; legacy torch.topk ties may be ordered differently'}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(finite_json(rows))


def analyze_model(record, destination):
    cache_path = Path(record['cache'])
    digest = sha256(cache_path)
    with np.load(cache_path, allow_pickle=False) as z:
        x = np.asarray(z['mu'], dtype=np.float64)
        p = np.asarray(z['proxy_ball'], dtype=np.float64)
        labels = np.asarray(z['labels'], dtype=np.int64)
        ids = np.asarray(z['sample_ids'], dtype=np.int64)
        c = float(np.asarray(z['c']).ravel()[0])
        tangent = np.asarray(z['proxy_tangent'], dtype=np.float64) if 'proxy_tangent' in z else None
        proxy_ids = np.asarray(z['proxy_ids'], dtype=np.int64) if 'proxy_ids' in z else np.arange(len(p))
        view_ids = np.asarray(z['view_ids'], dtype=np.int64) if 'view_ids' in z else np.zeros(len(x), np.int64)
        metadata = json.loads(str(np.asarray(z['metadata_json']).item())) if 'metadata_json' in z else {}
    if x.ndim != 2 or p.ndim != 2 or x.shape[1] != p.shape[1] or len(x) < 4:
        raise ValueError('Whole/proxy coordinate shape mismatch or insufficient whole pool.')
    if not (len(x) == len(labels) == len(ids) == len(view_ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError('Object identities are missing, duplicated, or misaligned.')
    if len(np.unique(proxy_ids)) != len(p):
        raise ValueError('Duplicate proxy identities.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError('Non-finite embedding.')
    # Fixed array positions and IDs are preserved; deterministic ties use identities.
    u, whole_norm = unit_rows(x)
    v, proxy_norm = unit_rows(p)
    if np.any(whole_norm < 1e-12) or np.any(proxy_norm < 1e-12):
        raise ValueError('Zero-norm embeddings have undefined directions; this comparison requires nonzero points.')
    whole_radial = radial_coordinates(x, c)
    proxy_radial = radial_coordinates(p, c)
    geometry, class_rows, cg = class_geometry(u, labels, whole_radial, ids)
    affinity, pa = proxy_affinity(v, proxy_radial, cg)
    neighbors, ra = retrieval(p, x, u, v, c, ids, labels, whole_radial['d0'])
    old = compare_old_csv(record.get('old_proxy_stats_csv'), ra['raw_top_index'], ids)
    identity = {'name': record['name'], 'kind': record.get('kind', 'unknown'),
                'split': record.get('split', metadata.get('split', 'unknown')),
                'cache_sha256': digest, 'c': c, 'D': x.shape[1], 'P': len(p), 'N': len(x),
                'metadata': metadata, 'spec_record': record}
    summary = {'identity': identity, 'radial': {'whole': {k: stats(a) for k, a in whole_radial.items()},
                                               'proxy': {k: stats(a) for k, a in proxy_radial.items()}},
               'whole_direction': geometry, 'proxy_direction': direction_resultant(v),
               'proxy_class_affinity': affinity, 'retrieval': neighbors, 'legacy_top4_comparison': old,
               'proxy_actual_training_usage': {'status': 'unknown; not inferred from frozen NN or argmin'},
               'definitions': {'r': 'Euclidean coordinate norm', 'rho': 'sqrt(c)*r',
                               'd0': 'full origin geodesic distance=2*atanh(rho)/sqrt(c)',
                               'q': 'dimensionless depth=sqrt(c)*d0',
                               'cone': 'direction cone covering 90% of class objects; not hyperbolic convex hull'}}
    if tangent is not None:
        summary['proxy_raw_tangent_norm'] = stats(np.linalg.norm(tangent, axis=1))
    cap = metadata.get('mapping_config', {}).get('clip_r') if isinstance(metadata.get('mapping_config'), dict) else None
    if cap is not None:
        cap = float(cap)
        summary['original_forward_cap'] = {'tangent_cap': cap, 'd0_cap': 2 * cap,
                                           'q_cap': 2 * cap * math.sqrt(c),
                                           'rho_cap': math.tanh(math.sqrt(c) * cap),
                                           'note': 'ideal cap; released implementation divides by norm+1e-5'}
        for domain, values in [('whole', whole_radial), ('proxy', proxy_radial)]:
            summary['radial'][domain]['fraction_within_1e_4_d0_of_cap'] = float(np.mean(values['d0'] >= 2 * cap - 1e-4))
    arrays = {'sample_ids': ids, 'labels': labels, 'proxy_ids': proxy_ids,
              'class_labels': cg['classes'], 'class_counts': cg['counts'], 'class_R': cg['R'],
              'class_cone90_deg': cg['cone90'], 'class_d0_p05': cg['d0_p05'],
              'class_d0_median': cg['d0_median'], 'class_d0_p95': cg['d0_p95'],
              'whole_own_center_angle_deg': cg['whole_own_center_angle_deg'],
              'whole_loo_prediction': cg['whole_loo_prediction'], **pa, **ra}
    for domain, radial in [('whole', whole_radial), ('proxy', proxy_radial)]:
        arrays.update({f'{domain}_{k}': a for k, a in radial.items()})
    depth_order = np.lexsort((ids, whole_radial['d0']))
    depth_rank = np.empty(len(x), np.int64)
    depth_rank[depth_order] = np.arange(1, len(x) + 1)
    objects = [{'sample_id': int(ids[i]), 'class_label': int(labels[i]),
                'view_id': int(view_ids[i]), 'd0': whole_radial['d0'][i], 'rho': whole_radial['rho'][i],
                'depth_rank': int(depth_rank[i]), 'raw_top4_slots': int(ra['raw_whole_top4_slots'][i]),
                'direction_top4_slots': int(ra['direction_whole_top4_slots'][i]),
                'own_class_center_angle_deg': cg['whole_own_center_angle_deg'][i],
                'nearest_raw_proxy_id': int(proxy_ids[ra['whole_nearest_raw_proxy'][i]])}
               for i in range(len(x))]
    proxies = []
    for i in range(len(p)):
        row = {'proxy_id': int(proxy_ids[i]), 'd0': proxy_radial['d0'][i], 'rho': proxy_radial['rho'][i],
               'actual_training_usage': 'unknown', 'nearest_class_label': int(pa['proxy_nearest_class'][i]),
               'nearest_class_angle_deg': pa['proxy_nearest_center_angle_deg'][i],
               'top2_center_cosine_margin': pa['proxy_top2_cosine_margin'][i],
               'within_direction_cone90': bool(pa['proxy_inside_direction_cone90'][i]),
               'within_class_radial90': bool(pa['proxy_inside_class_radial90'][i]),
               'raw_occupancy': int(ra['raw_proxy_occupancy'][i]),
               'direction_occupancy': int(ra['direction_proxy_occupancy'][i])}
        for name in ('raw', 'direction'):
            row[f'{name}_purity4'] = ra[f'{name}_proxy_purity4'][i]
            row[f'{name}_purity16'] = ra[f'{name}_proxy_purity16'][i]
            for j in range(4):
                index = ra[f'{name}_top_index'][i, j]
                row[f'{name}_top{j}_id'] = int(ids[index])
                row[f'{name}_top{j}_label'] = int(labels[index])
        proxies.append(row)
    raw_class_slots = np.bincount(cg['inverse'], weights=ra['raw_whole_top4_slots'], minlength=len(class_rows))
    direction_class_slots = np.bincount(cg['inverse'], weights=ra['direction_whole_top4_slots'], minlength=len(class_rows))
    center_class_counts = np.bincount(pa['proxy_nearest_class_index'], minlength=len(class_rows))
    for ci, row in enumerate(class_rows):
        row['raw_top4_slots'] = int(raw_class_slots[ci])
        row['direction_top4_slots'] = int(direction_class_slots[ci])
        row['raw_top4_enrichment_vs_population'] = row['raw_top4_slots'] / (len(p) * 4) / (row['n'] / len(x))
        row['direction_top4_enrichment_vs_population'] = row['direction_top4_slots'] / (len(p) * 4) / (row['n'] / len(x))
        row['nearest_center_proxy_count'] = int(center_class_counts[ci])
    destination.mkdir()
    np.savez_compressed(destination / 'plot_arrays.npz', **arrays)
    write_csv(destination / 'class_metrics.csv', class_rows)
    write_csv(destination / 'proxy_metrics.csv', proxies)
    write_csv(destination / 'object_hotspots.csv', objects)
    (destination / 'summary.json').write_text(json.dumps(finite_json(summary), ensure_ascii=False, indent=2), encoding='utf-8')
    if sha256(cache_path) != digest:
        raise RuntimeError('Source cache changed during analysis.')
    print(json.dumps({'model': record['name'], 'N': len(x), 'P': len(p),
                      'raw_coverage': neighbors['raw']['different_objects_top4'],
                      'direction_coverage': neighbors['direction']['different_objects_top4']}, ensure_ascii=False), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Output directory already exists; use a fresh path.')
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    models = spec['models']
    if not models:
        raise ValueError('No models.')
    names = [r['name'] for r in models]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Model names must be unique safe directory names.')
    output.mkdir(parents=True)
    summaries = [analyze_model(record, output / record['name']) for record in models]
    total = {'spec_sha256': sha256(spec_path), 'models': summaries,
             'private_artifact_note': 'Complete identity mappings; keep outside public Git.',
             'protocol': 'Frozen full-pool CPU geometry; no model forward/training/GPU. Different datasets and objectives do not isolate a causal HIER change.'}
    (output / 'summary.json').write_text(json.dumps(finite_json(total), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
