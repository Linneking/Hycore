"""Frozen clean/augmentation cache stability, CPU-only and identity aligned.

Spec: {"augmentation_groups": [{"dataset": "cub", "caches":
[{"variant": "train", "cache": "clean.npz"}, {"variant": "train_aug1", ...}, ...]}]}.
First listed view is the clean reference. Proxies must be exactly identical.
This compares cached augmented inputs under export/eval conditions, not train BN
or a training trajectory. Complete internal mappings remain private artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hier_original_vs_v7_geometry_v1 import (
    class_geometry, finite_json, radial_coordinates, retrieval, sha256, stats,
    unit_rows, write_csv,
)


def load_aligned(record, reference_ids=None, reference_labels=None, reference_proxy=None, reference_c=None):
    path = Path(record['cache'])
    digest = sha256(path)
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z['mu'], dtype=np.float64)
        p = np.asarray(z['proxy_ball'], dtype=np.float64)
        ids = np.asarray(z['sample_ids'], dtype=np.int64)
        labels = np.asarray(z['labels'], dtype=np.int64)
        c = float(np.asarray(z['c']).ravel()[0])
        metadata = json.loads(str(np.asarray(z['metadata_json']).item())) if 'metadata_json' in z else {}
    if x.ndim != 2 or p.ndim != 2 or x.shape[1] != p.shape[1] or len(ids) != len(x) or len(labels) != len(x):
        raise ValueError('Invalid coordinate/identity shapes.')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs in a view.')
    if reference_ids is not None:
        order = np.argsort(ids, kind='stable')
        positions = np.searchsorted(ids[order], reference_ids)
        if len(ids) != len(reference_ids) or np.any(positions >= len(ids)):
            raise ValueError('View pools differ.')
        remap = order[positions]
        if not np.array_equal(ids[remap], reference_ids):
            raise ValueError('A reference ID is absent from the augmented view.')
        x, labels, ids = x[remap], labels[remap], ids[remap]
        if not np.array_equal(labels, reference_labels):
            raise ValueError('Same sample ID has different labels across views.')
        if c != reference_c or not np.array_equal(p, reference_proxy):
            raise ValueError('Curvature or hierarchy proxies differ across views.')
    unit, norms = unit_rows(x)
    if np.any(norms < 1e-12):
        raise ValueError('A whole has undefined direction.')
    return {'x': x, 'p': p, 'ids': ids, 'labels': labels, 'c': c, 'unit': unit,
            'radial': radial_coordinates(x, c), 'sha256': digest, 'metadata': metadata,
            'record': record}


def mean_by_class(values, inverse, counts):
    return np.bincount(inverse, weights=values, minlength=len(counts)) / counts


def set_jaccard_rows(a, b):
    if a.shape != b.shape:
        raise ValueError('Neighborhood shape mismatch.')
    return np.asarray([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])


def analyze_group(group, destination):
    caches = group['caches']
    if len(caches) < 2:
        raise ValueError('Need a clean reference and at least one augmentation view.')
    variants = [row['variant'] for row in caches]
    if len(set(variants)) != len(variants):
        raise ValueError('Duplicate variants.')
    reference = load_aligned(caches[0])
    proxy_unit, proxy_norm = unit_rows(reference['p'])
    if np.any(proxy_norm < 1e-12):
        raise ValueError('A proxy has undefined direction.')
    ref_geom, ref_class_rows, ref_cg = class_geometry(reference['unit'], reference['labels'],
                                                    reference['radial'], reference['ids'], loo_max_classes=0)
    ref_nn, ref_arrays = retrieval(reference['p'], reference['x'], reference['unit'], proxy_unit,
                                   reference['c'], reference['ids'], reference['labels'],
                                   reference['radial']['d0'], k=4)
    view_rows = []
    reference_rows = []
    object_rows = []
    class_rows = []
    arrays = {'sample_ids': reference['ids'], 'labels': reference['labels'],
              'class_labels': ref_cg['classes'], 'class_counts': ref_cg['counts']}
    saved_depth, saved_rho, saved_class_r = [], [], []
    saved_raw_slots, saved_dir_slots, saved_raw_top, saved_dir_top = [], [], [], []
    summaries = []
    for vi, record in enumerate(caches):
        if vi == 0:
            view, geometry, cg = reference, ref_geom, ref_cg
            nn, nn_arrays = ref_nn, ref_arrays
        else:
            view = load_aligned(record, reference['ids'], reference['labels'], reference['p'], reference['c'])
            geometry, _, cg = class_geometry(view['unit'], view['labels'], view['radial'], view['ids'], loo_max_classes=0)
            nn, nn_arrays = retrieval(view['p'], view['x'], view['unit'], proxy_unit, view['c'],
                                       view['ids'], view['labels'], view['radial']['d0'], k=4)
        cosine = np.clip(np.einsum('ij,ij->i', reference['unit'], view['unit']), -1, 1)
        angle = np.degrees(np.arccos(cosine))
        delta_d0 = view['radial']['d0'] - reference['radial']['d0']
        result = {'variant': record['variant'], 'cache_sha256': view['sha256'],
                  'N': len(view['ids']), 'D': view['x'].shape[1], 'P': len(view['p']), 'c': view['c'],
                  'metadata': view['metadata'],
                  'whole_radial': {k: stats(v) for k, v in view['radial'].items()},
                  'whole_direction': geometry, 'retrieval_top4': nn}
        if vi > 0:
            comparisons = {'whole_same_id_cosine': stats(cosine),
                           'whole_same_id_angle_deg': stats(angle),
                           'whole_d0_delta_aug_minus_clean': stats(delta_d0),
                           'class_balanced_mean_whole_angle_deg': float(mean_by_class(angle, cg['inverse'], cg['counts']).mean()),
                           'class_balanced_mean_abs_whole_d0_delta': float(mean_by_class(np.abs(delta_d0), cg['inverse'], cg['counts']).mean()),
                           'proxy_coordinates_exactly_unchanged': True}
            for metric in ('raw', 'direction'):
                clean_ids = reference['ids'][ref_arrays[f'{metric}_top_index']]
                view_ids = reference['ids'][nn_arrays[f'{metric}_top_index']]
                jac = set_jaccard_rows(clean_ids, view_ids)
                comparisons[f'{metric}_proxy_to_whole_top4_jaccard'] = stats(jac)
                comparisons[f'{metric}_ordered_top4_proxy_fraction_unchanged'] = float(np.all(clean_ids == view_ids, axis=1).mean())
                comparisons[f'{metric}_whole_to_proxy_retention'] = float((ref_arrays[f'whole_nearest_{metric}_proxy'] ==
                                                                         nn_arrays[f'whole_nearest_{metric}_proxy']).mean())
                arrays[f'{record["variant"]}_{metric}_top4_jaccard_vs_clean'] = jac
            result['comparison_with_clean'] = comparisons
            for i in range(len(reference['ids'])):
                object_rows.append({'variant': record['variant'], 'sample_id': int(reference['ids'][i]),
                                    'class_label': int(reference['labels'][i]), 'clean_d0': reference['radial']['d0'][i],
                                    'view_d0': view['radial']['d0'][i], 'delta_d0': delta_d0[i],
                                    'same_id_cosine': cosine[i], 'same_id_angle_deg': angle[i],
                                    'clean_raw_top4_slots': int(ref_arrays['raw_whole_top4_slots'][i]),
                                    'view_raw_top4_slots': int(nn_arrays['raw_whole_top4_slots'][i]),
                                    'clean_direction_top4_slots': int(ref_arrays['direction_whole_top4_slots'][i]),
                                    'view_direction_top4_slots': int(nn_arrays['direction_whole_top4_slots'][i])})
            class_angles = mean_by_class(angle, cg['inverse'], cg['counts'])
            class_delta = mean_by_class(delta_d0, cg['inverse'], cg['counts'])
            class_abs_delta = mean_by_class(np.abs(delta_d0), cg['inverse'], cg['counts'])
            for ci, label in enumerate(cg['classes']):
                class_rows.append({'variant': record['variant'], 'class_label': int(label), 'n': int(cg['counts'][ci]),
                                   'clean_R': ref_cg['R'][ci], 'view_R': cg['R'][ci],
                                   'mean_same_id_angle_deg': class_angles[ci],
                                   'mean_d0_delta': class_delta[ci], 'mean_abs_d0_delta': class_abs_delta[ci]})
        saved_depth.append(view['radial']['d0'])
        saved_rho.append(view['radial']['rho'])
        saved_class_r.append(cg['R'])
        saved_raw_slots.append(nn_arrays['raw_whole_top4_slots'])
        saved_dir_slots.append(nn_arrays['direction_whole_top4_slots'])
        saved_raw_top.append(reference['ids'][nn_arrays['raw_top_index']])
        saved_dir_top.append(reference['ids'][nn_arrays['direction_top_index']])
        summaries.append(result)
        print(json.dumps({'dataset': group['dataset'], 'variant': record['variant'],
                          'raw_top4_coverage': nn['raw']['different_objects_top4'],
                          'direction_top4_coverage': nn['direction']['different_objects_top4']}, ensure_ascii=False), flush=True)
    arrays.update({'variants': np.asarray(variants), 'whole_d0_by_view': np.asarray(saved_depth),
                   'whole_rho_by_view': np.asarray(saved_rho), 'class_R_by_view': np.asarray(saved_class_r),
                   'raw_whole_top4_slots_by_view': np.asarray(saved_raw_slots),
                   'direction_whole_top4_slots_by_view': np.asarray(saved_dir_slots),
                   'raw_proxy_top4_ids_by_view': np.asarray(saved_raw_top),
                   'direction_proxy_top4_ids_by_view': np.asarray(saved_dir_top)})
    destination.mkdir()
    write_csv(destination / 'object_view_changes.csv', object_rows)
    write_csv(destination / 'class_view_changes.csv', class_rows)
    np.savez_compressed(destination / 'plot_arrays.npz', **arrays)
    summary = {'dataset': group['dataset'], 'reference_variant': variants[0], 'views': summaries,
               'interpretation': 'Frozen clean and augmented image caches; same model/proxies and eval export conditions. Does not measure train-BN geometry or temporal learning.',
               'private_artifact_note': 'Complete sample and proxy-neighborhood IDs remain private.'}
    (destination / 'summary.json').write_text(json.dumps(finite_json(summary), ensure_ascii=False, indent=2), encoding='utf-8')
    for record, result in zip(caches, summaries):
        if sha256(record['cache']) != result['cache_sha256']:
            raise RuntimeError('A source cache changed during analysis.')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Output already exists; use a fresh directory.')
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    groups = spec['augmentation_groups']
    names = [g['dataset'] for g in groups]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Dataset names must be unique safe directory names.')
    output.mkdir(parents=True)
    summaries = [analyze_group(group, output / group['dataset']) for group in groups]
    (output / 'summary.json').write_text(json.dumps(finite_json({'spec_sha256': sha256(spec_path),
                                                                'datasets': summaries}), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
