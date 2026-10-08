"""Frozen proxy-direction replacement control; preserves every proxy radius.

No optimizer, training, or encoder forward. Random directions are a descriptive
reference for full-pool coverage, not an independent trained baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hier_original_vs_v7_geometry_v1 import finite_json, radial_coordinates, retrieval, sha256, unit_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[20261008, 20261009, 20261010, 20261011, 20261012])
    args = ap.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh output directory.')
    dest.mkdir(parents=True)
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
    results = []
    for record in spec['models']:
        if record.get('split') != 'train':
            continue
        source = Path(record['cache'])
        before = sha256(source)
        with np.load(source, allow_pickle=False) as z:
            x, p = np.asarray(z['mu'], np.float64), np.asarray(z['proxy_ball'], np.float64)
            labels, ids = z['labels'], z['sample_ids']
            c = float(np.asarray(z['c']).ravel()[0])
        u, _ = unit_rows(x)
        _, radii = unit_rows(p)
        depth = radial_coordinates(x, c)['d0']
        trials = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            v, _ = unit_rows(rng.normal(size=p.shape))
            replacement = v * radii[:, None]
            summary, arrays = retrieval(replacement, x, u, v, c, ids, labels, depth, k=16)
            trials.append({'seed': seed, 'retrieval': summary})
            print(json.dumps({'model': record['name'], 'seed': seed,
                              'raw_coverage': summary['raw']['different_objects_top4'],
                              'direction_coverage': summary['direction']['different_objects_top4']}), flush=True)
            del arrays
        assert sha256(source) == before
        results.append({'name': record['name'], 'cache_sha256': before, 'c': c, 'D': x.shape[1],
                        'P': len(p), 'N': len(x), 'trials': trials})
    report = {'models': results, 'spec_sha256': sha256(args.spec),
              'definition': 'Replace each proxy direction with an independent uniform sphere direction; preserve proxy radius and all whole coordinates.',
              'interpretation': 'A frozen descriptive geometry reference. Does not measure retraining benefit, hierarchy correctness, or independent morphology.'}
    (dest / 'summary.json').write_text(json.dumps(finite_json(report), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
