"""Redraw only the depth figure from immutable existing aggregate tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .report import read_csv, validate_summary, _plotting, plot_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    source = args.input_dir
    inputs = [source/'summary.json', source/'tables/depth_hist.csv', source/'tables/depth_ecdf.csv']
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    summary = json.loads(inputs[0].read_text(encoding='utf-8'))
    models = validate_summary(summary)
    tables = {p.name: read_csv(p) for p in inputs[1:]}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    np, plt = _plotting()
    plot_depth(tables, models, args.out_dir, np, plt)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(source))] for p in inputs)
    files = ('01_depth.png', '01_depth.svg')
    result = {'status': 'completed', 'commit': args.commit, 'layout': 'percentage plus ECDF; count row removed',
              'rows': 2, 'columns': 4, 'objects_per_view': 9840, 'new_embedding_or_relation_calculation': False,
              'input_files_sha256': hashes,
              'output_files_sha256': {name: hashlib.sha256((args.out_dir/name).read_bytes()).hexdigest() for name in files}}
    (args.out_dir/'depth_figure_redraw.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
