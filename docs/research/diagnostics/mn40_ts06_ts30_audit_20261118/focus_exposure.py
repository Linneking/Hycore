"""Summarize existing frozen predictions; no sampling or model execution."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frequencies = read(args.delivery / 'train_exposures.csv')
    classes = sorted({row['label'] for row in frequencies})
    details, neighbours, boundaries = [], {}, {}
    for tag, field in [('best', 'TS30_best_count'), ('e300', 'TS30_final_count')]:
        predictions = read(args.delivery / (tag + '_train_objects.csv'))
        assert [r['id'] for r in frequencies] == [r['id'] for r in predictions]
        percentiles = {}
        for name in classes:
            indices = [i for i, r in enumerate(frequencies) if r['label'] == name]
            values = np.array([int(frequencies[i][field]) for i in indices])
            ranks = (rankdata(values) - .5) / len(values)
            percentiles.update((frequencies[i]['id'], float(x)) for i, x in zip(indices, ranks))
            low, high = ranks <= .1, ranks >= .9
            wrong = np.array([predictions[i]['TS30_prediction'] != name for i in indices])
            sh_only = np.array([predictions[i]['group'] == 'sh_only' for i in indices])
            details.append(dict(tag=tag, name=name, n=len(values), low10_n=int(low.sum()),
                high10_n=int(high.sum()), cb_wrong=int(wrong.sum()),
                low10_cb_wrong=int((low & wrong).sum()), high10_cb_wrong=int((high & wrong).sum()),
                sh_only_train=int(sh_only.sum()), low10_sh_only_train=int((low & sh_only).sum()),
                high10_sh_only_train=int((high & sh_only).sum())))
        if tag == 'best':
            rows = read(args.delivery / 'neighbours.csv')
            for name in ['table', 'dresser', 'vase', 'chair', 'xbox']:
                selected = [r for r in rows if r['class_name'] == name and r['group'] == 'sh_only' and int(r['rank']) <= 5]
                ranks = np.array([percentiles[r['train_id']] for r in selected])
                neighbours[name] = dict(query_n=len({r['test_id'] for r in selected}),
                    neighbour_slots=len(selected), low10_slots=int((ranks <= .1).sum()),
                    high10_slots=int((ranks >= .9).sum()),
                    mean_frequency_relative_to_class=float(np.mean([float(r['cb_class_relative']) for r in selected])))
        test = read(args.delivery / (tag + '_test_objects.csv'))
        boundaries[tag] = {}
        for first, second in [('table', 'desk'), ('dresser', 'night_stand'), ('vase', 'flower_pot')]:
            selected = [r for r in test if r['label'] in (first, second)]
            boundaries[tag][first + '/' + second] = dict(test_n=len(selected),
                truth_counts={k: sum(r['label'] == k for r in selected) for k in (first, second)},
                prediction_counts={model: {k: sum(r[model + '_prediction'] == k for r in selected)
                    for k in (first, second)} for model in ('TS06', 'TS30')})
    with (args.out / 'class_tail_details.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=details[0])
        writer.writeheader()
        writer.writerows(details)
    for filename, value in [('target_neighbour_details.json', neighbours), ('boundary_predictions.json', boundaries)]:
        (args.out / filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print('Summarized 80 class/cutoff groups, five neighbour groups and six boundary groups.')


if __name__ == '__main__':
    main()
