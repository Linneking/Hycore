"""Scientific plots from anonymous aggregate audit outputs; no model access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, target):
    fig.savefig(target.with_suffix('.png'), dpi=180, bbox_inches='tight')
    svg = target.with_suffix('.svg')
    fig.savefig(svg, bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


def median(stat):
    return stat['quantiles']['q50']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', type=Path, required=True)
    parser.add_argument('--adam', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    if args.angles.is_file():
        rows = json.loads(args.angles.read_text(encoding='utf-8'))['observations']
    else:
        for name in ('cars_summary.json', 'v9_summary.json'):
            rows.extend(json.loads((args.angles / name).read_text(encoding='utf-8'))['observations'])
    frozen = [r for r in rows if r['source_kind'] == 'existing_frozen_cache']
    actual = [r for r in rows if r['source_kind'] == '71_diagnostic_actual_training_forward']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    colors = ['#334155', '#2563eb', '#dc2626', '#059669']
    labels = ['HIER Cars best', 'V9 HIER best', 'V9 HIER e300', 'V9 B64 e300']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ['within_macro', 'between_class_pair_balanced'],
                               ['Within class (equal class mass)', 'Between classes (equal class-pair mass)']):
        for r, label, color in zip(frozen, labels, colors):
            curve = r[field]['angle_cdf']
            ax.plot(curve['x_deg'], np.clip(curve['y'], 0, 1), label=label, color=color)
        ax.set(xlim=(0, 130), ylim=(0, 1.01), xlabel='Angle (degrees)', ylabel='Cumulative probability', title=title)
        ax.grid(alpha=.2)
    axes[0].axvline(30.966654, color='#94a3b8', ls=':', label='Pair center depth = 4.14')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Frozen clean/eval directions; different tasks and label granularities')
    save(fig, args.out / '01_angle_cdf')

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for offset, field, color, label in [(-.16, 'pair_midpoint_depth_shell', '#2563eb', 'Same-class pair center'),
                                       (.16, 'triple_absolute_minimax_depth_shell', '#d97706', '+ Different-class k: triple center')]:
        boxes = []
        for r in frozen:
            stat = r['sameclass_pair_otherclass_k_reference'][field]
            q = stat['quantiles']
            boxes.append(dict(med=q['q50'], q1=q['q25'], q3=q['q75'], whislo=q['q05'], whishi=q['q95'], fliers=[]))
        artists = ax.bxp(boxes, positions=np.arange(len(frozen)) + offset, widths=.25,
                         patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in artists['boxes']:
            patch.set(facecolor=color, alpha=.65)
        ax.plot([], [], color=color, lw=8, alpha=.65, label=label)
    ax.axhline(4.6, color='#475569', ls='--', label='Whole shell = 4.6')
    ax.axhline(4.14, color='#94a3b8', ls=':', label='Typical initial proxy = 4.14')
    ax.set(xticks=np.arange(len(frozen)), xticklabels=labels, ylabel='Origin depth (c = 0.1)',
           ylim=(0, 4.9), title='Absolute minimax references, not HIER-loss optima (whiskers: 5-95%)')
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8, loc='lower left')
    save(fig, args.out / '02_reference_center_depth')

    adam = json.loads(args.adam.read_text(encoding='utf-8'))
    first = adam['snapshots'][0]['variants']
    names = ['native', 'lambda0p1_all_gradient_history', 'lr_half_only', 'eps1e4_only']
    titles = ['Native', 'lambda .5 -> .1', 'LR .01 -> .005', 'eps 1e-8 -> 1e-4']
    values = [first[k]['all']['step_norm']['median'] for k in names]
    counts = [first[k]['all']['newly_over_cap'] for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, data, title, ylabel in [(axes[0], values, 'Median vector displacement', 'Raw parameter units'),
                                    (axes[1], counts, 'New cap crossings (of 512)', 'Proxy count')]:
        bars = ax.bar(np.arange(4), data, color=['#475569', '#7c3aed', '#2563eb', '#059669'])
        ax.set(xticks=np.arange(4), xticklabels=titles, title=title, ylabel=ylabel)
        ax.tick_params(axis='x', labelsize=8)
        ax.bar_label(bars, labels=[str(v) if isinstance(v, int) else f'{v:.4f}' for v in data], padding=3)
        ax.set_ylim(0, max(data) * 1.18)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Same saved V9 first step: one-factor changes, no training continuation')
    save(fig, args.out / '03_first_step_sensitivity')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ticks = [f"e{r['epoch']} s{r['step']}" for r in actual]
    x = np.arange(len(actual))
    fields = [('pair_midpoint_depth_shell', 'Ideal pair center', '#2563eb', '-'),
              ('triple_absolute_minimax_depth_shell', 'Ideal triple center', '#d97706', '-'),
              ('selected_pair_proxy_depth', 'Selected pair proxy', '#dc2626', '--'),
              ('selected_triple_proxy_depth', 'Selected triple proxy', '#7c3aed', '--')]
    for field, label, color, style in fields:
        axes[0].plot(x, [median(r['actual_mined_triplets'][field]) for r in actual],
                     color=color, ls=style, marker='o', label=label)
    axes[0].axhline(4.6, color='#94a3b8', ls=':', lw=1)
    axes[0].set(ylabel='Median origin depth', ylim=(0, 4.85), title='Actual mined triplets: reference vs selected')
    bars = axes[1].bar(x, [100*r['actual_mined_triplets']['ij_same_class_fraction'] for r in actual], color='#475569')
    axes[1].bar_label(bars, fmt='%.1f%%', padding=3)
    axes[1].set(ylabel='Same-class i,j draws (%)', ylim=(0, 20), title='Most actual pair draws cross classes')
    for ax in axes:
        ax.set(xticks=x, xticklabels=ticks)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Saved actual training forwards from report-71 diagnostic repeat; no encoder rerun')
    save(fig, args.out / '04_actual_training_triplets')


if __name__ == '__main__':
    main()
