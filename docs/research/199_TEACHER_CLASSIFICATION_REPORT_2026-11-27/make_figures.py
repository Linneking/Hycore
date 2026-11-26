"""Create static figures from the already verified experiment summaries."""
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
rows = {x['id']: x for x in json.loads((ROOT / 'experiments.json').read_text(encoding='utf-8'))}
old = json.loads((ROOT.parent / '197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json').read_text(encoding='utf-8'))
for x in old['runs']:
    if x['run_id'] in ('TS06', 'TS23'):
        rows[x['run_id']] = dict(tail_oa=x['last10_test_oa_mean'], tail_aa=x['last10_test_aa_mean'])
blue, orange, gray = '#2F648B', '#D88735', '#A4ADB5'
evidence = {}
def fmt(value, signed=False):
    text = str(Decimal(str(round(value,8))).quantize(Decimal('.001'),rounding=ROUND_HALF_UP))
    return ('+' if signed and value >= 0 else '') + text

pairs = [
    ('MN40 当前批', [('4780', 'TS06', 'TS35'), ('4781', 'MN40-B05-S4781-H0', 'MN40-B05-S4781-HIER-CURRENT32')]),
    ('MN40 FIFO1', [('4780', 'TS06', 'TS44'), ('4781', 'MN40-B05-S4781-H0', 'MN40-B05-S4781-HIER-FIFO1')]),
    ('SO PointMLP 作者', [('12', 'SO-S12-H0', 'SO-S12-HIER'), ('4780', 'SO-S4780-H0', 'SO-S4780-HIER')]),
    ('SO DGCNN SH32', [('12', 'DG-S12-H0', 'DG-S12-HIER'), ('4780', 'DG-S4780-H0', 'DG-S4780-HIER')]),
]
fig, ax = plt.subplots(figsize=(11, 4.8))
pair_data = []
for i, (name, comparisons) in enumerate(pairs):
    values = []
    for j, (seed, base, arm) in enumerate(comparisons):
        a, b = rows[base]['tail_oa'], rows[arm]['tail_oa']
        delta = b - a
        y = len(pairs) - 1 - i + (.13 if j == 0 else -.13)
        ax.scatter(delta, y, s=85, c=[blue if j == 0 else orange], zorder=3)
        ax.annotate(f'seed {seed}: {fmt(delta,True)}', (delta, y), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9)
        values.append(delta)
        pair_data.append(dict(group=name, seed=seed, h0=base, hier=arm, h0_oa=a, hier_oa=b, delta_oa=delta))
    ax.plot(values, [len(pairs)-1-i+.13, len(pairs)-1-i-.13], color='#CCD3D8', linewidth=1.3, zorder=1)
ax.axvline(0, color='#3D4650', linewidth=1)
ax.set_yticks(range(len(pairs)), [x[0] for x in pairs][::-1])
ax.set_xlim(-.83, .91)
ax.set_ylim(-.5, 3.6)
ax.set_xlabel('末10轮 OA 差：HIER − 同seed H0（百分点）')
ax.set_title('同一配方在第二个 seed 中是否保持收益方向？', loc='left', fontweight='bold')
ax.grid(axis='x', alpha=.18)
fig.text(.18, .015, '每个点是一次完整配对训练；MN40 CURRENT32 的4781组与历史对照跨主机。', fontsize=9, color='#59636D')
fig.tight_layout(rect=[0, .05, 1, 1])
fig.savefig(OUT / 'paired_seed_effects.png', dpi=180)
plt.close(fig)
evidence['paired_seed_effects'] = pair_data

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
mn20 = []
for ax, key, title in zip(axes, ('tail_oa', 'tail_aa'), ('末10轮 OA', '末10轮 AA')):
    for i, (sampling, base, arm) in enumerate([('SH32', 'TS51', 'TS52'), ('CB32', 'TS54', 'TS55')]):
        values = [rows[base][key], rows[arm][key]]
        ax.bar([i-.18, i+.18], values, width=.31, color=[blue, orange])
        for x, value in zip([i-.18, i+.18], values):
            ax.text(x, value+.045, fmt(value), ha='center', fontsize=9)
        ax.text(i, max(values)+.25, f'Δ {fmt(values[1]-values[0],True)} pp', ha='center', fontsize=10)
        mn20.append(dict(metric=key, sampler=sampling, h0=values[0], hier=values[1]))
    ax.set_xticks([0, 1], ['SH32', 'CB32'])
    ax.set_ylim(90.3, 93.4 if key == 'tail_oa' else 92.1)
    ax.set_title(title)
    ax.set_ylabel('准确率（%）')
    ax.grid(axis='y', alpha=.15)
axes[0].legend([plt.Rectangle((0,0),1,1,color=blue),plt.Rectangle((0,0),1,1,color=orange)], ['同对象Part–Whole H0','H0＋HIER'], loc='lower left')
fig.suptitle('自定义 MN20：OA 与 AA 需要同时解释', x=.07, ha='left', fontweight='bold')
fig.text(.07,.015,'同子集、seed4780；没有该派生任务的论文同协议基准。纵轴为局部区间，柱顶标实际值。',fontsize=9,color='#59636D')
fig.tight_layout(rect=[0,.05,1,.91])
fig.savefig(OUT / 'mn20_oa_aa.png',dpi=180)
plt.close(fig)
evidence['mn20_oa_aa'] = mn20

labels = ['新 Part–Whole H0', '同 Part–Whole＋HIER', '原版 DeLA，从头训练', '作者发布权重单次复评']
values = [rows['DeLA-H0']['last_oa'], rows['DeLA-HIER']['last_oa'], rows['DeLA-v1-AUTHOR-S4780']['last_oa'], 90.42]
fig, ax = plt.subplots(figsize=(10.8, 4.4))
bars = ax.barh(range(4), np.array(values)-86, left=86, color=[gray, orange, blue, '#D8DFE4'], height=.54)
bars[-1].set_hatch('///')
for i,value in enumerate(values):ax.text(value+.06,i,f'{value:.3f}%',va='center',fontsize=11)
ax.set_yticks(range(4), labels)
ax.set_xlim(86,91.15)
ax.set_ylim(-1,3.6)
ax.set_xlabel('ScanObjectNN PB_T50_RS OA（%；训练组取固定 e250）')
ax.set_title('DeLA：HIER 净增量与共同基座改造成本同时存在',loc='left',fontweight='bold')
ax.grid(axis='x',alpha=.15)
ax.text(86.15,-.72,'新H0 → HIER：+0.763 pp；原版e250 → 新H0：−1.906 pp',fontsize=10)
fig.text(.28,.012,'发布权重来自另一训练，只作绝对水平参照；新HIER仍低于原版e250 1.143 pp。',fontsize=9,color='#59636D')
fig.tight_layout(rect=[0,.055,1,1])
fig.savefig(OUT / 'dela_absolute_levels.png',dpi=180)
plt.close(fig)
evidence['dela_absolute_levels'] = [dict(label=label, oa=value) for label,value in zip(labels,values)]
(ROOT / 'figure_data.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('Created 3 figures from verified summaries')
