"""Create private scientific figures and a Chinese review from frozen audit outputs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ORDER = ['HIER_cub_train', 'HIER_cars_train', 'HIER_sop_train', 'V7_e005', 'V7_e187', 'V7_e300']
LABELS = ['HIER CUB best', 'HIER Cars best', 'HIER SOP best', 'V7 e5', 'V7 best e187', 'V7 last e300']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(fig, dest, name):
    fig.savefig(dest / (name + '.png'), dpi=160, bbox_inches='tight')
    fig.savefig(dest / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(v))}</th>' for v in headers) + '</tr></thead><tbody>' + \
           ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--conversion', required=True)
    ap.add_argument('--random', required=True)
    ap.add_argument('--numeric', required=True)
    ap.add_argument('--aug', required=True)
    ap.add_argument('--roles')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a fresh report directory.')
    out.mkdir(parents=True)
    figures = out / 'figures'
    figures.mkdir()
    analysis = Path(args.analysis)
    totals = read(analysis / 'summary.json')
    byname = {m['identity']['name']: m for m in totals['models']}
    models = [byname[n] for n in ORDER]
    arrays = {n: dict(np.load(analysis / n / 'plot_arrays.npz', allow_pickle=False)) for n in ORDER}
    random = {m['name']: m for m in read(args.random)['models']}
    numerical, aug, loader = read(args.numeric), read(args.aug), read(Path(args.conversion) / 'loader_manifest.json')
    roles = read(args.roles) if args.roles else None
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange, green = '#2679b5', '#e48632', '#42945b'

    fig, axs = plt.subplots(2, 3, figsize=(14, 8.0), sharex=True)
    for ax, name, label, m in zip(axs.ravel(), ORDER, LABELS, models):
        a = arrays[name]
        ax.hist(a['proxy_d0'], bins=np.linspace(0, 6.5, 85), color=orange, alpha=.78,
                weights=np.ones(len(a['proxy_d0'])) / len(a['proxy_d0']), label='代理（每组归一化）')
        if m['radial']['whole']['d0']['std_population'] < 1e-4:
            ax.axvline(np.median(a['whole_d0']), color=blue, lw=2.5, label='whole：全部位于薄壳')
            ax.text(.03, .91, 'whole σ < 6×10⁻⁷\n几何上可视为同层', transform=ax.transAxes,
                    va='top', color=blue, fontsize=9)
        else:
            ax.hist(a['whole_d0'], bins=np.linspace(0, 6.5, 85), color=blue, alpha=.45,
                    weights=np.ones(len(a['whole_d0'])) / len(a['whole_d0']), label='whole（每组归一化）')
        ax.set(title=f'{label} · c={m["identity"]["c"]}', xlabel='原点双曲深度 d₀', ylabel='每个 bin 的组内比例', xlim=(0, 6.5))
        ax.legend(fontsize=8, loc='upper left' if name.startswith('V7') else 'upper right')
    fig.suptitle('原版 whole 被 cap 压在同一深度；V7 whole 径向分散、代理出现两群', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, figures, '01_depth_distributions')

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.7))
    for ax, key, title in zip(axs, ['rho', 'q'], ['无量纲球半径 ρ = √c·‖x‖', '无量纲双曲深度 q = √c·d₀']):
        positions = np.arange(len(ORDER))
        for offset, domain, color in [(-.17, 'whole', blue), (.17, 'proxy', orange)]:
            b = ax.boxplot([arrays[n][f'{domain}_{key}'] for n in ORDER], positions=positions+offset,
                           widths=.28, whis=(0, 100), showfliers=False, patch_artist=True,
                           medianprops={'color': '#222', 'linewidth': 1.2})
            for patch in b['boxes']:
                patch.set_facecolor(color); patch.set_alpha(.65)
            ax.plot([], [], color=color, lw=5, label=domain)
        ax.set_xticks(positions, ['CUB', 'Cars', 'SOP', 'V7 e5', 'V7 e187', 'V7 e300'], rotation=15)
        ax.set_title(title); ax.legend(); ax.grid(axis='y', alpha=.2)
        if key == 'rho':
            ax.axhline(1, color='#888', ls='--', lw=1)
            ax.set_ylim(0, 1.03)
    fig.suptitle('不同曲率不能只比较 d₀：箱体 Q1–Q3，须线为完整最小–最大范围', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, figures, '02_dimensionless_radial_ranges')

    pos = np.arange(len(ORDER))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.0))
    for offset, mode, color in [(-.18, 'raw', blue), (.18, 'direction', orange)]:
        coverage = [m['retrieval'][mode]['different_objects_top4'] for m in models]
        axs[0].bar(pos+offset, coverage, .34, label=mode, color=color, alpha=.8)
        for x, y in zip(pos+offset, coverage):
            axs[0].text(x, y+20, str(y), ha='center', fontsize=9)
        reference = np.array([[t['retrieval'][mode]['different_objects_top4'] for t in random[n]['trials']] for n in ORDER])
        mean = reference.mean(1)
        axs[0].errorbar(pos+offset, mean, yerr=np.vstack((mean-reference.min(1), reference.max(1)-mean)),
                        fmt='D', color=green, ms=4, capsize=3, label='随机方向，5 seed' if mode == 'raw' else None)
        axs[1].bar(pos+offset, [m['retrieval'][mode]['maximum_repeated_quartet'] for m in models],
                   .34, label=mode, color=color, alpha=.8)
    axs[0].axhline(2048, color='#aaa', ls='--', lw=1)
    axs[0].set(ylabel='512×4 槽中覆盖的不同样本数', ylim=(0, 2240), title='raw vs whole 等深度（= direction）')
    axs[0].legend(fontsize=9)
    axs[1].set(yscale='log', ylabel='同一四例集合被多少代理选择', title='最大重复四例；对数坐标')
    axs[1].legend(fontsize=9)
    for ax in axs:
        ax.set_xticks(pos, ['CUB', 'Cars', 'SOP', 'V7 e5', 'V7 e187', 'V7 e300'], rotation=15)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('高覆盖不是已学到结构的充分证据：随机代理方向也能产生高覆盖', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, figures, '03_coverage_and_repeated_quartets')

    fig, axs = plt.subplots(2, 3, figsize=(14, 8.0), sharex=True, sharey=True)
    for ax, name, label in zip(axs.ravel(), ORDER, LABELS):
        a = arrays[name]
        scatter = ax.scatter(a['proxy_d0'], a['proxy_nearest_center_angle_deg'], s=14,
                             c=a['direction_proxy_purity4'], cmap='viridis', vmin=.25, vmax=1., alpha=.75)
        ax.axhline(np.median(a['class_cone90_deg']), ls='--', color='#999', lw=1)
        ax.set(title=label, xlabel='代理原点双曲深度 d₀', ylabel='到相对最近类别方向中心的角度（°）', xlim=(0, 6.3), ylim=(0, 100))
        cone = int(a['proxy_inside_direction_cone90'].sum())
        ax.text(.03, .05, f'落入各自最近类的 90% 方向锥：{cone}/512', transform=ax.transAxes, fontsize=8,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 2})
    fig.suptitle('NN4 同类不等于代理位于类簇内；虚线是类别 90% 方向锥角度的中位数', fontsize=13)
    fig.subplots_adjust(top=.90, bottom=.09, left=.07, right=.86, hspace=.3, wspace=.2)
    color_ax = fig.add_axes([.895, .22, .015, .55])
    fig.colorbar(scatter, cax=color_ax, label='direction NN4 主类比例')
    save(fig, figures, '04_proxy_depth_angle_and_purity')

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    axs[0].bar(pos-.16, [m['proxy_direction']['R'] for m in models], .3, color=orange, label='代理整体 R')
    axs[0].bar(pos+.16, [m['whole_direction']['class_R']['p50'] for m in models], .3, color=blue, label='whole 类内 R 中位数')
    axs[0].set(ylabel='R = ‖单位方向的均值‖', ylim=(0, 1.05), title='整体代理冗余与类内方向集中'); axs[0].legend(fontsize=9)
    for name, label, color in [('HIER_cub_train', 'CUB raw', '#347baa'), ('HIER_cars_train', 'Cars raw', '#599966'),
                               ('HIER_sop_train', 'SOP raw', '#a577ad'), ('V7_e300', 'V7 e300 raw', '#d96345')]:
        a = arrays[name]
        counts = np.sort(a['raw_whole_top4_slots'])[::-1]
        counts = counts[counts > 0]
        axs[1].plot(np.arange(1, len(counts)+1), counts, label=label, color=color)
    axs[1].set(xscale='log', yscale='log', xlabel='按热点频次排序的样本名次', ylabel='在 2048 个 NN4 槽中的频次', title='原版 SOP 也有热点；V7 更集中')
    axs[1].legend(fontsize=9)
    axs[0].set_xticks(pos, ['CUB', 'Cars', 'SOP', 'V7 e5', 'V7 e187', 'V7 e300'], rotation=15)
    for ax in axs:
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, figures, '05_direction_concentration_and_hotspots')

    radial_rows = []
    for m in totals['models']:
        for domain in ('whole', 'proxy'):
            for coordinate in ('r', 'rho', 'd0', 'q'):
                radial_rows.append({'model': m['identity']['name'], 'domain': domain, 'coordinate': coordinate,
                                    'c': m['identity']['c'], **m['radial'][domain][coordinate]})
    with (out / 'radial_statistics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(radial_rows[0])); w.writeheader(); w.writerows(radial_rows)
    radius_table, coverage_table, angular_table, random_table = [], [], [], []
    for name, label, m in zip(ORDER, LABELS, models):
        w, p = m['radial']['whole']['d0'], m['radial']['proxy']['d0']
        radius_table.append([label, f'{w["p00"]:.6f}–{w["p100"]:.6f}', f'{w["p50"]:.6f}', f'{w["std_population"]:.3g}',
                             f'{p["p00"]:.3f}–{p["p100"]:.3f}', f'{p["p50"]:.3f}', f'{p["std_population"]:.3f}'])
        r, d = m['retrieval']['raw'], m['retrieval']['direction']
        coverage_table.append([label, r['different_objects_top4'], d['different_objects_top4'], r['maximum_repeated_quartet'],
                               f'{100*r["top4_hot_object_slot_share"]:.2f}%', f'{100*d["top4_hot_object_slot_share"]:.2f}%'])
        a = m['proxy_class_affinity']
        randangle = np.mean([v['angle_deg']['p50'] for v in a['random_direction_reference']])
        angular_table.append([label, f'{m["whole_direction"]["class_R"]["p50"]:.4f}', f'{m["proxy_direction"]["R"]:.4f}',
                              f'{a["nearest_center_angle_deg"]["p50"]:.2f}°', f'{randangle:.2f}°',
                              a['proxy_count_in_nearest_class_direction_cone90'], f'{r["purity4"]["mean"]:.3f}'])
        trials = random[name]['trials']
        vals = [t['retrieval']['direction']['different_objects_top4'] for t in trials]
        rawvals = [t['retrieval']['raw']['different_objects_top4'] for t in trials]
        random_table.append([label, r['different_objects_top4'], f'{min(rawvals)}–{max(rawvals)}',
                             d['different_objects_top4'], f'{min(vals)}–{max(vals)}'])

    intro = ('这批资料已足够回答 best 模型的静态几何问题。已检查三个 best 权重、12 个完整缓存及样本身份表，'
             '没有下载图像数据集，没有编码器前向、优化更新或 GPU 使用。原版包含 CUB e17、Cars e45、SOP e96；'
             'V7 对照为 e5、验证选中的 best e187 和末期 e300。不同数据集、维度、曲率和训练目标，不能据此隔离单一算法的因果收益。')
    radial_text = ('关键发现是原版 whole 全部位于 cap 薄壳：clean train/eval 和两个增强视图都满足 d₀≈4.6，'
                   '标准差仅约 3–6×10⁻⁷，微差主要处于 FP32 数值尺度。官方 ToPoincare 对 whole 和层次代理均先截切空间范数至 2.3；'
                   'c=0.1 时这对应 d₀≤4.6、坐标半径 r≤1.9651、ρ≤0.6214。whole 的径向表达在这些 best 中几乎没有展开。'
                   '之前“原版未约束 whole 半径，因此 whole 应有明显深浅差”的前提应修正。代理则 CUB/Cars 集中于约 4.1–4.3，'
                   'SOP 有 497/512 个原始切空间参数超过 cap，中位映射深度也约 4.6。V7 whole 与代理的径向分散显著更大。')
    hotspot_text = ('whole 等深度控制保留代理的所有位置和 whole 的方向，只把 whole 半径统一：对严格双曲距离而言，'
                    '每个非零代理的样本排序就等于方向余弦排序。原版 NN4 集合与覆盖均不变；V7 e300 覆盖 27→221，'
                    '最大重复四例 287→45，最热四样本槽位 80.47%→28.08%。这直接确认径向差异是 V7 raw 热点的重要来源，'
                    '但去径向后仍明显集中，因此不是唯一因素。原版 SOP raw/direction 同为 432 个覆盖，最热四例占 35.25%，'
                    '说明原版也会产生方向热点。“原版没有热点”仅适用于这次 CUB/Cars 相对分散的情况。')
    random_text = ('五个 seed 的冻结控制把每个代理方向替换为独立均匀球面随机方向，保留各自半径、全部 whole、标签和输入。'
                   'CUB/Cars 的随机方向覆盖并不低于已训练代理，SOP 方向覆盖则由 432 提高到约 2,000；'
                   'V7 e300 在 whole 等深度下由 221 提高到约 1,250。说明高覆盖本身不是代理学出有效类内关系的证据，'
                   'V7 还存在当前代理方向排布的冗余。随机方向在 V7 原始半径下只覆盖 5 个样本，反而比已训练代理更差，'
                   '说明学习方向与径向偏置有交互，不能据此建议直接随机重置代理或宣称重训会提升分类。')
    direction_text = ('相对最近类别定义为代理单位方向与全部类别单位方向均值中心余弦最大的类别；它不是训练标签，也不是最近双曲类别中心。'
                      'CUB/Cars 代理最近类别角度中位数约 86°，与相同维度/类别中心下随机方向参照相近；'
                      'SOP 约 75°，明显小于随机参照约 83°。这三个原版 train best 都没有代理落入各自最近类别覆盖 90% 样本的方向锥，'
                      '即使部分代理的 NN4 全部同类，也不能称代理在类簇内部。该方向锥只是描述性几何检验，不是双曲凸包判定。'
                      '共同祖先本来可以位于样本类簇之外，这一结果本身不能判定 HIER 正则无效。'
                      'V7 e300 也为 0/512；whole 类内方向更集中，而代理整体 R 更高，方向冗余明显。')
    aug_rows = []
    for ds in aug['datasets']:
        views = ds['views']
        aug_rows.append([ds['dataset'].upper(), '/'.join(str(v['retrieval_top4']['raw']['different_objects_top4']) for v in views),
                         '/'.join(str(v['retrieval_top4']['raw']['maximum_repeated_quartet']) for v in views)])
    augmentation_text = ('两个增强缓存均使用相同模型/代理并以 eval 模式导出。CUB/Cars 的无大热点现象和 whole 薄壳均保留，'
                         'SOP 热点也保留；但同一代理的具体 NN4 随增强明显变化，不能把单个 clean 四例看作稳定类别绑定。'
                         '这些对照不是实际训练 BN 几何或逐轮轨迹。')
    numeric_text = ('官方源码 HEAD 已由 root 核对为 3986a744a1a54fd357e307d1cb3f2e81910b9ffc。'
                    '保存的 lcas 经未修改官方映射后，与缓存代理最大坐标误差≤1.49×10⁻⁷。'
                    '未修改官方 FP32 距离全量复算的覆盖为 CUB 1647、Cars 1664、SOP 432；严格 FP64 为 1646、1664、432。'
                    'CUB 只有 1 个代理的 NN4 集合因近并列发生变化，Cars/SOP 集合一致，不改变热点结论。'
                    '旧 proxy_stats.csv 极少行无法逐位复现，但指定同 ID 距离均差接近零；与旧 PyTorch/GPU 和当前 CPU 的近并列舍入差异相容，不能声称逐位重放。'
                    '独立重跑图像前向仍未执行：现有缓存足够本次几何分析，导出源码存在 out 在初始化前引用的问题，训练本地 diff 未交付。'
                    '目前不需要下载任何图像数据集。')
    role_text = ('pair/triple 检查使用交接中的 40 批冻结 argmin draws：45 类×2 实例、batch90、K20、每 anchor 50 次三元组采样；它不是实际训练使用日志。'
                 '只有 sample 项保存了逐 draw 索引，proxy 项与 Gumbel 只提供聚合摘要，不能由此恢复完整训练激活历史。')
    role_rows, role_imgs = [], []
    if roles:
        (out / 'frozen_roles_reference.json').write_text(json.dumps(roles, ensure_ascii=False, indent=2), encoding='utf-8')
        for ds, m in roles['datasets'].items():
            r = m['sample_argmin_noncollision_roles']
            role_rows.append([ds.upper(), f'{r["pair_occurrence_weighted_depth"]["q50"]:.6f}',
                              f'{r["triple_occurrence_weighted_depth"]["q50"]:.6f}',
                              f'{r["pair_minus_triple_depth_occurrence_weighted"]["mean"]:.6f}',
                              f'{r["overlap_proxies"]}/{r["union_proxies"]}'])
        role_text += ('按非碰撞 draw 的出现次数加权，CUB/Cars 的 pair 比 triple 平均深约 0.010/0.066，'
                      '但角色集合交集占各自并集约 99.8%，深度分布高度重叠。SOP 两类角色中位数几乎同为 4.6，差值只有约 10⁻⁶；'
                      '98.60% 的 SOP 角色深度差在 ±1e−4 内，不能把 cap 附近的严格正负号比例当作显著层级。原版没有表现为固定的 pair 深层群与 triple 浅层群。'
                      '代理—代理项的旧聚合报告有更明显的 pair 较深趋势，但缺逐 draw 记录，无法核验其完整角色分布。')
        for name in ('sample_argmin_role_depth', 'sample_argmin_proxy_role_mixture'):
            for ext in ('png', 'svg'):
                shutil.copyfile(Path(args.roles).parent / (name+'.'+ext), figures / (name+'.'+ext))
            role_imgs.append(name)
    paragraphs = [('检查范围', intro), ('径向分布', radial_text), ('热点与等深度控制', hotspot_text),
                  ('随机代理方向控制', random_text), ('代理是否在类内', direction_text),
                  ('增强稳定性', augmentation_text), ('pair/triple 的含义', role_text), ('数值核验与限制', numeric_text)]
    radius_headers = ['模型', 'whole 深度上下界', 'whole 中位数', 'whole 标准差', '代理深度上下界', '代理中位数', '代理标准差']
    coverage_headers = ['模型', 'raw 覆盖', '等深度覆盖', 'raw 最大重复四例', 'raw 最热四例槽位', '等深度最热四例槽位']
    angle_headers = ['模型', 'whole 类内 R 中位', '代理整体 R', '最近类角度中位', '随机方向角度参照', '类内方向锥代理数', 'raw NN4 纯度均值']
    random_headers = ['模型', '学习方向 raw', '随机方向 raw 范围', '学习方向等深度', '随机方向等深度范围']
    md = ['# 原版 HIER best：whole、代理与 V7 热点对照', intro]
    body = '<h1>原版 HIER best：whole、代理与 V7 热点对照</h1><p>' + html.escape(intro) + '</p>'
    sections = [
        ('径向分布', radial_text, radius_headers, radius_table, ['01_depth_distributions', '02_dimensionless_radial_ranges']),
        ('热点与等深度控制', hotspot_text, coverage_headers, coverage_table, ['03_coverage_and_repeated_quartets']),
        ('随机代理方向控制', random_text, random_headers, random_table, []),
        ('代理是否在类内', direction_text, angle_headers, angular_table, ['04_proxy_depth_angle_and_purity', '05_direction_concentration_and_hotspots']),
        ('增强稳定性', augmentation_text, ['数据集', 'clean/aug1/aug2 覆盖', 'clean/aug1/aug2 最大重复四例'], aug_rows, []),
        ('pair/triple 的含义', role_text, ['数据集', 'pair 深度中位', 'triple 深度中位', 'pair−triple 平均', '角色交集/并集'], role_rows, role_imgs),
        ('数值核验与限制', numeric_text, None, None, [])]
    for title, text, headers, rows, imgs in sections:
        body += f'<h2>{title}</h2><p>{html.escape(text)}</p>'
        md.extend(['\n## ' + title, text])
        if headers:
            body += table(headers, rows); md.append(markdown_table(headers, rows))
        for name in imgs:
            body += f'<a href="figures/{name}.svg"><img src="figures/{name}.png" alt="{name}"></a>'
            md.append(f'![{name}](figures/{name}.png)')
    body += '<h2>完整数据表</h2><p><a href="radial_statistics.csv">上下界、所有分位数、均值、方差、标准差：radial_statistics.csv</a></p>'
    body += '<p>每个模型的全对象/全代理/逐类数据表位于 analysis 目录；包含完整 ID，仅供本地研究审查。</p>'
    for name in byname:
        folder = analysis / name
        body += f'<p>{html.escape(name)}：' + ' · '.join(f'<a href="{(folder/f).as_uri()}">{label}</a>' for f,label in
                 [('summary.json','完整统计'), ('class_metrics.csv','逐类表'), ('proxy_metrics.csv','逐代理表'), ('object_hotspots.csv','样本热点表')]) + '</p>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原版 HIER best 几何审查</title><style>' + \
           'body{font-family:Microsoft YaHei,sans-serif;max-width:1250px;margin:34px auto;padding:0 24px;color:#273341;line-height:1.75}h1{font-size:28px}h2{margin-top:36px}p{max-width:1100px}img{width:100%;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border-bottom:1px solid #ddd;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef3f8}a{color:#236c9f}' + '</style>' + body + '</html>'
    (out / 'report.html').write_text(page, encoding='utf-8')
    (out / 'report.md').write_text('\n\n'.join(md), encoding='utf-8')
    anonymous = {'models': [], 'random_directions': read(args.random),
                 'method': 'Frozen complete-pool geometry; original 3 best; V7 e5/e187/e300; no causal isolation.'}
    for m in totals['models']:
        anonymous['models'].append({'name': m['identity']['name'],
                                    'N': m['identity']['N'], 'D': m['identity']['D'], 'P': m['identity']['P'], 'c': m['identity']['c'],
                                    'radial': m['radial'], 'proxy_direction': m['proxy_direction'],
                                    'whole_direction': m['whole_direction'], 'retrieval': m['retrieval'],
                                    'proxy_angle': m['proxy_class_affinity']['nearest_center_angle_deg'],
                                    'proxy_inside_cone90_count': m['proxy_class_affinity']['proxy_count_in_nearest_class_direction_cone90']})
    if roles:
        anonymous['original_frozen_sample_roles'] = {ds: m['sample_argmin_noncollision_roles'] for ds, m in roles['datasets'].items()}
    (out / 'anonymous_aggregate_evidence.json').write_text(json.dumps(anonymous, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(out/'report.html'), 'figures': 5+len(role_imgs), 'radial_rows': len(radial_rows),
                      'optimizer_updates': 0, 'gpu_used': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
