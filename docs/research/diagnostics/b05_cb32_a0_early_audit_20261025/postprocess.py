"""Reuse audit85 unchanged, then summarize the fresh CB32 A0 e0..e10 trajectory.

No model or GPU access.  Epochs of the original completed run are auxiliary
observations; missing historical epochs are never relabelled as fresh replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT85 = HERE.parent / "embedding_audit85" / "analyze.py"
REPLAY = "B05_CB32_A0_REPLAY"
ORIGINAL = "B05_CB32_A0_ORIGINAL"
TOL = 1e-5


def audit85_module():
    spec = importlib.util.spec_from_file_location("cb32_early_reused_audit85", AUDIT85)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Expected finite scalar")
    return value


def percent(value):
    value = number(value)
    return None if value is None else 100 * value


def equivalence_scalars(value):
    """Only allow the agreed numeric e1/e5 comparison, never private metadata."""
    result={}
    for epoch in ['1','5']:
        if epoch not in (value or {}):continue
        result[epoch]={}
        for component in ['net','proxy']:
            entry=value[epoch][component]
            if not isinstance(entry['bitwise_equal'],bool):raise ValueError('Bitwise equality must be explicit boolean')
            result[epoch][component]={'bitwise_equal':entry['bitwise_equal'],
                'max_abs_difference':number(entry['max_abs_difference']),
                'mean_abs_difference':number(entry['mean_abs_difference'])}
    return result


def movement_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Movement requires one finite scalar per proxy")
    return dict(population=len(values), p05=float(np.quantile(values, .05)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                mean=float(values.mean()), inward_percent=float(100*np.mean(values < -TOL)),
                outward_percent=float(100*np.mean(values > TOL)),
                stationary_percent=float(100*np.mean(np.abs(values) <= TOL)))


def replay_observations(summary, expected_epochs=tuple(range(11))):
    rows = sorted((s for s in summary["observations"] if s["run_id"] == REPLAY), key=lambda s: s["epoch"])
    actual = [r["epoch"] for r in rows]
    if actual != list(expected_epochs):
        raise ValueError(f"Fresh replay epochs are incomplete/duplicated: {actual}")
    if not rows or rows[0]["epoch"] != 0:
        raise ValueError("An explicit e0 is needed for paired proxy movements")
    for row in rows:
        if row["stage"] != f"e{row['epoch']:03d}":
            raise ValueError("Replay stage must state its actual epoch")
        if not row["proxies"]["available"]:
            raise ValueError("Replay needs the actual proxy population")
    if any(s.get("available") for s in rows[0]["proxies"]["usage"].values()):
        raise ValueError("Initialization has no training usage; e0 must remain N/A")
    return rows


def proxy_trajectories(observations, proxy_rows):
    """Paired scalar changes only within this one fresh replay, never cross-run IDs."""
    by_observation = {}
    expected_ids = None
    population = observations[0]["proxies"]["proxy_count"]
    for obs in observations:
        rows = [r for r in proxy_rows if r["observation_id"] == obs["observation_id"]]
        ids = [int(r["proxy_id"]) for r in rows]
        if len(ids) != population or len(set(ids)) != population:
            raise ValueError("Every replay epoch needs all unique proxy IDs")
        rows.sort(key=lambda r: int(r["proxy_id"]))
        ordered_ids = [int(r["proxy_id"]) for r in rows]
        if expected_ids is None:
            expected_ids = ordered_ids
        if ordered_ids != expected_ids:
            raise ValueError("Proxy IDs changed within the replay")
        mapped = np.asarray([number(r["depth"]) for r in rows], float)
        raw = np.asarray([number(r["raw_tangent_norm"]) for r in rows], float)
        if not np.isfinite(mapped).all() or not np.isfinite(raw).all():
            raise ValueError("Mapped depths and raw tangent norms must both exist")
        if np.any(mapped < 0) or np.any(raw <= 0):
            raise ValueError("Invalid depth/norm")
        by_observation[obs["observation_id"]] = (mapped, raw)
    first = by_observation[observations[0]["observation_id"]]
    previous = first
    scalar_rows, epoch_rows = [], []
    for obs in observations:
        current = by_observation[obs["observation_id"]]
        arrays = {"depth_delta_epoch": current[0]-previous[0], "depth_delta_init": current[0]-first[0],
                  "raw_norm_delta_epoch": current[1]-previous[1], "raw_norm_delta_init": current[1]-first[1]}
        aggregate = {"observation_id": obs["observation_id"], "epoch": obs["epoch"]}
        for name, values in arrays.items():
            aggregate.update({name+"_"+k: v for k,v in movement_stats(values).items()})
        for i, proxy_id in enumerate(expected_ids):
            scalar_rows.append(dict(observation_id=obs["observation_id"], epoch=obs["epoch"], proxy_id=proxy_id,
                                    depth=float(current[0][i]), raw_tangent_norm=float(current[1][i]),
                                    **{name: float(values[i]) for name,values in arrays.items()}))
        epoch_rows.append(aggregate)
        previous = current
    return scalar_rows, epoch_rows


def movement_histograms(scalars):
    result = []
    epochs = sorted({r["epoch"] for r in scalars})
    for field in ("depth_delta_epoch", "depth_delta_init", "raw_norm_delta_epoch", "raw_norm_delta_init"):
        all_values = np.asarray([r[field] for r in scalars], float)
        low = min(0., np.floor(all_values.min()/.025)*.025)
        high = max(.025, np.ceil(all_values.max()/.025)*.025)
        edges = np.arange(low-.0125, high+.025, .025)
        for epoch in epochs:
            values = np.asarray([r[field] for r in scalars if r["epoch"] == epoch], float)
            counts, _ = np.histogram(values, bins=edges)
            if counts.sum() != len(values):
                raise ValueError("Movement histogram clipped values")
            for left,right,count in zip(edges[:-1],edges[1:],counts):
                result.append(dict(epoch=epoch, domain=field, bin_left=float(left), bin_right=float(right),
                                   count=int(count), population=len(values), percent=float(100*count/len(values))))
    return result


def order_value(proxy, domain, subset, field):
    return percent(proxy["frozen_orders"][domain]["domains"][subset]["tolerance"]["0.001"][field])


def summary_rows(observations, movement_rows):
    result = []
    for obs, movement in zip(observations, movement_rows):
        p = obs["proxies"]
        n = p["proxy_count"]
        row = dict(epoch=obs["epoch"], observation_id=obs["observation_id"], objects=obs["objects"], proxies=n,
                   cap_raw_count=p["raw_above_tangent_cap_count"],
                   cap_raw_percent=100*p["raw_above_tangent_cap_count"]/n,
                   near_cap_percent=100*p["cap_near_count_tol_1e3"]/n,
                   whole_depth_median=obs["domains"]["native_whole"]["depth"]["median"],
                   proxy_depth_median=p["depth"]["median"], proxy_depth_p05=p["depth"]["p05"], proxy_depth_p95=p["depth"]["p95"],
                   raw_norm_median=p["raw_tangent_norm"]["median"], raw_norm_p05=p["raw_tangent_norm"]["p05"],
                   raw_norm_p95=p["raw_tangent_norm"]["p95"], raw_norm_max=p["raw_tangent_norm"]["max"],
                   intra_angle_pair_weighted_median=obs["angles"]["intra_pair_weighted"]["median"],
                   intra_angle_class_equal_mean_of_medians=obs["angles"]["class_balanced_intra_average_of_class_statistics"]["median"],
                   inter_angle_pair_weighted_median=obs["angles"]["inter_pair_weighted"]["median"],
                   centroid_angle_median=obs["angles"]["centroid_separation_upper_triangle"]["median"],
                   negative_cone_gap_p95_class_pairs=obs["angles"]["class_pairs_with_negative_cone_gap_p95"],
                   sample_order_strict_noncollision_percent=order_value(p,"sample","noncollision","all_three_orders_strict"),
                   proxy_order_strict_noncollision_percent=order_value(p,"proxy","noncollision","all_three_orders_strict"),
                   proxy_order_strict_noncollision_nonendpoint_percent=order_value(p,"proxy","noncollision_nonendpoint","all_three_orders_strict"),
                   proxy_collision_percent=percent(p["frozen_orders"]["proxy"]["collision_fraction"]),
                   proxy_pair_endpoint_self_percent=percent(p["frozen_orders"]["proxy"]["pair_equals_an_endpoint_fraction"]),
                   proxy_triple_endpoint_self_percent=percent(p["frozen_orders"]["proxy"]["triple_equals_an_endpoint_fraction"]),
                   nn4_same_class_percent=percent(p["angular_nn4_same_class_proxy_fraction"]),
                   nn4_coverage_percent=p["angular_nn4_hotspots"]["coverage_percent"],
                   nn4_top4_slot_percent=percent(p["angular_nn4_hotspots"]["top4_object_slot_share"]))
        for name in ("sample_usage", "proxy_usage", "sample_effective_usage", "proxy_effective_usage"):
            value=p["usage"][name]
            row[name+"_available"]=value["available"]
            for key in ("selection_count", "used_proxy_count", "effective_proxy_count", "maximum_selection_share"):
                row[name+"_"+key] = value.get(key) if value["available"] else None
        row.update(movement)
        result.append(row)
    return result


def make_figures(out, observations, rows, tables, movements, movement_hist, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,"svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False})
    folder=out/'figures';folder.mkdir()
    files=[];epochs=[r["epoch"] for r in rows];ids={s["observation_id"]:s["epoch"] for s in observations}
    def save(fig,name,title,note=""):
        fig.suptitle(title,fontsize=13,y=.995)
        if note:fig.text(.015,.009,note,fontsize=8,va="bottom")
        fig.tight_layout(rect=(0,.055 if note else .015,1,.955))
        for suffix in ['png','svg']:
            p=folder/(name+'.'+suffix);fig.savefig(p,dpi=170,bbox_inches='tight');files.append({'file':p.relative_to(out).as_posix(),'sha256':sha(p)})
        plt.close(fig)
    def curve(ax,field,label,**kw):
        ax.plot(epochs,[r.get(field) if r.get(field) is not None else np.nan for r in rows],marker='o',ms=3,label=label,**kw)
        ax.set(xlabel='Completed replay epoch');ax.set_xticks(epochs)
    def select(table,observation,**criteria):
        return [r for r in tables[table] if r['observation_id']==observation and all(str(r.get(k))==str(v) for k,v in criteria.items())]
    def heatmap(ax,group_rows,xfield,leftfield='bin_left',rightfield='bin_right',population_note='',vmax=None):
        selected=[r for r in group_rows if r['observation_id'] in ids]
        xs=sorted({float(r[xfield]) for r in selected})
        image=np.array([[float(next(r['percent'] for r in selected if ids[r['observation_id']]==e and float(r[xfield])==x)) for x in xs] for e in epochs])
        left=min(float(r[leftfield]) for r in selected);right=max(float(r[rightfield]) for r in selected)
        p=ax.imshow(image,origin='lower',aspect='auto',extent=[left,right,epochs[0]-.5,epochs[-1]+.5],vmin=0,vmax=vmax,cmap='magma')
        ax.set(xlabel=population_note,ylabel='Completed replay epoch',yticks=epochs);plt.colorbar(p,ax=ax,label='Population per bin (%)')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,domain,title in zip(axes,['native_whole','proxy'],['Native whole (9840 objects)','Proxies (512 parameters)']):
        group=[r for r in tables['depth_histograms'] if r['domain']==domain]
        heatmap(ax,group,'bin_center',population_note='Origin hyperbolic distance, c0.1')
        maximum=max(s['domains']['native_whole']['depth']['max'] if domain=='native_whole' else s['proxies']['depth']['max'] for s in observations)
        ax.set_xlim(0,max(4.8,maximum+.15));ax.set_title(title);ax.axvline(4.6,color='cyan',ls='--',lw=1)
    save(fig,'01_depth_percentage_trajectory','Clean/eval depth distributions: each epoch/population sums to100%', 'Fixed HIER view at4.6 is constructed; raw learned whole is separate. No KDE.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));chosen=[e for e in [0,1,2,5,10] if e in epochs]
    for domain,ax in zip(['native_whole','proxy'],axes):
        for obs in observations:
            if obs['epoch'] not in chosen:continue
            values=select('depth_cdf',obs['observation_id'],domain=domain)
            ax.step([float(r['depth']) for r in values],[float(r['cumulative_percent']) for r in values],where='post',label=f"e{obs['epoch']}")
        ax.set(title=domain.replace('_',' '),xlabel='Origin distance c0.1',ylabel='Cumulative population (%)',ylim=(0,102));ax.axvline(4.6,color='.4',ls='--');ax.legend()
    save(fig,'02_depth_ecdf','Bin-independent depth ECDF at fixed early epochs')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    curve(axes[0,0],'cap_raw_percent','raw norm>2.3');curve(axes[0,0],'near_cap_percent','mapped |depth-4.6|<=1e-3');axes[0,0].set(ylabel='Proxies (%)',ylim=(0,102));axes[0,0].legend()
    curve(axes[0,1],'raw_norm_median','median raw norm');axes[0,1].fill_between(epochs,[r['raw_norm_p05'] for r in rows],[r['raw_norm_p95'] for r in rows],alpha=.2,label='P5-P95');axes[0,1].axhline(2.3,color='.4',ls='--');axes[0,1].set(ylabel='Raw tangent norm');axes[0,1].legend()
    for field,label in [('whole_depth_median','native whole'),('proxy_depth_median','proxy')]:curve(axes[1,0],field,label)
    axes[1,0].set(ylabel='Median origin depth');axes[1,0].axhline(4.6,color='.4',ls='--');axes[1,0].legend()
    for prefix,label in [('depth_delta_init','mapped depth'),('raw_norm_delta_init','raw norm')]:
        curve(axes[1,1],prefix+'_median',label);axes[1,1].fill_between(epochs,[r[prefix+'_p05'] for r in rows],[r[prefix+'_p95'] for r in rows],alpha=.14)
    axes[1,1].axhline(0,color='.4');axes[1,1].set(ylabel='Signed change from own initialization');axes[1,1].legend()
    save(fig,'03_cap_raw_and_paired_movement','Mapped cap can hide continuing raw parameter motion','Same proxy IDs are paired only within this one replay. Bands are P5-P95, not uncertainty intervals.')
    fig,axes=plt.subplots(2,1,figsize=(13,11),gridspec_kw={'height_ratios':[1,2.5]})
    curve(axes[0],'intra_angle_pair_weighted_median','within class, all pairs');curve(axes[0],'intra_angle_class_equal_mean_of_medians','mean of40 class medians');axes[0].set(ylabel='Within-class angle (degrees)');axes[0].legend()
    matrix=np.array([[float(next(r['median'] for r in select('class_intra_angles',s['observation_id']) if int(r['class_id'])==k)) for s in observations] for k in range(len(names))])
    p=axes[1].imshow(matrix,aspect='auto',origin='upper',extent=[epochs[0]-.5,epochs[-1]+.5,len(names)-.5,-.5],cmap='viridis');axes[1].set(yticks=np.arange(len(names)),yticklabels=names,xticks=epochs,xlabel='Completed replay epoch');axes[1].tick_params(axis='y',labelsize=7);plt.colorbar(p,ax=axes[1],label='Class median intra angle (degrees)')
    save(fig,'04_whole_class_intra_angles','Whole direction compactness: pair weighting and class weighting differ')
    rep=tables['representative_pair_angles'];pairs=list(dict.fromkeys((r['name_a'],r['name_b']) for r in rep));fig,axes=plt.subplots(3,1,figsize=(12,11))
    for ax,field,title in zip(axes,['centroid_angle','median','cone_gap_p95'],['Class-center separation','All cross-class sample pairs: median','Center angle minus two P95 cone radii']):
        matrix=np.array([[float(next(r[field] for r in rep if r['observation_id']==obs['observation_id'] and (r['name_a'],r['name_b'])==pair)) for obs in observations] for pair in pairs]);p=ax.imshow(matrix,aspect='auto',extent=[epochs[0]-.5,epochs[-1]+.5,len(pairs)-.5,-.5],cmap='coolwarm' if field=='cone_gap_p95' else 'viridis');ax.set(title=title,yticks=np.arange(len(pairs)),yticklabels=[a+' / '+b for a,b in pairs],xticks=epochs,xlabel='Completed replay epoch');plt.colorbar(p,ax=ax,label='Degrees')
    save(fig,'05_fixed_representative_pairs','The same11 class pairs are used at every epoch','Six predefined pairs plus five extra pairs selected once from the original TS13-best reference; not selected for favorable replay trends.')
    fig,axes=plt.subplots(4,3,figsize=(14,13));representative_epochs=[e for e in [0,1,5,10] if e in epochs]
    for ax,pair in zip(axes.flat,pairs):
        occupied=[]
        for obs in observations:
            if obs['epoch'] not in representative_epochs:continue
            values=[r for r in select('representative_pair_histograms',obs['observation_id']) if (r['name_a'],r['name_b'])==pair]
            values.sort(key=lambda r:float(r['bin_left']))
            if not values:raise ValueError('Missing actual representative-pair histogram')
            mass=np.asarray([float(r['percent']) for r in values]);assert abs(mass.sum()-100)<1e-7
            edges=[float(values[0]['bin_left'])]+[float(r['bin_right']) for r in values]
            ax.step(edges,np.r_[0,np.cumsum(mass)],where='post',label=f"e{obs['epoch']}")
            occupied.extend((float(r['bin_left']),float(r['bin_right'])) for r in values if float(r['percent'])>0)
        if occupied:ax.set_xlim(max(0,min(v[0] for v in occupied)-1),min(180,max(v[1] for v in occupied)+1))
        ax.set(title=pair[0]+' / '+pair[1],xlabel='Cross-class sample-pair angle (degrees)',ylabel='Cumulative pairs (%)',ylim=(0,102));ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(pairs):]:ax.set_visible(False)
    save(fig,'05b_representative_pair_ecdf','Direct distributions for all11 fixed representative class pairs','CDF from the same audit85 .25-degree cross-pair histograms; shared class-pair list across epochs. Display ranges retain every nonzero bin.')
    snapshots=[e for e in [0,1,5,10] if e in epochs];fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,e in zip(axes.flat,snapshots):
        obs=next(s for s in observations if s['epoch']==e);matrix=np.zeros((len(names),len(names)))
        for row in select('class_centroid_angles',obs['observation_id']):a,b=int(row['class_a']),int(row['class_b']);matrix[a,b]=matrix[b,a]=float(row['angle_degrees'])
        p=ax.imshow(matrix,vmin=0,vmax=180,cmap='viridis');ax.set(title=f'Fresh replay e{e}',xticks=np.arange(len(names)),yticks=np.arange(len(names)),xticklabels=names,yticklabels=names);ax.tick_params(axis='both',labelsize=5);ax.tick_params(axis='x',rotation=90);plt.colorbar(p,ax=ax,label='Degrees')
    for ax in list(axes.flat)[len(snapshots):]:ax.set_visible(False)
    save(fig,'06_class_center_matrices','Within-model40x40 class-center angles','Angles compare relations within each model; raw coordinate axes across checkpoints are not used as semantic alignment.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    curve(axes[0,0],'sample_order_strict_noncollision_percent','sample -> proxy');axes[0,0].set(ylabel='Noncollision draws, all3 strict (%)',ylim=(0,102));axes[0,0].legend()
    curve(axes[0,1],'proxy_order_strict_noncollision_percent','noncollision');curve(axes[0,1],'proxy_order_strict_noncollision_nonendpoint_percent','also exclude endpoint self-choice');axes[0,1].set(ylabel='Proxy -> proxy all3 strict (%)',ylim=(0,102));axes[0,1].legend()
    curve(axes[1,0],'proxy_collision_percent','pair/triple same ancestor');axes[1,0].set(ylabel='All proxy diagnostic draws (%)',ylim=(0,102));axes[1,0].legend()
    curve(axes[1,1],'proxy_pair_endpoint_self_percent','pair ancestor is endpoint');curve(axes[1,1],'proxy_triple_endpoint_self_percent','triple ancestor is endpoint');axes[1,1].set(ylabel='All proxy diagnostic draws (%)',ylim=(0,102));axes[1,1].legend()
    save(fig,'07_frozen_ancestor_orders','Audit85 common frozen readout: depth tolerance1e-3','Fixed audit CB64/K20/T50, deterministic minimax. Not this run\'s training CB32/K10-K20/Gumbel replay; denominators remain in public_stats/summary.json.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for field,label in [('sample_usage','sample: all draws'),('sample_effective_usage','sample: active noncollision'),('proxy_usage','proxy: all draws'),('proxy_effective_usage','proxy: active noncollision')]:curve(axes[0,0],field+'_effective_proxy_count',label)
    axes[0,0].set(ylabel='Actual training entropy-effective proxies');axes[0,0].legend(fontsize=7)
    for field,label in [('sample_usage','sample'),('proxy_usage','proxy')]:curve(axes[0,1],field+'_used_proxy_count',label)
    axes[0,1].set(ylabel='Actual training proxies selected at least once');axes[0,1].legend()
    curve(axes[1,0],'nn4_coverage_percent','different objects /9840');curve(axes[1,0],'nn4_top4_slot_percent','hottest4 objects /2048 slots');axes[1,0].axhline(100*512*4/9840,color='.5',ls='--',label='Coverage upper bound20.81%');axes[1,0].set(ylabel='Frozen NN4: stated population (%)');axes[1,0].legend(fontsize=7)
    curve(axes[1,1],'nn4_same_class_percent','NN4 all same class');axes[1,1].set(ylabel='Frozen proxy neighborhoods (%)',ylim=(0,102));axes[1,1].legend()
    save(fig,'08_actual_usage_and_static_hotspots','Training usage and static nearest-neighbor binding are separate','Actual usage belongs to CB32/Ksample10/Kproxy20 in the recorded epoch; initialization is N/A. Static NN4 binding is not usage.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,e in zip(axes.flat,snapshots):
        obs=next(s for s in observations if s['epoch']==e);values=select('proxy_stats',obs['observation_id']);association=obs['proxies']['depth_vs_class_breadth'];rho=association.get('spearman_depth_vs_nn32_entropy');text='N/A at1e-3 depth resolution' if rho is None else f'rho={rho:.3f}'
        ax.scatter([float(r['depth']) for r in values],[float(r['angular_nn32_entropy_nats']) for r in values],s=9,alpha=.5);ax.axvline(4.6,color='.5',ls='--');ax.set(title=f'e{e}; {text}',xlabel='Proxy origin depth c0.1',ylabel='NN32 class entropy (nats)')
    for ax in list(axes.flat)[len(snapshots):]:ax.set_visible(False)
    save(fig,'09_depth_and_class_breadth','Proxy depth vs supervised class-neighborhood breadth','Depth ranks follow audit85:1e-3 resolution with cap ties; class labels are not independent morphology ground truth.')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,field in zip(axes.flat,['depth_delta_epoch','depth_delta_init','raw_norm_delta_epoch','raw_norm_delta_init']):
        selected=[r for r in movement_hist if r['domain']==field];xs=sorted({r['bin_left'] for r in selected});matrix=np.array([[next(r['percent'] for r in selected if r['epoch']==e and r['bin_left']==x) for x in xs] for e in epochs]);p=ax.imshow(matrix,aspect='auto',origin='lower',extent=[min(xs),max(r['bin_right'] for r in selected),epochs[0]-.5,epochs[-1]+.5],vmin=0,cmap='magma');ax.axvline(0,color='cyan',ls='--');ax.set(title=field.replace('_',' '),xlabel='Signed scalar change: negative=inward',ylabel='Completed replay epoch',yticks=epochs);plt.colorbar(p,ax=ax,label='Proxies per bin (%)')
    save(fig,'10_paired_depth_movement_distributions','Paired movement distributions: each row sums to100% of512 proxies','Endpoint changes are not cumulative path length. Raw norm reveals outward motion hidden by the mapped cap. e0 deltas are defined as zero.')
    return files


def report(out, observations, rows, summary, figure_files, equivalence=None):
    fmt=lambda x:'N/A' if x is None else f'{float(x):.4g}'
    lines=['# B05均衡32＋旧HIER：前十轮同配置复跑与85号几何审查','',
           '**这是新增fresh同配置十轮复跑的e0—e10轨迹，不是补造原300轮任务缺失的历史权重。** 原任务可用e1／e5单独作为辅助读出，不与复跑串接；本审查不包含PartHier。', '',
           '主配置为B05 c0.1／cap2.3／基础GS／500/N与2，CB32＝16类×2有放回，样本K10／代理K20，代理AdamW LR0.01／eps1e-8，seed4780。训练前十轮沿用300轮调度及原正式评估路径；下述几何提取只用9840训练对象clean first1024、canonical batch32、eval保存BN，不校准、不优化。','',
           '## 同85号的测量口径','',
           '全量whole深度、类内全对／780类对／40×40中心角、P95类锥间隔、固定11代表类对、代理深度／NN4与NN32绑定、稳健祖先次序及使用率，均复用原85 analyze.py计算和CSV定义。百分比whole每轮各按9840归一，proxy每轮各按512归一；构造HIER壳4.6单独标注，不当作自然学出的薄壳。','',
           '**共同冻结诊断仍按85号CB64／K20／T50、seed4780和确定性minimax。** 它便于跨审查保持共同读出，不是本任务真实训练CB32／K10-K20／Gumbel的重放。真实训练usage只使用该轮保存的sample／proxy计数，e0无usage标N/A；非碰撞、非端点自选分母和1e-4／1e-3／1e-2容差均保留在完整summary中。','',
           '## 每轮核心轨迹','',
           '| 轮末 | raw越cap／512 | native whole深度中位 | proxy深度P5／中位／P95 | 类内全对角中位° | 异类全对角中位° | 相对初始化Δ深度中位 | 比自身初始化更浅比例 |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| e{r['epoch']} | {r['cap_raw_count']} | {fmt(r['whole_depth_median'])} | {fmt(r['proxy_depth_p05'])}／{fmt(r['proxy_depth_median'])}／{fmt(r['proxy_depth_p95'])} | {fmt(r['intra_angle_pair_weighted_median'])} | {fmt(r['inter_angle_pair_weighted_median'])} | {fmt(r['depth_delta_init_median'])} | {fmt(r['depth_delta_init_inward_percent'])}% |")
    end=rows[-1];lines+=['',f"末轮有{end['cap_raw_count']}/{end['proxies']}个raw代理越cap；raw模长中位{fmt(end['raw_norm_median'])}、最大{fmt(end['raw_norm_max'])}。每个代理相对自己初始化的映射深度位移中位为{fmt(end['depth_delta_init_median'])}；它不等于两次总体中位数相减，也不是累计移动路程。",'',
        '当前raw越cap数量不是历史越界次数。每轮快照能给轮末存量和相邻轮末变化，不能推断轮内越界／返回事件；若需这些事件，应另读训练逐步记录，而不是从轮末CDF补造。','',
        '## 角度、具体类别与代理次序','',
        f"e0→e{end['epoch']}，whole类内全对角中位{fmt(rows[0]['intra_angle_pair_weighted_median'])}°→{fmt(end['intra_angle_pair_weighted_median'])}°；类间全对角中位{fmt(rows[0]['inter_angle_pair_weighted_median'])}°→{fmt(end['inter_angle_pair_weighted_median'])}°。完整40类和780类对表同时保留，不用总体均值代替局部困难类对。",'',
        f"末轮共同冻结诊断中，非碰撞样本→代理三项严格次序为{fmt(end['sample_order_strict_noncollision_percent'])}%；代理→代理为{fmt(end['proxy_order_strict_noncollision_percent'])}%，再排除端点自选后为{fmt(end['proxy_order_strict_noncollision_nonendpoint_percent'])}%。主容差1e-3；必须连同完整分母、碰撞和端点自选比例解释。",'',
        '代理回内、代理内部成树、类别邻域绑定和分类收益是不同命题。监督类标签参与了训练；NN同类纯度和深度关联不能独自证明独立语义／形态层次。训练test指标来自原训练路径，本几何审查不再以test挑类别或调阈值。','']
    original=[s for s in summary['observations'] if s['run_id']==ORIGINAL]
    if original:
        lines+=['## 原300轮任务现存检查点：辅助对照','',
                '| 原历史轮末 | raw越cap | native whole深度中位 | proxy深度中位 | 类内全对角中位° |','|---|---:|---:|---:|---:|']
        for s in sorted(original,key=lambda s:s['epoch']):
            lines.append(f"| e{s['epoch']} | {s['proxies']['raw_above_tangent_cap_count']} | {fmt(s['domains']['native_whole']['depth']['median'])} | {fmt(s['proxies']['depth']['median'])} | {fmt(s['angles']['intra_pair_weighted']['median'])} |")
        lines+=['','这些是原任务自己的已保存权重，不是复跑阶段别名。相同seed／配置不自动承诺不同硬件环境下逐位重放；初始化和e1/e5一致性须按采集验收另核实。','']
    if equivalence:
        lines+=['## 原任务／复跑的e1、e5参数一致性','',
                '| 轮末 | 参数组 | 逐位相同 | 最大绝对差 | 平均绝对差 |','|---|---|---|---:|---:|']
        for epoch,components in equivalence.items():
            for component,value in components.items():
                lines.append(f"| e{epoch} | {component} | {'是' if value['bitwise_equal'] else '否'} | {fmt(value['max_abs_difference'])} | {fmt(value['mean_abs_difference'])} |")
        lines+=['','首批输入身份一致与训练截至轮末参数逐位相同是不同约束。若此表存在差异，复跑仍应称同配置新轨迹，不能称原任务全部缺失轮次的精确恢复。原任务e1／e5的独立几何读出保留为参照。','']
    lines+=['## 图表与完整数据','']
    for item in figure_files:
        if item['file'].endswith('.png'):
            lines.append(f"![{Path(item['file']).stem}]({item['file']})");lines.append('')
    lines+=['原样85统计保存在[public_stats/summary.json](public_stats/summary.json)，以及该目录的40类、780类对、类中心、深度直方图／ECDF、固定代表类对和逐代理统计CSV。', '',
            '[逐轮轨迹表](trajectory_summary.csv)、[逐代理标量位移](proxy_depth_movements.csv)、[位移百分比直方图](movement_histograms.csv)、[轨迹与口径验收](trajectory_verification.json)。逐代理表仅有深度／raw模长标量，不含嵌入坐标、对象ID或点云。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():raise FileExistsError('Postprocessing requires a new output directory')
    started=time.monotonic();args.out.mkdir(parents=True)
    private_manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    equivalence=equivalence_scalars(private_manifest.get('replay_equivalence'))
    audit=audit85_module();audit.main(['--manifest',str(args.manifest),'--out',str(args.out/'public_stats')])
    summary=json.loads((args.out/'public_stats/summary.json').read_text(encoding='utf-8'))
    observations=replay_observations(summary)
    tables={name:read_csv(args.out/'public_stats'/(name+'.csv')) for name in ['depth_histograms','depth_cdf','class_intra_angles','class_centroid_angles','representative_pair_angles','representative_pair_histograms','proxy_stats']}
    scalars,movement=proxy_trajectories(observations,tables['proxy_stats']);rows=summary_rows(observations,movement);hist=movement_histograms(scalars)
    write_csv(args.out/'trajectory_summary.csv',rows);write_csv(args.out/'proxy_depth_movements.csv',scalars);write_csv(args.out/'movement_histograms.csv',hist)
    figures=make_figures(args.out,observations,rows,tables,scalars,hist,summary['class_names'])
    verification={'schema':'b05-cb32-a0-early-audit85-postprocess-1','status':'completed','replay_epochs':[s['epoch'] for s in observations],
        'replay_run_id':REPLAY,'original_epochs_are_auxiliary':True,'optimizer_updates_by_postprocessor':0,'GPU_used_by_postprocessor':False,
        'audit85_analyze_sha256':sha(AUDIT85),'proxy_ID_movement_scope':'this one fresh replay only; not semantic matching between original/replay',
        'depth_movement_tolerance':TOL,'within_epoch_crossing_events_inferred':False,'training_graph':{'batch':32,'sample_K':10,'proxy_K':20,'choice':'stochastic Gumbel'},
        'frozen_audit85_graph':{'batch':64,'classes':32,'objects_per_class':2,'K':20,'T':50,'batches':8,'seed':4780,'choice':'deterministic minimax'},
        'e0_training_usage':'N/A','replay_equivalence':equivalence,'figures':figures,'elapsed_seconds':time.monotonic()-started}
    (args.out/'trajectory_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report(args.out,observations,rows,summary,figures,equivalence)
    print(json.dumps({'status':'completed','epochs':len(observations),'figures':len(figures)//2,'out':str(args.out)}),flush=True)


if __name__=='__main__':main()
