"""CPU-only paired B05 eps diagnostic tables, distributions and report.

Consumes completed private observer scalar/one-dimensional depth records. It
never loads a model/checkpoint, allocates CUDA, or reads point clouds/test data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

QUANTILES = {"min": 0., "p01": .01, "p05": .05, "p10": .1, "p25": .25,
             "median": .5, "p75": .75, "p90": .9, "p95": .95, "p99": .99, "max": 1.}
ARM_LABELS = {"eps1e-8": "eps=1e-8", "eps1e-4": "eps=1e-4 (A1)"}
MOVEMENT_TOLERANCE = 1e-5


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Distribution must be nonempty and finite")
    result = dict(count=len(values), mean=float(values.mean()), std_population=float(values.std()))
    result.update({name: float(value) for name, value in zip(QUANTILES, np.quantile(values, list(QUANTILES.values())))})
    return result


def movement(values, tolerance=MOVEMENT_TOLERANCE):
    values = np.asarray(values, np.float64).reshape(-1)
    result = describe(values)
    result.update(tolerance=tolerance, outward_percent=float(100*np.mean(values > tolerance)),
                  inward_percent=float(100*np.mean(values < -tolerance)),
                  stationary_percent=float(100*np.mean(np.abs(values) <= tolerance)))
    return result


def centered_edges(values, width, *, signed):
    if width <= 0:
        raise ValueError("Positive bin width required")
    values = np.asarray(values, np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Finite histogram pool required")
    extent = max(width, math.ceil(float(np.abs(values).max())/width)*width)
    if signed:
        centers = np.arange(-extent, extent+width/2, width)
        edges = np.r_[centers-width/2, centers[-1]+width/2]
    else:
        centers = np.arange(0., extent+width/2, width)
        edges = np.r_[0., centers[1:]-width/2, centers[-1]+width/2]
    return edges


def histogram_percent(values, edges):
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty histogram required")
    counts, _ = np.histogram(values, edges)
    if int(counts.sum()) != len(values):
        raise ValueError("Histogram bins omit points")
    return counts.astype(np.float64)*100/len(values)


def flatten_scalars(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix+"."+str(key) if prefix else str(key)
            result.update(flatten_scalars(item, name))
    elif isinstance(value, (int, float, bool)) and value is not None:
        if not math.isfinite(float(value)):
            raise ValueError("Nonfinite observer scalar: "+prefix)
        result[prefix] = float(value)
    return result


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_arm(directory, key, expected_epochs, expected_steps):
    directory = Path(directory)
    eps = 1e-8 if key == "eps1e-8" else 1e-4
    epochs, sources = [], []
    for epoch in range(expected_epochs+1):
        path = directory / ("observer_epoch_%03d.json" % epoch)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if int(value["epoch"]) != epoch or float(value["eps"]) != eps:
            raise ValueError("Observer epoch/eps identity differs")
        for field in ("proxy_depth", "raw_tangent_norm"):
            array = np.asarray(value[field], np.float64)
            if array.shape != (512,) or not np.isfinite(array).all():
                raise ValueError("Expected512 finite scalar proxy observations: "+field)
            value[field] = array
        if np.any(value["proxy_depth"] < 0) or np.any(value["proxy_depth"] > 4.6001):
            raise ValueError("Mapped proxy depth outside approved c.1/cap2.3 geometry")
        cap = int(np.count_nonzero(value["raw_tangent_norm"] > 2.3))
        if "cap_count" in value and int(value["cap_count"]) != cap:
            raise ValueError("Raw norm cap count differs from recorded count")
        value["cap_count"] = cap
        epochs.append(value)
        sources.append(dict(file=path.name, sha256=digest(path)))
    step_path = directory / "proxy_steps.jsonl"
    with step_path.open(encoding="utf-8-sig") as stream:
        steps = [json.loads(line) for line in stream if line.strip()]
    if len(steps) != expected_epochs*expected_steps:
        raise ValueError("Expected all actual optimizer-step observations")
    coordinates = {(int(row["epoch"]), int(row["step"])) for row in steps}
    expected = {(e, s) for e in range(1, expected_epochs+1) for s in range(1, expected_steps+1)}
    if coordinates != expected or len(coordinates) != len(steps):
        raise ValueError("Step observations are repeated, missing or not epoch-local")
    sources.append(dict(file=step_path.name, sha256=digest(step_path)))
    for epoch, value in enumerate(epochs):
        value["derived_depth_delta_epoch"] = value["proxy_depth"]-(epochs[epoch-1]["proxy_depth"] if epoch else value["proxy_depth"])
        value["derived_depth_delta_init"] = value["proxy_depth"]-epochs[0]["proxy_depth"]
        value["derived_raw_delta_epoch"] = value["raw_tangent_norm"]-(epochs[epoch-1]["raw_tangent_norm"] if epoch else value["raw_tangent_norm"])
        value["derived_raw_delta_init"] = value["raw_tangent_norm"]-epochs[0]["raw_tangent_norm"]
        for source, derived in (("depth_delta_epoch", "derived_depth_delta_epoch"), ("depth_delta_init", "derived_depth_delta_init"),
                                ("raw_norm_delta_epoch", "derived_raw_delta_epoch"), ("raw_norm_delta_init", "derived_raw_delta_init")):
            if source in value and not np.allclose(value[source], value[derived], rtol=1e-7, atol=2e-7):
                raise ValueError("Recorded depth/raw movement differs from endpoint observations")
    return dict(key=key, label=ARM_LABELS[key], eps=eps, epochs=epochs, steps=steps, directory=directory, sources=sources)


def epoch_tables(arm):
    rows, proxy_rows, step_rows, audit_rows = [], [], [], []
    for step in arm["steps"]:
        flat = flatten_scalars(step)
        step_rows.append(dict(arm=arm["key"], **flat))
    for value in arm["epochs"]:
        epoch = int(value["epoch"])
        row = dict(arm=arm["key"], epoch=epoch, eps=arm["eps"], proxies=512, cap_count=value["cap_count"])
        for field in ("ever_cap_count", "returned_ever_count", "first_crossings_epoch", "returns_epoch"):
            row[field] = value.get(field)
        for name, array in (("depth", value["proxy_depth"]), ("raw_norm", value["raw_tangent_norm"]),
                            ("depth_delta_epoch", value["derived_depth_delta_epoch"]), ("depth_delta_init", value["derived_depth_delta_init"]),
                            ("raw_delta_epoch", value["derived_raw_delta_epoch"]), ("raw_delta_init", value["derived_raw_delta_init"])):
            summary = movement(array) if "delta" in name else describe(array)
            row.update({name+"_"+key: item for key, item in summary.items()})
        epoch_steps = [s for s in step_rows if int(s["epoch"]) == epoch]
        keys = sorted(set().union(*(set(s) for s in epoch_steps))) if epoch_steps else []
        for key in keys:
            if key in ("arm", "epoch", "step"):
                continue
            values = [s[key] for s in epoch_steps if key in s]
            if len(values) != len(epoch_steps) and key in ("proxy_gradient_norm","actual_update_norm","sqrt_vhat_median","eps_dominance_fraction"):
                raise ValueError("A required step scalar is missing within an epoch: "+key)
            stat = describe(values)
            row.update({"step_"+key+"_"+name: stat[name] for name in ("mean", "median", "p05", "p95", "min", "max")})
            row["step_"+key+"_observed_steps"] = len(values)
        if epoch:
            audit_path = arm["directory"] / ("parameter_audit_epoch_%03d.json" % epoch)
            audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
            group = audit["groups"]["shared_encoder"]
            readout = dict(arm=arm["key"], epoch=epoch, sampling="first batch only; weighted HIER gradient; saved BN/RNG restored",
                           weighted_hier_norm=group["weighted_hier_norm"], base_norm=group["base_norm"],
                           hier_base_ratio=group["hier_vs_base"]["norm_ratio"], hier_base_cosine=group["hier_vs_base"]["cosine"])
            audit_rows.append(readout)
            row.update({"backbone_"+key: value for key, value in readout.items() if isinstance(value, (int,float)) and key != "epoch"})
            arm["sources"].append(dict(file=audit_path.name,sha256=digest(audit_path)))
        rows.append(row)
        for index in range(512):
            proxy_rows.append(dict(arm=arm["key"],epoch=epoch,proxy_index=index,depth=float(value["proxy_depth"][index]),
                                   raw_tangent_norm=float(value["raw_tangent_norm"][index]),
                                   depth_delta_epoch=float(value["derived_depth_delta_epoch"][index]),
                                   depth_delta_init=float(value["derived_depth_delta_init"][index]),
                                   raw_norm_delta_epoch=float(value["derived_raw_delta_epoch"][index]),
                                   raw_norm_delta_init=float(value["derived_raw_delta_init"][index])))
    return rows,proxy_rows,step_rows,audit_rows


def distribution_tables(arms):
    domains = {"depth": ("proxy_depth", .05, False), "raw_norm": ("raw_tangent_norm", .05, False),
               "depth_delta_epoch": ("derived_depth_delta_epoch", .01, True), "depth_delta_init": ("derived_depth_delta_init", .01, True),
               "raw_delta_epoch": ("derived_raw_delta_epoch", .01, True), "raw_delta_init": ("derived_raw_delta_init", .01, True)}
    histograms,cdfs,binning = [],[],{}
    for domain,(field,width,signed) in domains.items():
        all_values = np.concatenate([value[field] for arm in arms for value in arm["epochs"]])
        edges = centered_edges(all_values,width,signed=signed)
        grid = np.unique(np.r_[edges,0.,4.6] if domain == "depth" else np.r_[edges,0.])
        binning[domain] = dict(width=width,edges=edges.tolist(),grid=grid.tolist(),normalization="512 proxies independently per arm/epoch")
        for arm in arms:
            for value in arm["epochs"]:
                values = value[field]
                percent = histogram_percent(values,edges)
                for left,right,share in zip(edges[:-1],edges[1:],percent):
                    histograms.append(dict(arm=arm["key"],epoch=int(value["epoch"]),domain=domain,
                                           left=float(left),right=float(right),percent=float(share),population=512))
                ordered = np.sort(values)
                cumulative = 100*np.searchsorted(ordered,grid,side="right")/len(values)
                for point,share in zip(grid,cumulative):
                    cdfs.append(dict(arm=arm["key"],epoch=int(value["epoch"]),domain=domain,value=float(point),cumulative_percent=float(share),population=512))
    return histograms,cdfs,binning


def save_figures(out, arms, epoch_rows, audit_rows, histograms, cdfs, binning):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    figure_dir=out/"figures";figure_dir.mkdir(exist_ok=True)
    figures=[];colors={"eps1e-8":"#d95f02","eps1e-4":"#1b9e77"}
    def finish(fig,name,title):
        fig.suptitle(title)
        fig.tight_layout(rect=(0,0,1,.96))
        for extension in ("png","svg"):
            path=figure_dir/(name+"."+extension);fig.savefig(path,dpi=180,bbox_inches="tight")
            figures.append(dict(file="figures/"+path.name,sha256=digest(path)))
        plt.close(fig)
    def curve(ax,field,*,ylabel=None,log=False):
        available=False
        for arm in arms:
            rows=[r for r in epoch_rows if r["arm"]==arm["key"] and r["epoch"]>0 and r.get(field) is not None]
            if not rows:continue
            y=np.asarray([r[field] for r in rows],float)
            if log:y=np.where(y>0,y,np.nan)
            ax.plot([r["epoch"] for r in rows],y,"o-",label=arm["label"],color=colors[arm["key"]],markersize=3)
            available=True
        if not available:ax.text(.5,.5,"No recorded metric: "+field,ha="center",va="center",transform=ax.transAxes,wrap=True)
        if log:ax.set_yscale("log")
        ax.set_xlabel("Completed epoch");ax.set_ylabel(ylabel or field);ax.grid(alpha=.2)
        if available:ax.legend(fontsize=8)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,key,label in zip(axes,("cap_count","ever_cap_count","returned_ever_count"),("Beyond tangent cap2.3","Ever crossed (unique proxies)","Ever returned (unique proxies)")):curve(ax,key,ylabel=label)
    finish(fig,"01_proxy_cap_crossings","Proxy cap crossing and return: identical LR0.005, only eps differs")
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    fields=(("step_proxy_gradient_norm_mean","Proxy optimizer-input gradient L2",True),("step_actual_update_norm_mean","Actual tangent update L2",True),
            ("step_sqrt_vhat_median_mean","Median sqrt(v_hat)",True),("step_eps_dominance_fraction_mean","Fraction of coordinates dominated by eps",False),
            ("step_update_attenuation_median_mean","Median sqrt(v_hat)/(sqrt(v_hat)+eps)",False),
            ("step_fixed_moment_eps1e4_vs_eps1e8_update_norm_ratio_mean","Same moments: update L2(eps1e-4) / L2(eps1e-8)",False))
    for ax,(key,label,log) in zip(axes.flat,fields):curve(ax,key,ylabel=label,log=log)
    finish(fig,"02_gradients_updates_denominator","Raw gradient, actual update and AdamW denominator are different quantities")
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,field,label in zip(axes,("backbone_weighted_hier_norm","backbone_base_norm","backbone_hier_base_ratio"),("Weighted HIER gradient L2","Base gradient L2","HIER / base L2")):curve(ax,field,ylabel=label,log=True)
    finish(fig,"03_backbone_hier_gradient","Shared-backbone parameter gradients: first training batch of each epoch only")
    for domain,name,title in (("depth","04_depth_percentage_heatmap","Mapped proxy depth distributions; each epoch sums to100%"),
                              ("depth_delta_epoch","06_epoch_depth_movement_heatmap","Mapped depth movement since previous epoch (positive=outward)"),
                              ("depth_delta_init","07_cumulative_depth_movement_heatmap","Mapped depth movement since initialization (positive=outward)"),
                              ("raw_norm","08_raw_tangent_norm_heatmap","Raw tangent norm: movements beyond cap remain visible")):
        edges=np.asarray(binning[domain]["edges"])
        fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True,layout="constrained")
        data=[]
        for arm in arms:
            selected=[r for r in histograms if r["arm"]==arm["key"] and r["domain"]==domain]
            matrix=np.asarray([[r["percent"] for r in selected if r["epoch"]==epoch] for epoch in range(len(arm["epochs"]))])
            data.append(matrix)
        vmax=max(float(matrix.max()) for matrix in data)
        for ax,arm,matrix in zip(axes,arms,data):
            image=ax.pcolormesh(edges,np.arange(matrix.shape[0]+1)-.5,matrix,cmap="magma",vmin=0,vmax=vmax,shading="flat")
            ax.set_title(arm["label"]);ax.set_ylabel("Completed epoch");ax.set_xlabel("Origin depth c0.1" if domain=="depth" else domain)
            ax.set_yticks(range(matrix.shape[0]))
            ax.axvline(4.6 if domain=="depth" else 2.3 if domain=="raw_norm" else 0,color="#00b8d9",linestyle="--",linewidth=1)
        fig.colorbar(image,ax=axes,label="Proxies per bin (%)",fraction=.025,pad=.02)
        # Constrained layout reserves its own colorbar axis.
        fig.suptitle(title)
        for extension in ("png","svg"):
            path=figure_dir/(name+"."+extension);fig.savefig(path,dpi=180,bbox_inches="tight");figures.append(dict(file="figures/"+path.name,sha256=digest(path)))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,arm in zip(axes,arms):
        for epoch in sorted(set((0,1,min(5,len(arm["epochs"])-1),len(arm["epochs"])-1))):
            rows=[r for r in cdfs if r["arm"]==arm["key"] and r["domain"]=="depth" and r["epoch"]==epoch]
            ax.step([r["value"] for r in rows],[r["cumulative_percent"] for r in rows],where="post",label="epoch%d"%epoch)
        ax.axvline(4.6,color="black",linestyle="--",linewidth=1);ax.set_xlabel("Origin depth c0.1");ax.set_ylabel("Proxies at/below depth (%)");ax.set_title(arm["label"]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    finish(fig,"05_depth_ecdf","Mapped depth ECDF; cap reference4.6, no KDE smoothing")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,domain in zip(axes,("depth_delta_epoch","depth_delta_init")):
        for arm in arms:
            rows=[r for r in epoch_rows if r["arm"]==arm["key"] and r["epoch"]>0]
            x=[r["epoch"] for r in rows];median=[r[domain+"_median"] for r in rows]
            ax.plot(x,median,"o-",color=colors[arm["key"]],label=arm["label"])
            ax.fill_between(x,[r[domain+"_p05"] for r in rows],[r[domain+"_p95"] for r in rows],alpha=.15,color=colors[arm["key"]])
        ax.axhline(0,color="black",linewidth=.8);ax.set_xlabel("Completed epoch");ax.set_ylabel(domain+" (median and P5-P95)");ax.grid(alpha=.2);ax.legend(fontsize=8)
    finish(fig,"09_depth_movement_quantiles","Signed mapped-depth movements: initial and per-epoch changes are separate")
    (out/"figures_manifest.json").write_text(json.dumps(dict(figures=figures,normalization="512 proxies separately per arm/epoch",script_sha256=digest(__file__)),indent=2),encoding="utf-8")
    return figures


def build_report(out, summary, rows):
    def num(value, digits=4):return "N/A" if value is None else f"{value:.{digits}g}"
    lines=["# B05双卡：AdamW eps单变量十轮审查","",
           "两组均从相同seed4780初始化，严格B05、双卡CB64、K20/K20、每轮200步；代理初始LR均0.005、最低LR0.0005，仅eps分别为1e-8与1e-4。实际运行前10轮，余弦调度保留300轮时间尺度。不进行test前向或选模。","",
           "eps不修改同一计算图反向得到的原始梯度；它进入AdamW分母，改变梯度最终产生的参数更新。固定状态下更新为 `−lr·m_hat/(sqrt(v_hat)+eps)`，另有独立weight decay。初步同梯度情形若尺度为1e-5，eps1e-4相对1e-8的适应项约为9.1%；尺度为1e-6时约为1%。这不是对整段训练更新量的固定缩放：参数轨迹改变后，后续梯度、关系选择和动量也会改变。","",
           "## 每轮cap及深度移动","",
           "原始cap计数按raw tangent norm>2.3；映射深度上限约4.6。位移正值代表向外，负值代表向内；基本不动以绝对值≤1e-5为准。首次越界与返回次数来自逐步观察器，不能仅从每轮末态推断。","",
           "| eps | 轮次 | 越cap | 曾越cap | 曾返回 | 深度P5／中位／P95 | 相对上轮Δ深度中位 | 相对初始Δ深度中位 | 向内／向外比例 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if not row["epoch"]:continue
        depth="／".join(num(row["depth_"+key]) for key in ("p05","median","p95"))
        lines.append(f"| {row['arm']} | {row['epoch']} | {row['cap_count']} | {row.get('ever_cap_count','N/A')} | {row.get('returned_ever_count','N/A')} | {depth} | {num(row['depth_delta_epoch_median'])} | {num(row['depth_delta_init_median'])} | {row['depth_delta_epoch_inward_percent']:.1f}%／{row['depth_delta_epoch_outward_percent']:.1f}% |")
    lines += ["","![越cap与返回](figures/01_proxy_cap_crossings.png)","","![当前深度百分比分布](figures/04_depth_percentage_heatmap.png)","",
              "[深度ECDF](figures/05_depth_ecdf.png)、[逐轮深度位移](figures/06_epoch_depth_movement_heatmap.png)、[累计深度位移](figures/07_cumulative_depth_movement_heatmap.png)、[raw模长分布](figures/08_raw_tangent_norm_heatmap.png)、[位移分位区间](figures/09_depth_movement_quantiles.png)。每张分布按各自512个代理归一化到100%，未作KDE平滑。","",
              "## 原始HIER梯度与实际更新","",
              "代理梯度是实际同步后送入AdamW的切空间参数梯度，已经包含HIER权重0.5及前向映射的反向路径；主干梯度是每轮首批的独立参数分项审查，保留其BN／随机流，不能冒充整轮均值。","",
              "| eps | 轮次 | 整轮代理梯度L2均值 | 实际参数位移L2均值 | sqrt(v_hat)坐标中位的轮均值 | eps主导坐标比例轮均值 | 首批主干HIER/base梯度比 |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if not row["epoch"]:continue
        keys=("step_proxy_gradient_norm_mean","step_actual_update_norm_mean","step_sqrt_vhat_median_mean","step_eps_dominance_fraction_mean","backbone_hier_base_ratio")
        lines.append("| "+" | ".join((row["arm"],str(row["epoch"]),*(num(row.get(k)) for k in keys)))+" |")
    lines += ["","![梯度、更新与分母](figures/02_gradients_updates_denominator.png)","","![主干HIER梯度](figures/03_backbone_hier_gradient.png)","",
              "完整[每轮统计](epoch_summary.csv)、[逐步标量](step_metrics.csv)、[逐代理深度及位移](proxy_depth_movements.csv)、[百分比直方图](distribution_histograms.csv)、[ECDF](distribution_ecdf.csv)、[主干分项梯度](backbone_gradient_audits.csv)及[验收摘要](summary.json)均已保存。逐代理表只有标量深度／模长，没有嵌入坐标或样本ID。","",
              "本次是同学习率的eps单变量短轨迹比较；不据10轮结果推断完整300轮分类效果。cap后的深度不再反映继续增长的raw模长，所以必须结合raw模长、原始梯度、Adam状态与实际位移判断。若后期原始梯度不同，这可能是两条轨迹已分叉的结果，不能称eps直接改写了反向梯度。",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-run","--small-dir",dest="small",type=Path,required=True)
    parser.add_argument("--large-run","--large-dir",dest="large",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-epochs",type=int,default=10)
    parser.add_argument("--expected-steps",type=int,default=200)
    parser.add_argument("--no-plot",action="store_true",help="Only for pure CPU numerical tests; normal diagnostic produces figures")
    args=parser.parse_args(argv)
    if args.expected_epochs<1 or args.expected_steps<1:parser.error("Positive expected epochs/steps required")
    if args.out.exists():raise FileExistsError("Analysis must use a new output directory")
    started=time.monotonic()
    arms=[read_arm(args.small,"eps1e-8",args.expected_epochs,args.expected_steps),read_arm(args.large,"eps1e-4",args.expected_epochs,args.expected_steps)]
    for field in ("proxy_depth","raw_tangent_norm"):
        if not np.array_equal(arms[0]["epochs"][0][field],arms[1]["epochs"][0][field]):raise ValueError("Paired proxy initialization differs")
    parameter_hashes=[arm["epochs"][0].get("initial_proxy_parameter_sha256") for arm in arms]
    if any(parameter_hashes) and (not all(parameter_hashes) or parameter_hashes[0]!=parameter_hashes[1]):
        raise ValueError("Paired raw proxy parameter identity differs")
    args.out.mkdir(parents=True)
    rows,proxy_rows,steps,audits=[],[],[],[]
    for arm in arms:
        values=epoch_tables(arm)
        for destination,new in zip((rows,proxy_rows,steps,audits),values):destination.extend(new)
    histograms,cdfs,binning=distribution_tables(arms)
    for name,values in (("epoch_summary",rows),("proxy_depth_movements",proxy_rows),("step_metrics",steps),("backbone_gradient_audits",audits),("distribution_histograms",histograms),("distribution_ecdf",cdfs)):
        csv_write(args.out/(name+".csv"),values)
    summary=dict(schema="b05-eps-single-variable-10e-1",status="completed",expected_epochs=args.expected_epochs,steps_per_epoch=args.expected_steps,
                 schedule_epochs=300,initial_proxy_lr=.005,eps_values=[1e-8,1e-4],seed=4780,global_batch=64,sample_K=20,proxy_K=20,
                 paired_scalar_initialization_identical=True,proxy_population=512,movement_tolerance=MOVEMENT_TOLERANCE,
                 paired_initial_proxy_parameter_sha256=parameter_hashes[0],
                 gradient_semantics="Actual rank-synchronized optimizer-input tangent gradient, weighted by lambda0.5; not altered in-place by eps",
                 backbone_gradient_semantics="First batch each epoch; weighted HIER and base parameter gradients; BN/RNG restored",
                 depth_units="Physical origin hyperbolic distance at c0.1; mapped cap about4.6; raw tangent cap2.3",
                 count_semantics="Cross/return events and unique counts are observer values; epoch endpoint deltas are not within-epoch event counts",
                 binning=binning,epochs=rows,sources={arm["key"]:arm["sources"] for arm in arms},
                 interpretation_limits=["Ten-epoch optimization diagnostic, not complete classification evidence","Raw gradients can diverge after parameter trajectories diverge","Mapped cap hides continued outward raw-parameter motion","A fixed-state eps attenuation is not a constant whole-training learning-rate rescaling"])
    if not args.no_plot:summary["figures"]=save_figures(args.out,arms,rows,audits,histograms,cdfs,binning)
    summary["elapsed_seconds"]=time.monotonic()-started
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    if not args.no_plot:build_report(args.out,summary,rows)
    print(json.dumps(dict(status="completed",epochs=args.expected_epochs,step_rows=len(steps),proxy_scalar_rows=len(proxy_rows),out=str(args.out))),flush=True)


if __name__=="__main__":main()
