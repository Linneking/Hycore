"""Build a Chinese offline report from validated frozen HIER audit results.

No private input paths or sample/proxy identities are embedded in this source.
The report is a private artifact. It uses real metrics only, records input SHA,
and keeps cache replay separate from optional fresh-image zero-update epochs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import struct
from pathlib import Path

DATASETS = ("cub", "cars", "sop")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summary_path(value, geometry=False):
    path = Path(value)
    if path.is_file():
        return path
    choices = [path / "summary.json"]
    if geometry:
        choices.append(path / "analysis" / "summary.json")
    matches = [p for p in choices if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one summary at {path}, found {matches}")
    return matches[0]


def load(path, identities):
    path = Path(path)
    identities[str(path.resolve())] = sha256(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(v, digits=3):
    if v is None:
        return "未记录"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}g}" if abs(v) < .001 and v != 0 else f"{v:.{digits}f}"
    return str(v)


def table(headers, rows):
    head = "".join(f"<th tabindex='0'>{html.escape(str(c))}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(number(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class='tablebox'><table class='sortable'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_csv(path, headers, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers); writer.writerows(rows)


def png_size(path):
    with Path(path).open("rb") as f:
        block = f.read(24)
    if block[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG: {path}")
    return struct.unpack(">II", block[16:24])


def copy_figure(source, output, name, caption, identities, copied, metrics_link=None):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    identities[str(source.resolve())] = sha256(source)
    destination = output / "figures" / f"{name}.png"
    shutil.copyfile(source, destination)
    size = png_size(destination)
    if min(size) < 400:
        raise ValueError(f"Figure too small: {source}: {size}")
    copied.append({"name": name, "relative_path": f"figures/{name}.png", "width": size[0], "height": size[1],
                   "source_sha256": identities[str(source.resolve())]})
    provenance = f" · <a href='{html.escape(metrics_link)}'>源统计</a>" if metrics_link else ""
    return f"<figure><a href='figures/{name}.png' target='_blank'><img loading='lazy' src='figures/{name}.png' alt='{html.escape(caption)}'></a><figcaption>{html.escape(caption)}{provenance}</figcaption></figure>"


def role_count(r, key):
    return r.get("roles", {}).get(key, {}).get("used_proxies")


def gradient(r, item, mode, key):
    return r.get("gradients", {}).get(f"{item}_{mode}", {}).get(key)


def image_epoch_data(values, identities):
    data, pending = {}, []
    for value in values:
        path = summary_path(value)
        item = load(path, identities)
        ds = item.get("dataset")
        if ds not in DATASETS or ds in data:
            raise ValueError(f"Unknown or duplicate image-epoch dataset: {ds}")
        result = item.get("results", {})
        valid = (result.get("full_epoch_completed") is True
                 and result.get("replay_batches") == result.get("full_source_epoch_batches")
                 and item.get("optimizer_updates") == 0
                 and item.get("strict_original_resume") is False)
        data[ds] = {"payload": item, "results": result, "complete": valid, "path": path}
        if not valid:
            pending.append(ds)
    return data, pending


def plot_image_epoch(image, direction, extra, output, copied):
    """Aggregate-only figure; fresh-image results must pass completion checks."""
    complete={ds:item for ds,item in image.items() if item["complete"]}
    if not complete:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    metrics={"image_results":{ds:item["results"] for ds,item in complete.items()},
             "cache_gradients":{ds:extra["results"][f"{ds}_train"]["gradients"] for ds in complete},
             "nn4_objects":{ds:direction["models"][ds]["retrieval_official"]["unique_top4_objects"] for ds in complete}}
    (output/"metrics"/"image_epoch_figure_sources.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,axs=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    labels=["NN4 whole", "Sample pair +hinge", "Sample triple +hinge", "Proxy pair +hinge", "Proxy triple +hinge", "Sample ST gradient", "Proxy ST gradient"]
    for col,ds in enumerate(DATASETS):
        if ds not in complete:
            for ax in axs[:,col]:
                ax.text(.5,.5,f"{ds.upper()}: pending complete image epoch",ha="center",va="center",transform=ax.transAxes)
                ax.set_axis_off()
            continue
        r=complete[ds]["results"]
        vals=[metrics["nn4_objects"][ds], *[role_count(r,k) for k in
              ("sample_pair_positive_hinge","sample_triple_positive_hinge","proxy_pair_positive_hinge","proxy_triple_positive_hinge")],
              gradient(r,"sample","full_st","above_1e-8_any"),gradient(r,"proxy","full_st","above_1e-8_any")]
        locations=np.arange(len(labels))
        good=[i for i,v in enumerate(vals) if v is not None]
        colors=["#777777","#0072B2","#56B4E9","#D55E00","#E69F00","#0072B2","#D55E00"]
        axs[0,col].bar(locations[good],[vals[i] for i in good],color=[colors[i] for i in good])
        axs[0,col].set(xticks=locations,xticklabels=labels,title=f"{ds.upper()}: different counting domains",ylabel="Distinct objects (first bar) / proxy nodes")
        axs[0,col].tick_params(axis="x",labelrotation=58,labelsize=9)
        for i in good:
            axs[0,col].text(i,vals[i]+max(v for v in vals if v is not None)*.025,str(vals[i]),ha="center",fontsize=9)
        axs[0,col].axhline(512,ls="--",c="#777777",lw=.8)
        axs[0,col].set_ylim(0,max(v for v in vals if v is not None)*1.15)
        axs[0,col].text(.02,.97,f"Ancestors: {r.get('replay_batches')} image batches\nGradients: first {r.get('gradient_batches')} batches",va="top",fontsize=9,transform=axs[0,col].transAxes)
        radial_vals=[]
        radial_labels=[]
        radial_colors=[]
        for item in ("sample","proxy"):
            for mode,results in (("clean cache",extra["results"][f"{ds}_train"]),("fresh images",r)):
                v=gradient(results,item,"full_st","radial_absolute_fraction_median")
                if v is not None and v>0:
                    radial_vals.append(v);radial_labels.append(f"{item}\n{mode}")
                    radial_colors.append(("#56B4E9" if mode=="clean cache" else "#0072B2") if item=="sample" else ("#E69F00" if mode=="clean cache" else "#D55E00"))
        if radial_vals:
            xx=np.arange(len(radial_vals))
            axs[1,col].bar(xx,radial_vals,color=radial_colors)
            axs[1,col].set(yscale="log",xticks=xx,xticklabels=radial_labels,ylabel="Median |radial gradient| / total norm",title="Full-ST raw tangent gradients")
            axs[1,col].set_ylim(min(radial_vals)*.15,min(1.,max(radial_vals)*5))
            for i,v in enumerate(radial_vals):
                axs[1,col].text(i,v*1.2,f"{v:.2e}",ha="center",fontsize=9)
        else:
            axs[1,col].text(.5,.5,"Absolute radial fractions were not recorded",ha="center",va="center",transform=axs[1,col].transAxes)
            axs[1,col].set_axis_off()
    fig.suptitle("Fixed best: source-style augmented image epoch, zero optimizer updates\nNN4 is reverse retrieval; ancestors and ST gradients use different definitions",fontsize=14)
    name="image_epoch_activation_domains"
    fig.savefig(output/"figures"/f"{name}.png",dpi=180,bbox_inches="tight")
    fig.savefig(output/"figures"/f"{name}.svg",bbox_inches="tight")
    plt.close(fig)
    size=png_size(output/"figures"/f"{name}.png")
    copied.append({"name":name,"relative_path":f"figures/{name}.png","width":size[0],"height":size[1],
                   "source_metrics_relative_path":"metrics/image_epoch_figure_sources.json","generated_from_complete_image_epochs":list(complete)})
    return f"<figure><a href='figures/{name}.png' target='_blank'><img src='figures/{name}.png' alt='真实图像epoch的检索、祖先与梯度计数，以及径向梯度占比'></a><figcaption>完整真实图像epoch：第一根柱数对象，后续柱数代理；梯度只取声明的前几批。径向指标为每批每代理绝对分量占比的中位数，不是有符号径向均值除范数。<a href='metrics/image_epoch_figure_sources.json'>源统计</a></figcaption></figure>"


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Fresh output required: {output}")
    direction_path = summary_path(args.direction)
    activation_path = summary_path(args.activation)
    geometry_path = summary_path(args.geometry, geometry=True)
    identities = {}
    direction = load(direction_path, identities)
    activation = load(activation_path, identities)
    geometry = load(geometry_path, identities)
    extra_path = (Path(args.activation_extra) if args.activation_extra else
                  activation_path.parent / "review_figures_v1" / "extended_summary.json")
    extra = load(extra_path, identities)
    image, pending_image = image_epoch_data(args.image_epoch, identities)
    dmodels = direction["models"]
    gmodels = {r["identity"]["name"]: r for r in geometry["models"]}
    for ds in DATASETS:
        if ds not in dmodels or f"{ds}_train" not in activation["results"]:
            raise ValueError(f"Missing dataset metrics: {ds}")
    for ds,item in image.items():
        checkpoint=item["payload"].get("source_checkpoint_sha256",item["payload"].get("checkpoint_sha256"))
        expected=gmodels[f"HIER_{ds}_train"]["identity"]["metadata"].get("checkpoint_hash")
        if checkpoint is not None and checkpoint!=expected:
            raise ValueError(f"Fresh-image checkpoint differs from geometry best: {ds}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "metrics").mkdir()
    (output / "metrics" / "direction_summary.json").write_text(json.dumps(direction,ensure_ascii=False,indent=2),encoding="utf-8")
    (output / "metrics" / "activation_summary.json").write_text(json.dumps(activation,ensure_ascii=False,indent=2),encoding="utf-8")
    (output / "metrics" / "activation_extended_summary.json").write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding="utf-8")
    geometry_aggregate=[{k:r[k] for k in ("radial","whole_direction","proxy_direction","retrieval")}
                        | {"name":r["identity"]["name"],"c":r["identity"]["c"],"D":r["identity"]["D"]} for r in geometry["models"]]
    (output / "metrics" / "geometry_aggregate.json").write_text(json.dumps(geometry_aggregate,ensure_ascii=False,indent=2),encoding="utf-8")
    for ds,item in image.items():
        image_aggregate={"dataset":ds,"protocol":item["payload"].get("protocol"),
                         "optimizer_updates":item["payload"].get("optimizer_updates"),
                         "complete":item["complete"],"results":item["results"]}
        (output/"metrics"/f"image_epoch_{ds}.json").write_text(json.dumps(image_aggregate,ensure_ascii=False,indent=2),encoding="utf-8")
    copied = []
    geometry_root = geometry_path.parent.parent if geometry_path.parent.name == "analysis" else geometry_path.parent
    geometry_figures = geometry_root / "review_final" / "figures"
    dirfig = direction_path.parent / "figures"
    actfig = Path(args.activation_figures) if args.activation_figures else activation_path.parent / "review_figures_v2"

    overview_headers = ["数据集", "whole数量", "细类数量", "whole深度中位数", "whole深度std", "proxy深度中位数", "cap代理", "whole方向R", "proxy方向R", "NN4覆盖", "热点4槽位%"]
    overview = []
    for ds in DATASETS:
        r, g = dmodels[ds], gmodels[f"HIER_{ds}_train"]
        overview.append([ds.upper(), r["n_whole"], g["whole_direction"]["class_count"],
                         r["whole_depth"]["p50"], r["whole_depth"]["std_population"], r["proxy_depth"]["p50"],
                         r["proxy_groups"]["forward_cap"]["n"], r["whole_direction"]["R"], r["proxy_direction"]["R"],
                         r["retrieval_official"]["unique_top4_objects"], r["retrieval_official"]["hottest4_slot_fraction"]*100])
    write_csv(output / "overview.csv", overview_headers, overview)

    control_headers = ["数据集", "官方NN4覆盖", "纯方向覆盖", "中心残差覆盖", "整体旋转覆盖min", "整体旋转覆盖max", "原NN4细类purity", "中心残差purity"]
    controls = []
    angular_headers = ["数据集", "proxy两两cos中位数", "proxy首主方向能量%", "proxy熵有效秩", "最近细类中心夹角中位数°", "随机方向最近中心夹角中位数范围°", "进入细类90%方向锥代理"]
    angular = []
    for ds in DATASETS:
        r, g = dmodels[ds], gmodels[f"HIER_{ds}_train"]
        c, affinity = r["controls"], g["proxy_class_affinity"]
        random_medians = [q["angle_deg"]["p50"] for q in affinity["random_direction_reference"]]
        controls.append([ds.upper(), r["retrieval_official"]["unique_top4_objects"], r["retrieval_direction"]["unique_top4_objects"],
                         c["proxy_center_residual_same_radius"]["unique_top4_objects"], c["rotation_coverage"]["p00"], c["rotation_coverage"]["p100"],
                         r["retrieval_official"]["mean_fine_class_top4_purity"], c["proxy_center_residual_same_radius"]["mean_fine_class_top4_purity"]])
        angular.append([ds.upper(), r["proxy_direction"]["pair_cosine_distribution"]["p50"],
                        100*r["proxy_direction"]["uncentered_spectrum"]["top1_share"],
                        r["proxy_direction"]["uncentered_spectrum"]["entropy_effective_rank"], affinity["nearest_center_angle_deg"]["p50"],
                        f"{min(random_medians):.2f}–{max(random_medians):.2f}", affinity["proxy_count_in_nearest_class_direction_cone90"]])
    write_csv(output / "direction_controls.csv", control_headers, controls)
    write_csv(output / "direction_alignment.csv", angular_headers, angular)

    activation_headers = ["数据集", "cache采样批数", "sample pair选中", "sample triple选中", "sample pair正hinge", "sample triple正hinge", "proxy pair选中", "proxy triple选中", "proxy pair正hinge", "proxy triple正hinge", "sample梯度>1e−8", "proxy梯度>1e−8", "梯度批数"]
    activation_rows, gradient_rows, overlap_rows = [], [], []
    for ds in DATASETS:
        r = activation["results"][f"{ds}_train"]
        e = extra["results"][f"{ds}_train"]
        activation_rows.append([ds.upper(), r.get("replay_batches"), *[role_count(r,k) for k in
                                ("sample_pair_selected", "sample_triple_selected", "sample_pair_positive_hinge", "sample_triple_positive_hinge",
                                 "proxy_pair_selected", "proxy_triple_selected", "proxy_pair_positive_hinge", "proxy_triple_positive_hinge")],
                                gradient(r,"sample","full_st","above_1e-8_any"), gradient(r,"proxy","full_st","above_1e-8_any"), r.get("gradient_batches")])
        for item in ("sample", "proxy"):
            for mode in ("full_st", "direct"):
                gg = e["gradients"][f"{item}_{mode}"]
                gradient_rows.append([ds.upper(), item, mode, gg["batches"], gg["above_1e-8_any"],
                                      gg["mean_norm_per_proxy"]["median"], gg.get("radial_absolute_fraction_median")])
            role = e["ancestor_role_overlap"][item]
            overlap_rows.append([ds.upper(), item, role["positive_pair_and_triple_proxies"], role["positive_role_jaccard"],
                                 role["positive_pair_draw_weighted_depth"], role["positive_triple_draw_weighted_depth"]])
    write_csv(output / "activation_counts.csv", activation_headers, activation_rows)
    gradient_headers = ["数据集", "损失来源", "梯度口径", "批数", ">1e−8代理", "代理梯度范数中位数", "径向绝对分量占比中位数"]
    overlap_headers = ["数据集", "来源", "pair/triple共有正hinge代理", "角色集合Jaccard", "pair抽取加权深度", "triple抽取加权深度"]
    write_csv(output / "gradient_sources.csv", gradient_headers, gradient_rows)
    write_csv(output / "role_overlap.csv", overlap_headers, overlap_rows)

    content = "<header><p class='eyebrow'>原版 HIER · best 固定权重审查</p><h1>SOP 的热点来自哪些嵌入差异？</h1><p>完整高维几何、冻结关系抽取和梯度来源分别核验。点击表头排序，点击图查看原图。</p></header>"
    content += "<nav><a href='#geometry'>整体几何</a><a href='#directions'>方向机制</a><a href='#activation'>代理激活</a><a href='#images'>真实图像 epoch</a><a href='#definitions'>定义与边界</a></nav>"
    content += "<section id='geometry'><h2>三个 best 的整体几何</h2><p>原版三个数据集的 whole 几乎都处于 d₀≈4.6 的 cap 薄壳；10⁻⁷量级径向标准差主要反映浮点精度。SOP 的代理则几乎也挤在这个上限：497/512 参数超过切空间 cap，CUB/Cars 分别为9/1。不能从这些薄壳内的微小半径排序推断真实层级。</p>"
    content += table(overview_headers, overview) + "<p class='small'><a href='overview.csv'>下载整体统计 CSV</a>。所有原版空间为 c=.1、D=512、P=512；V7为c=1、D=256，须区分距离单位、归一球半径和深度。</p>"
    for name, caption in (("01_depth_distributions", "原始 whole/proxy 深度分布及 V7 对照；每组为独立数据池"),
                          ("02_dimensionless_radial_ranges", "无量纲球半径与双曲深度范围，避免把不同曲率球直接比较")):
        content += copy_figure(geometry_figures/f"{name}.png", output, f"geometry_{name}", caption, identities, copied,"metrics/geometry_aggregate.json")
    content += "</section><section id='directions'><h2>SOP 的显著差异在代理方向</h2><p>whole 的共同方向集中度三组都约 .58；代理的 R 在CUB/Cars约 .04–.05，在SOP达到 .648。SOP 所有非自身代理对的余弦都为正，首个主方向占约42%方向能量。高覆盖的 CUB/Cars 并不意味着每个代理都进入细类簇：其最近类别角度与同维度随机方向参考相近。</p>"
    content += table(angular_headers, angular) + "<p>角度直接在原始512维中计算，未做二维投影。最近类别是夹角最小的类别中心，不是赋予代理类别标签。SOP有11318个细类，而CUB/Cars约100类，因此应先比较各数据集自身的随机方向参考。</p>"
    captions = {"01_proxy_radial_angular_groups": "代理深度与高维方向：紫色为 forward cap 参数；SOP 少量浅代理同样偏向共同方向",
                "02_direction_spectrum_common_pole": "方向谱与共同方向集中度：SOP差异主要出现在代理，而非whole共同方向R",
                "03_hotspot_direction_location": "NN4热点的位置：SOP热点朝向proxy共同方向，同时相对自身细类留一中心更偏外围",
                "04_direction_controls_semantic_hotspots": "保持代理半径的冻结方向控制及SOP图像目录大类别分布；变化不是重训收益"}
    for name, caption in captions.items():
        content += copy_figure(dirfig/f"{name}.png", output, f"direction_{name}", caption, identities, copied,"metrics/direction_summary.json")
    content += table(control_headers, controls)
    content += "<p>整体正交旋转保留每对代理夹角和每个代理半径；中心残差则减去代理单位方向均值，再归一化并恢复原半径，它改变内部角度。SOP前者覆盖880–1065，后者1778，支持当前整体朝向和内部共同分量参与热点。中心残差的细类purity反而下降，覆盖改善不是类内结构改善或训练收益。</p>"
    sop_sem = dmodels["sop"]["sop_semantic_categories"]
    content += table(["图像目录大类别", "样本池%", "NN4槽位%", "相对富集倍数"],
                     [[r["category_from_image_path"],100*r["pool_fraction"],100*r["raw_top4_slot_fraction"],r["slot_pool_enrichment"]] for r in sop_sem])
    content += "<p>目录大类别来自每个样本真实 image_path；SOP的训练细类表示不同商品实例，大类别不能替代细类标签。</p>"
    v7 = gmodels.get("V7_e300")
    if v7:
        rr=v7["retrieval"]
        content += f"<aside><strong>纠正此前等深度数字的归属：</strong>1210–1295是V7 e300先统一whole深度、再随机替换proxy方向的结果，不能拿来与原版CUB/Cars的已训练1647/1664直接比较。V7仅统一whole深度的覆盖是{rr['raw']['different_objects_top4']}→{rr['direction']['different_objects_top4']}；原版whole本已近乎同深，SOP仅改为方向检索仍覆盖{dmodels['sop']['retrieval_direction']['unique_top4_objects']}。SOP的热点因此有明确的方向来源。</aside>"
    content += "</section><section id='activation'><h2>热点检索与训练项激活是两种统计</h2><p>这里先审查固定 best 的已有全缓存，以源式180图像batch、每类2图像和hard Gumbel抽取关系。clean缓存的采样窗长度为完整source epoch，但没有重新读取或增强图像，也没有更新网络。增强缓存是另外16批窗口；梯度仅取2批，不能将所有行称为真实再训练一轮。</p>"
    content += table(activation_headers, activation_rows)
    content += "<ol><li>NN4覆盖：每个proxy在完整样本池反向检索4个最近样本，衡量热点与覆盖。</li><li>hard selected：被某个sample/proxy三元组抽为pair或triple祖先，包含两祖先相同的碰撞。</li><li>positive hinge：碰撞置零后，三个hinge总和大于零的抽取；这仍不等于非零参数梯度。</li><li>full-ST梯度：官方hard Gumbel straight-through完整路径；未硬选中的候选也可能经概率路径得到梯度。</li><li>direct梯度：保持同次hard抽取、detach Gumbel概率，排除概率路径；proxy项还包含自身作为i/j/k查询端点的路径。</li></ol>"
    for name, caption in (("01_role_sources_depth", "hard祖先/正hinge来源按代理深度分开计数；不是NN4反向检索"),
                          ("02_gradient_source_depth", "sample/proxy损失来源及full-ST/direct参数梯度；梯度只审查2批"),
                          ("03_pair_triple_role_overlap", "pair与triple角色重叠：存在不同角色选择，但不是固定两组层次节点"),
                          ("04_gradient_radial_fraction", "梯度的径向占比：forward cap代理主要沿角方向得到梯度，并非全部休眠")):
        content += copy_figure(actfig/f"{name}.png",output,f"activation_{name}",caption,identities,copied,"metrics/activation_extended_summary.json")
    content += table(overlap_headers, overlap_rows) + "<p>pair与triple祖先在每个三元组中是明确的两个角色，且不同选择才产生有效项；累计代理集合重叠说明同一个节点能在不同关系中承担两种角色。它不能反推“每个三元组没有层次差”，也不支持把512代理切成固定pair层和triple层。</p>"
    content += table(gradient_headers, gradient_rows)
    content += "<p>当raw切空间参数范数超过2.3，forward相当于先固定其范数再映射，改变raw范数而保持方向已不能改变球内位置。因此SOP约497个cap代理的径向导数近零；sample/proxy full-ST径向绝对占比中位数约2.74×10⁻⁶/1.85×10⁻⁶，但两项仍各有512个代理梯度超过10⁻⁸。这里说的是<strong>参数范数的forward截断</strong>，没有执行optimizer step或全局梯度范数clip。官方Riemannian backward hook被保留。</p>"
    content += "</section><section id='images'><h2>实际读取图像的无更新 epoch</h2>"
    image_headers=["数据集","状态","完整批数","实际批数","采样槽数","不同图像数","sample pair正hinge","sample triple正hinge","proxy pair正hinge","proxy triple正hinge","whole深度中位数"]
    image_rows=[]
    for ds in DATASETS:
        item=image.get(ds)
        if not item:
            image_rows.append([ds.upper(),"待运行",*[None]*9])
        else:
            r=item["results"]
            depth=r.get("whole_depth_sampled_slots",{})
            image_rows.append([ds.upper(),"完整无更新epoch" if item["complete"] else "未通过完整窗验收",
                               r.get("full_source_epoch_batches"),r.get("replay_batches"),r.get("sampled_image_slots"),r.get("distinct_sampled_objects"),
                               *[role_count(r,k) for k in ("sample_pair_positive_hinge","sample_triple_positive_hinge","proxy_pair_positive_hinge","proxy_triple_positive_hinge")],
                               depth.get("median",depth.get("p50"))])
    content += table(image_headers,image_rows)
    write_csv(output/"image_epoch_status.csv",image_headers,image_rows)
    if not image:
        content += "<p class='pending'>本报告尚未取得完整的真实图像epoch结果；上述缓存回放结果不替代这一环节，不宣称再训练已经完成。</p>"
    else:
        completed=[ds.upper() for ds,item in image.items() if item["complete"]]
        content += f"<p>当前已验收完整图像窗：{html.escape('、'.join(completed) or '无')}。它们使用固定best网络、源式随机增强与类别采样，optimizer更新次数为0。网络前向能提供当前best在训练输入条件下的使用信息；由于原PA分类代理未保存在checkpoint，这不是严格原协议续训，也不是历史训练记录。</p>"
        content += "<p>图像epoch源统计：" + " · ".join(f"<a href='metrics/image_epoch_{ds}.json'>{ds.upper()}</a>" for ds in image) + "</p>"
        image_figure=plot_image_epoch(image,direction,extra,output,copied)
        if image_figure:
            content += image_figure
    content += "</section><section id='definitions'><h2>如何读这些量</h2><p>单位方向 u=x/‖x‖；夹角 θ=arccos(uᵀv)。方向集中度 R=‖mean(u)‖，范围0–1，它不是代理对某个类别的余弦相似度。非自身平均两两余弦为(nR²−1)/(n−1)。熵有效秩反映方向能量分布，不等于实际嵌入维度。</p><p>球坐标半径 r=‖x‖，归一球半径ρ=√c·r，原点双曲深度d₀=2·atanh(ρ)/√c。原版cap约束映射前切空间范数≤2.3，对应d₀≤4.6，而不是把r限制为2.3。</p><p>本次可定位冻结几何与当前激活来源；三组数据域、类别数量、采样重复程度及训练历史不同，不能单凭best横截面对训练因果或分类收益下结论。交接训练配置也存在混杂：SOP初始学习率6×10⁻⁴，CUB/Cars为10⁻⁴；weight decay分别10⁻⁴与.01；best分别在96和17/45轮，源式每轮批数330对32/44。这些配置差异与数据域同时改变，不能把SOP热点归因于商品数据域本身。角色梯度统计归因到sample或proxy损失；proxy损失内部祖先和查询端点的梯度未进一步独立分解。</p><p>数值来源与执行身份保存在各CSV和验收清单。全部图使用真实高维数值，不以二维投影代替距离或方向证据。</p>"
    content += "<p><a href='direction_controls.csv'>方向控制CSV</a> · <a href='direction_alignment.csv'>方向一致性CSV</a> · <a href='activation_counts.csv'>激活CSV</a> · <a href='gradient_sources.csv'>梯度CSV</a> · <a href='role_overlap.csv'>角色CSV</a> · <a href='image_epoch_status.csv'>图像epoch状态CSV</a> · <a href='acceptance.json'>身份与链接验收</a></p></section>"

    css="""body{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;color:#243044;background:#f4f6fa;margin:0}header,section,nav{max-width:1180px;margin:22px auto;background:white;padding:24px 30px;border-radius:12px;box-shadow:0 2px 12px #2030500a}h1{font-size:32px;margin:6px 0}h2{font-size:24px}p{max-width:1050px}.eyebrow,.small,figcaption{color:#596579;font-size:14px}nav{display:flex;gap:24px;flex-wrap:wrap;position:sticky;top:0;z-index:2}a{color:#12629c}figure{margin:28px 0}img{width:100%;height:auto;display:block}figcaption{margin-top:8px}.tablebox{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:10px 9px;border-bottom:1px solid #dce2ea;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eef3f8;cursor:pointer}tbody tr:hover{background:#eef6fe}aside,.pending{background:#fff4dc;border-left:4px solid #d19423;padding:16px;margin:18px 0}li{margin:8px 0}@media(max-width:600px){header,section,nav{padding:18px;margin:12px}h1{font-size:26px}}"""
    js="""document.querySelectorAll('table.sortable th').forEach(th=>th.addEventListener('click',()=>{const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),b=t.tBodies[0],rows=[...b.rows],dir=th.dataset.dir==='asc'?-1:1;rows.sort((a,z)=>{const x=a.cells[i].textContent,y=z.cells[i].textContent,nx=Number(x),ny=Number(y);return dir*(Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y,'zh-CN'))});rows.forEach(r=>b.appendChild(r));th.dataset.dir=dir===1?'asc':'desc'}));"""
    page=f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>原版HIER三数据集机制审查</title><style>{css}</style></head><body>{content}<script>{js}</script></body></html>"
    (output/"report.html").write_text(page,encoding="utf-8")
    relative_links=re.findall(r"(?:src|href)='([^']+)'",page)
    for link in relative_links:
        if not link.startswith(("#","http://","https://")) and link!="acceptance.json" and not (output/link).is_file():
            raise FileNotFoundError(f"Broken report link: {link}")
    for path,digest in identities.items():
        if sha256(path)!=digest:
            raise AssertionError(f"Input changed: {path}")
    acceptance={"script_sha256":sha256(__file__),"inputs_sha256":identities,"figures":copied,
                "all_relative_links_exist":True,"relative_link_count":len(relative_links),
                "all_png_dimensions_valid":True,"input_sha256_unchanged":True,
                "image_epoch_complete_datasets":[ds for ds,item in image.items() if item["complete"]],
                "image_epoch_pending_datasets":[ds for ds in DATASETS if ds not in image or not image[ds]["complete"]],
                "report_uses_only_aggregate_statistics":True,"training_updates_performed_by_builder":0}
    (output/"acceptance.json").write_text(json.dumps(acceptance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in acceptance.items() if k not in ("inputs_sha256","figures")},ensure_ascii=False))
    return acceptance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--direction",required=True)
    parser.add_argument("--activation",required=True)
    parser.add_argument("--geometry",required=True)
    parser.add_argument("--activation-extra")
    parser.add_argument("--activation-figures")
    parser.add_argument("--image-epoch",action="append",default=[])
    parser.add_argument("--output",required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
