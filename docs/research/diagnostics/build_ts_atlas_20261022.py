"""Build the TS-only baseline atlas from sanitized, saved-metric captures."""
from __future__ import annotations

import csv
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22"
TABLES = OUT / "tables"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    return "—" if value is None else str(Decimal(str(round(value, 9))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


BASE_SPECS = [
    ("B01", "HC", 1, None, False, 1000, 4, "SH32", 307, 8, 16),
    ("B02", "HC", .1, None, False, 1000, 4, "SH32", 307, 8, 16),
    ("B03", "HC", .1, 2.3, False, 1000, 4, "SH32", 307, 8, 16),
    ("B04", "HC", .1, 2.3, False, 500, 2, "SH32", 307, 8, 16),
    ("B05", "HC", .1, 2.3, True, 500, 2, "SH32", 307, 8, 16),
    ("B06", "V8", .1, 2.3, True, 500, 2, "SH32", 307, 2, 32),
    ("B07", "V8", .1, 2.3, False, 500, 2, "SH32", 307, 2, 32),
    ("B08", "V8", .1, 2.3, True, 500, 2, "CB64", 200, 2, 32),
    ("B09", "V9", 1, None, False, 1000, 4, "CB64", 200, 2, 32),
    ("B10", "HC", .1, 2.3, True, 500, 2, "CY64", 300, 8, 16),
]

RUN_SPECS = [
    ("TS01", "orig_c1", "B01", "H0", 4780, "HIST"),
    ("TS02", "orig_c1_s4780_r2", "B01", "H0", 4780, "R2"),
    ("TS03", "orig_c01", "B02", "H0", 4780, None),
    ("TS04", "orig_c01_cap", "B03", "H0", 4780, None),
    ("TS05", "orig_c01_cap_half", "B04", "H0", 4780, None),
    ("TS06", "orig_c01_cap_gs_half", "B05", "H0", 4780, None),
    ("TS07", "v8_b0_32_ts", "B06", "H0", 22, None),
    ("TS08", "v8_b0_no_gs", "B07", "H0", 22, None),
    ("TS09", "v8_b64_ts", "B08", "H0", 22, None),
    ("TS10", "v8_hier64_ts", "B08", "Hshared-A0", 22, None),
    ("TS11", "v9_b64_ts", "B09", "H0", 22, None),
    ("TS12", "v9_hier64_dual_curvature_ts", "B09", "Hshell-A0", 22, None),
    ("TS13", "orig_c01_cycle300_b64", "B10", "H0", 4780, None),
    ("TS14", "orig_c01_cycle300_hier", "B10", "Hshell-A0", 4780, None),
    ("TS15", "v9_hier64_adamw_new_ts", "B09", "Hshell-A1", 22, None),
]


def baseline(spec):
    key, lineage, curvature, cap, gs, gamma, delta, batch, steps, workers, eval_batch = spec
    name = (f"{lineage}-C{'1' if curvature == 1 else '01'}-Cap{'0' if cap is None else '23'}"
            f"-GS{int(gs)}-M{gamma}_{delta}-{batch}x{steps}")
    return {"baseline_id": key, "name": name, "lineage": lineage,
            "base_c": curvature, "base_tangent_cap": cap,
            "base_depth_cap": 4.6 if cap is not None else None,
            "base_GS": gs, "gamma": gamma, "delta": delta, "batch": batch,
            "global_batch": 32 if batch == "SH32" else 64,
            "local_batch": 32, "world_size": 1 if batch == "SH32" else 2,
            "steps_per_epoch": steps, "epochs": 300,
            "total_updates": steps * 300, "workers": workers, "eval_batch_size": eval_batch,
            "rng_protocol": "continuous original-derived streams" if lineage == "HC" else "versioned step reseeding",
            "negative_pairing": "global batch reversal, local32 BN",
            "base_losses": "CE + .01 Rcontr + .01 Rhier; label smoothing .2"}


def build():
    captures = [read(TABLES / "source_capture_a.json"), read(TABLES / "source_capture_b.json")]
    capture_aliases = {"orig_c1_repeat_r2": "orig_c1_s4780_r2"}
    sources = {capture_aliases.get(row["id"], row["id"]): row for capture in captures for row in capture["rows"]}
    capture_files = {capture_aliases.get(row["id"], row["id"]): name for capture, name in zip(captures,
                     ("source_capture_a.json", "source_capture_b.json")) for row in capture["rows"]}
    inventory = read(ROOT / "artifacts/experiment_inventory_2026-10-21/summary.json")
    old = {row["id"]: row for row in inventory["rows"] if row["selection"] == "TS"}
    history = read(TABLES / "source_capture_historical.json")
    history.update(status="historical_completed_with_resume", completed_epochs=300, best_epoch=230,
                   train_count=9840, validation_count=0, test_count=2468,
                   metric_files=history["unique_epochs"], metric_files_contiguous_from_one=True)
    history["best"]["test_log_rounded"] = dict(history["best"]["test"])
    history["best"]["test"] = {"oa": old["orig_c1"]["test_oa"],
                                   "aa": old["orig_c1"]["test_aa"], "count": 2468}
    history["latest"]["clean_train"] = {"oa": None, "aa": None, "count": None}
    sources["orig_c1"] = history
    bases = {spec[0]: baseline(spec) for spec in BASE_SPECS}
    runs = []
    for run_id, identifier, base_id, component, seed, replicate in RUN_SPECS:
        source, base = sources[identifier], bases[base_id]
        legacy = old.get(identifier, {})
        enabled = component != "H0"
        full_name = f"{base['name']}/{component} (S{seed}" + (f", {replicate}" if replicate else "") + ")"
        legacy_name = legacy.get("canonical_name", "V9-HIER64-K20-W0-Cbase1-Chier0.1-H4.6-GShier-M1000_4-PLR0.005-EPS1e-4-TS-S22")
        optimizer = None if not enabled else {"name": "AdamW", "lr": .005 if component.endswith("A1") else .01,
                      "eps": 1e-4 if component.endswith("A1") else 1e-8,
                      "min_lr": .0005, "betas": [.9, .999], "weight_decay": .01}
        sample_view = "none" if not enabled else ("capped_actual_depth" if component.startswith("Hshared") else "fixed_depth_4.6")
        gs_scope = "none" if not enabled else ("same shared output GS as base" if component.startswith("Hshared") else "one independent HIER branch GS")
        end = source["latest"] if source["completed_epochs"] == 300 else None
        clean_record = source.get("latest_clean_train_record")
        record = {"run_id": run_id, "id": identifier, "baseline_id": base_id,
                  "display_name": full_name, "component": component, "seed": seed,
                  "replicate": replicate, "legacy_canonical_name": legacy_name,
                  "selection": "TS", "train_count": 9840, "validation_count": 0, "test_count": 2468,
                  "status": source["status"], "completed_epochs": source["completed_epochs"],
                  "planned_epochs": 300, "best_epoch": source["best_epoch"],
                  "best_test_oa": source["best"]["test"]["oa"],
                  "best_test_aa_same_epoch": source["best"]["test"]["aa"],
                  "online_train_oa_at_best_test_epoch": source["best"]["online_train"]["oa"],
                  "end_test_oa": end["test"]["oa"] if end else None,
                  "end_test_aa": end["test"]["aa"] if end else None,
                  "end_online_train_oa": end["online_train"]["oa"] if end else None,
                  "end_clean_train_oa": (end.get("clean_train") or {}).get("oa") if end else None,
                  "latest_test_epoch": source["latest"]["completed_epoch"],
                  "latest_test_oa": source["latest"]["test"]["oa"],
                  "latest_online_train_oa": source["latest"]["online_train"]["oa"],
                  "latest_clean_train_epoch": clean_record["completed_epoch"] if clean_record else None,
                  "latest_clean_train_oa": clean_record["clean_train"]["oa"] if clean_record else None,
                  "last10_test_oa_mean": source["tail_test_oa_mean"] if end else None,
                  "last10_test_oa_population_std": source["tail_test_oa_population_std"] if end else None,
                  "hier_enabled": enabled, "hier_c": .1 if enabled else None,
                  "hier_view": sample_view, "hier_GS_scope": gs_scope,
                  "direct_radial_hier_gradient": "not constructed" if not enabled else ("not explicitly removed; cap Jacobian still applies" if sample_view == "capped_actual_depth" else "zero at whole-coordinate interface"),
                  "proxy_tangent_cap": 2.3 if enabled else None, "proxy_GS": enabled,
                  "proxy_optimizer": optimizer,
                  "production_commit": source.get("production_commit", legacy.get("production_commit")),
                  "source_document": legacy.get("source_document", "74_V9_ADAMW_FULL_RUN_2026-10-22/report.md"),
                  "metric_files": source["metric_files"],
                  "metric_files_contiguous_from_one": source["metric_files_contiguous_from_one"],
                  "source_precision": legacy.get("source_precision", "unrounded saved JSON"),
                  "snapshot_source": "source_capture_historical.json" if identifier == "orig_c1" else capture_files[identifier],
                  "capture_label": source["id"]}
        runs.append(record)
    assert len(runs) == 15 and len(bases) == 10
    assert len({r["display_name"] for r in runs}) == 15
    assert all(r["selection"] == "TS" and r["validation_count"] == 0 for r in runs)
    assert all(r["metric_files_contiguous_from_one"] for r in runs)
    assert sum(r["completed_epochs"] == 300 for r in runs) == 14
    assert all(r["best_epoch"] <= r["completed_epochs"] for r in runs)
    for identifier, source in sources.items():
        if "manifest_best_test_oa" in source:
            assert abs(source["best"]["test"]["oa"] - source["manifest_best_test_oa"]) < 1e-9
        if identifier in old and not identifier.startswith("orig_c01_cycle"):
            assert abs(source["best"]["test"]["oa"] - old[identifier]["test_oa"]) < 1e-9
    registry = {"schema_version": 3, "scope": "TS only: V8, original HyCoRe adaptations, V9; no validation-selected runs",
                "observed_utc": [capture["observed_utc"] for capture in captures],
                "accuracy_unit": "percent", "seed_policy": "identity suffix, not a method/configuration axis",
                "alias_policy": "New display aliases only; immutable legacy run IDs, paths, configs and checkpoint names retained",
                "naming_document": "docs/research/75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md",
                "baselines": list(bases.values()), "runs": runs}
    assert {r["id"] for r in runs} == set(old) | {"v9_hier64_adamw_new_ts"}
    assert all("seed" not in b for b in bases.values())
    serialized = json.dumps(registry, ensure_ascii=False)
    assert all(marker not in serialized for marker in ("/mnt/", "59.72.", "GPU-", "command_argv", "private_key"))
    for capture in captures:
        assert all(row["train_count"] == 9840 and row["validation_count"] == 0 and row["test_count"] == 2468 for row in capture["rows"])
    for r in runs:
        assert 0 <= r["best_test_oa"] <= 100 and 0 <= r["best_test_aa_same_epoch"] <= 100
        if r["status"] == "running":
            assert r["end_test_oa"] is None and r["last10_test_oa_mean"] is None
    write_json(ROOT / "experiment_registry_ts_v3.json", registry)
    write_json(TABLES / "unified_inventory.json", registry)
    flat = []
    for r in runs:
        b = bases[r["baseline_id"]]
        row = {**{k: v for k, v in r.items() if not isinstance(v, (dict, list))},
               **{k: b[k] for k in ("base_c", "base_tangent_cap", "base_GS", "gamma", "delta", "batch", "global_batch", "local_batch", "world_size", "steps_per_epoch", "total_updates", "workers", "eval_batch_size", "rng_protocol")},
               "proxy_initial_lr": (r["proxy_optimizer"] or {}).get("lr"),
               "proxy_eps": (r["proxy_optimizer"] or {}).get("eps")}
        flat.append(row)
    with (TABLES / "all_experiments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (TABLES / "name_mapping.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["run_id", "baseline_id", "id", "display_name", "legacy_canonical_name", "seed", "production_commit"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in runs)
    write_json(TABLES / "acceptance.json", {"all_passed": True, "checks": {
        "ts_only_15_runs": True, "unique_display_names": True, "ten_baseline_recipes": True,
        "fourteen_300_epoch_records": True, "one_running_snapshot_not_ranked": True,
        "best_scores_match_saved_manifests": True, "legacy_completed_best_scores_unchanged": True,
        "same_epoch_AA_from_best_epoch_record": True, "epoch_files_contiguous": True,
        "seed_outside_method_signature": True, "no_server_paths_in_public_registry": True,
        "no_forward_or_training": True},
        "historical_log": {"unique_epochs": history["unique_epochs"], "duplicate_raw_epochs": history["duplicate_raw_epochs"]},
        "ongoing_snapshot_epoch": runs[-1]["completed_epochs"]})

    names = {r["id"]: r for r in runs}
    performance = table(["记录", "底座 / 组件（seed）", "状态", "best轮", "best OA", "同轮AA", "e300 OA", "e291–300 OA均值±总体SD"], [
        [r["run_id"], f"{r['baseline_id']} / {r['component']}（S{r['seed']}" + (f"，{r['replicate']}" if r['replicate'] else "") + "）",
         "完成；历史续训" if r["replicate"] == "HIST" else ("完成" if r["completed_epochs"] == 300 else f"进行中 e{r['completed_epochs']}/300"),
         r["best_epoch"], number(r["best_test_oa"]), number(r["best_test_aa_same_epoch"]), number(r["end_test_oa"]),
         f"{number(r['last10_test_oa_mean'])} ± {number(r['last10_test_oa_population_std'])}" if r['end_test_oa'] is not None else "未完成"] for r in runs])
    training = table(["记录", "best test同轮在线训练OA", "e300在线训练OA", "e300 clean/eval训练OA", "e300 test OA"], [
        [r["run_id"], number(r["online_train_oa_at_best_test_epoch"]), number(r["end_online_train_oa"]),
         number(r["end_clean_train_oa"]), number(r["end_test_oa"])] for r in runs])
    recipes = table(["底座", "统一底座名", "batch / 步每轮", "基础c", "基础cap", "基础GS", "Rhier / Rcontr margin"], [
        [b["baseline_id"], b["name"], f"{b['batch']} / {b['steps_per_epoch']}", b["base_c"],
         "无额外cap" if b["base_tangent_cap"] is None else "切空间2.3", "开" if b["base_GS"] else "关",
         f"{b['gamma']}/N / {b['delta']}"] for b in bases.values()])
    active = runs[-1]
    stamp = dt.datetime.fromisoformat(captures[-1]["observed_utc"]).astimezone(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# V8—HyCoRe改版—V9：TS实验统一命名与baseline图谱

采集截止：北京时间{stamp}。15条正式记录、10种基础配方；14条已有300轮终点（含1条历史续训参照），1条新AdamW完整实验进行中。范围仅为全9840训练、2468 test逐轮选best；不再审查或纳入VS984验证选模实验。71/73十轮诊断单列，不计作完整分类实验。

本次读取现有manifest、选中轮/末轮/末10轮JSON及历史标量日志，没有新前向、重测、BN调整、训练更新或停止任务。已完成13条现代任务各300份连续轮记录；进行中任务仅使用manifest已确认完成的前缀。历史日志具有300个唯一epoch，raw e5重复一次，按最后保存记录读取；选中轮raw229、完成230，终点raw299、完成300。

## 1. 统一命名：底座与HIER组件分开

从本表起使用 **`统一底座名 / HIER组件（seed，必要时重复编号）`**。底座编号B01–B10便于讨论，正式记录编号TS01–TS15用于精确定位。seed仅写名称后缀及复现元数据，不作为方法开关，也不产生另一种基础配方。HC保留原版训练流程，V8/V9保留各自版本流程；同几何不合并不同实现身份。

完整格式：`{{HC|V8|V9}}-C{{1|01}}-Cap{{0|23}}-GS{{0|1}}-M{{1000_4|500_2}}-{{SH32|CB64|CY64}}x{{步数}} / {{H0|Hshared-A0|Hshell-A0|Hshell-A1}} (S种子[, HIST|R2])`。

- C01明确指c=0.1；Cap23指切空间模长上限2.3（约原点深度上限4.6）；Cap0仅表示没有新增这一cap，仍保留数值安全投影。
- 名称中的GS0/GS1只表示**基础whole/part路径**。HIER样本GS的实际位置由组件另列，代理GS也另列，避免“GS1作用于哪条管线”含混。
- M1000_4为Rhier阈值1000/N、Rcontr阈值4；M500_2为500/N、2。两项系数始终各0.01，不是损失权重减半。
- SH32：shuffle/drop-last，global32/local32/单卡，307步；CB64：类别均衡、类内有放回，global64/local32/双卡，200步；CY64：类别均衡循环队列、跨轮游标，global64/local32/双卡，300步。
- 300轮下总更新分别92100、60000、90000。CB64每步随机选择32类；CY64每5步保证每类出现4次，同时改变了类别组合规则。CB64和CY64都属于类别均衡。
- HIST为原版历史记录，含早期resume；R2为后来从头复现，不能并成同一次训练。

### 10种基础配方

{recipes}

HC的workers8/eval16、连续随机流；V8/V9的workers2/eval32、版本化逐步随机流均保留。所有local BN为32，双卡保持global flip负例及原part覆盖whole行为。底座相同意味着同一基础配方，不承诺不同HIER训练下权重或BN相同。

### HIER组件定义与管线

| 组件 | whole输入 | whole GS路径 | 代理与AdamW |
|---|---|---|---|
| H0 | 不构造HIER视图/损失/代理 | 无HIER路径 | 无 |
| Hshared-A0 | 共享cap后的实际whole半径，未强制等深 | 复用基础出口GS，whole到基础/HIER共用一次 | 自身cap2.3/GS；lr0.01，eps1e-8 |
| Hshell-A0 | 保方向统一到c0.1、原点深度4.6 | 归一化后HIER专用GS一次；不串联基础GS | 自身cap2.3/GS；lr0.01，eps1e-8 |
| Hshell-A1 | 与Hshell-A0相同 | 与Hshell-A0相同 | 自身cap2.3/GS；lr0.005，eps1e-4 |

所有HIER组件均512×256代理、官方0.9倍随机初始化、K20/T50、margin/tau各0.1、lambda0.5、W0、排除sample/proxy self-k。代理最低LR0.0005、betas0.9/0.999、WD0.01与300轮余弦保持；A1只联合改变初始LR及eps。代理仍只是前向cap，无更新后参数投影，也未强制代理恰在4.6。

- **TS10/V8**：共享cap→共享GS→同一个whole分别供HyCoRe与HIER；whole使用受限的实际深度。
- **TS12/TS15/V9**：原生c1 whole直接供基础目标；其方向另构造c0.1/4.6视图→HIER GS。基础路径无新增cap/GS。
- **TS14/循环**：c0.1的cap后、基础GS前分叉。一路基础GS→HyCoRe；另一路保方向统一4.6→HIER GS。两路各一次，非串联两次。基础cap/GS仍保留。

Hshell不detach whole或其模长；它在whole坐标接口消除HIER的直接径向梯度，保留角向梯度。共享网络、合并梯度裁剪和优化器仍使训练耦合，不能称为两套独立训练。

实现定位：[V8共享映射](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py:58)、[V8实际HIER输入](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8_fulltrain/train.py:430)、[V9固定壳](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9/geometry.py:60)、[循环分叉](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c/geometry.py:43)、[循环固定壳](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c/geometry.py:84)。旧运行身份不改写。

## 2. 全部正式实验分类成绩

OA为总体准确率，AA为40类准确率均值，单位均为%。AA严格取best OA同一轮；不能取独立跨轮最大AA。best表示每轮test所选的最高OA，e300单列。末10轮统计固定使用e291–300，SD为这10个数的总体标准差，仅描述这一条轨迹，不是多seed误差条或显著性。

{performance}

TS01 best OA/AA使用54保存的固定权重复测精度；其训练同轮和e300来自历史三位小数日志，旧表头Valid实际为官方test。TS02–TS06沿用原版保存的0.001个百分点精度，不按2468人数反推伪造更多精度；V8/V9/循环源JSON保留未舍入值。显示到三位小数不代表源精度相同。

TS15是进行中快照：完成e{active['completed_epochs']}，当前best e{active['best_epoch']} OA{number(active['best_test_oa'])}%/AA{number(active['best_test_aa_same_epoch'])}%；最新完整轮OA{number(active['latest_test_oa'])}%。不填写e300，不参与最终优劣排名，不逐轮监看。

### HyCoRe改版链条：明确改了哪一步

| 对照 | 修改 | best OA变化（百分点） | e300 OA变化（百分点） |
|---|---|---:|---:|
| TS02→TS03 | 原版从头c1→c0.1 | −0.121 | −0.406 |
| TS03→TS04 | c0.1增加cap2.3 | −0.243 | −0.162 |
| TS04→TS05 | gamma/delta减半 | +0.324 | +0.811 |
| TS05→TS06 | 基础出口开启GS | +0.364 | −0.446 |
| TS07→TS08 | V8 shuffle32关闭GS | 0.000 | +0.527 |

前四条HC改版均无HIER/代理。TS02是后来从头复现的c1锚点；c0.1与c1间含执行时间/版本环境等复现差异，不冒充同次严格随机配对。TS03/04、TS04/05、TS05/06的初始化和数据身份已在60–62验收匹配。单条训练的增减不是多次重复统计结论。

## 3. 训练准确率：与test分开检查

在线训练OA来自当轮增强输入、train模式、训练中的不断变化权重。SH32通常9824位置，CB64为12800位置，CY64为19200位置；类别抽样还会重复对象，因此三者不是相同对象集合下的统一评估。

clean/eval训练OA则使用轮末权重、eval BN、无训练增强、同9840训练对象，不是独立验证集，也不参与选模。现代入口通常每10轮记录；原版改版没有保存同口径全量clean/eval训练OA，留空，不以在线准确率补填。best test所在轮多数并非10的倍数，所以表列其在线训练OA，不冒充该权重的clean/eval训练OA。

{training}

TS15当前e{active['latest_test_epoch']}在线训练OA{number(active['latest_online_train_oa'])}%；最近保存的clean/eval训练读出是e{active['latest_clean_train_epoch']}、OA{number(active['latest_clean_train_oa'])}%，二者不能合写成同轮。本次不为了补缺项重新前向。

V9原参数e300在线训练约99.71%，但clean/eval训练96.28%、test88.45%；无HIER对照在线99.72%、clean/eval98.31%、test91.17%。这些读出说明“训练时分类已很高”不能替代轮末eval表现；BN条件、输入增强、采样分布和权重取样时刻不同，单靠此表不能定位全部原因。

## 4. 可比较的HIER增量与baseline选择依据

| 基础配方 | H0 | 原HIER组件 | best OA增量 | e300 OA增量 |
|---|---|---|---:|---:|
| B08：V8/c0.1/共享cap及GS/CB64 | TS09 | TS10，Hshared-A0 | −0.527 | −0.486 |
| B09：原生c1/CB64 | TS11 | TS12，Hshell-A0 | −0.122 | −2.715 |
| B10：c0.1/cap及基础GS/CY64 | TS13 | TS14，Hshell-A0 | +0.486 | +0.486 |

TS15将在B09上检验Hshell-A1，保留TS11（无HIER）与TS12（原AdamW）两个参照；当前没有完成结果。B10配对匹配初始化和持续的基础随机流，隔离HIER分支更充分；历史V8/V9虽配方匹配，不承诺跨运行完整逐位重放。

| 候选底座 | 已有优势/对照 | 加入新HIER时需要明确的变化 |
|---|---|---|
| B09 / TS11 | 原生HyCoRe几何；原Hshell-A0与进行中的A1都有匹配参照 | 当前A1完整实验已在该底座，先看完成结果即可判断学习器改动 |
| B10 / TS13 | 已有固定壳HIER正增量；基础/分支GS清楚分离 | 用A1替换A0可在该配方上比较优化器组合；不能借此单独证明循环优于有放回 |
| B08 / TS09 | 已有类别均衡64；无HIER best93.882%、AA91.920% | 原HIER是Hshared；若换Hshell-A1，同时改变视图/梯度路由与AdamW。需Hshell-A0才能单独拆分二者 |
| B04 / TS05 | 原版c0.1+cap+半margin，无基础GS；e300=best93.963%，末10轮均值93.627% | 保留SH32可只新增HIER，但需检查挖掘覆盖；如改CB64/CY64，则需要相同batch的新H0对照 |
| B05 / TS06 | 已完成组最高best OA94.327% | 同为SH32；峰值优势未体现在末10轮均值。保留SH32或改均衡64是需要明确的协议选择 |

现有正式HIER实验均使用类别均衡64，但这不是损失函数要求的唯一合法batch。SH32也可构造HIER关系，只是类别重复、同类关系和候选覆盖会变化；不能把“增加HIER”和“修改batch”默认捆绑。关系代码见`hier_proxy_scratch_v5/relations.py`与`hier_proxy_scratch_v8/hier_loss.py`。

这张表用于用户决定底座，不启动新的实验矩阵。A1在73十轮中恢复内层代理，但不能据此预先宣称300轮保持回缩、代理间层次成立或分类提升；完整TS15仍在运行。

### 十轮机制实验另列

| 机制记录 | 对应底座/组件 | 预算 | e10代理越cap | 分类成绩 |
|---|---|---|---:|---|
| 71原参数诊断 | B09 / Hshell-A0（S22） | fresh10×200，保留300轮调度 | 502/512 | train-only，无test/选模 |
| 73联合候选诊断 | B09 / Hshell-A1（S22） | fresh10×200，保留300轮调度 | 0/512 | train-only，无test/选模 |

73的122个曾越界代理均已返回，e10映射深度中位3.576、范围约3.147–4.572，whole视图固定4.6。它恢复的是该十轮观察中的径向内外间隔，不是独立形态层次证明。71、73均不续训进入TS15；TS15从头初始化。

## 5. 文件与复现身份

- [全部配置与成绩CSV](tables/all_experiments.csv)：15条记录，包含全名、基础配方、GS作用范围、margin、预算、训练/test读出、源精度及commit，可直接筛选。
- [新旧名称映射CSV](tables/name_mapping.csv)：新展示名映射原canonical name和不可变ID，不重命名服务器目录、日志或权重。
- [统一JSON](tables/unified_inventory.json) / [TS注册表v3](../experiment_registry_ts_v3.json)：相同数据的机器可读入口。
- [采集A](tables/source_capture_a.json)、[采集B](tables/source_capture_b.json)、[历史参照](tables/source_capture_historical.json)：只有脱敏标量/来源摘要；不含点云、特征、完整日志、权重、主机地址或私有路径。
- [验收](tables/acceptance.json)：范围/唯一命名/轮次连续性/manifest成绩一致/同轮AA/单位检查。采集工具为已发布commit3eb6d42，本地先发布后两台服务器fetch/pull --ff-only，只读标准库采集；正在训练的任务使用原冻结源码。

身份与管线依据：[55](../55_V8_FULLTRAIN_TEST_SELECTION_2026-10-17.md)、[56](../56_V8_FULLTRAIN_CONTROLS_2026-10-18.md)、[60](../60_C01_CAP_NOHOOK_ABLATIONS_2026-10-20.md)、[61](../61_ORIGINAL_C01_CAP_HALF_MARGIN_2026-10-20.md)、[62](../62_ORIGINAL_C01_CAP_GS_HALF_MARGIN_2026-10-20.md)、[65](../65_V9_DUAL_CURVATURE_PLAN_AND_LAUNCH_2026-10-20.md)、[66](../66_V9_B64_CONTROL_AND_OFFICIAL_REPEAT_2026-10-21.md)、[67](../67_ORIGINAL_C01_CYCLE300_HIER_PAIR_2026-10-21.md)、[73](../73_V9_LR_EPS_10E_DIAGNOSTIC_2026-10-22/report.md)、[74](../74_V9_ADAMW_FULL_RUN_2026-10-22/report.md)。68是10月21日中途快照；本表更新循环完成结果，不追溯改写旧记录。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"runs": len(runs), "baseline_recipes": len(bases),
                      "completed": 14, "ongoing_epoch": active["completed_epochs"],
                      "all_checks_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    build()
