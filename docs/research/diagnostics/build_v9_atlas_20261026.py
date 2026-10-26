"""Extend atlas75 using saved scalar evidence; never import Torch or use GPUs."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    return "—" if value is None else str(Decimal(str(round(value, 9))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows])


def dump_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture8000", type=Path, required=True)
    parser.add_argument("--offline86", type=Path, required=True)
    parser.add_argument("--diagnostic90", type=Path, required=True)
    parser.add_argument("--capture8001", type=Path)
    args = parser.parse_args()
    out = ROOT / "93_V9_EXPERIMENT_ATLAS_2026-10-26"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    old = read(ROOT / "75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/tables/unified_inventory.json")
    inventory = copy.deepcopy(old)
    runs = inventory["runs"]
    bases = inventory["baselines"]
    remote = read(args.capture8000)
    restored = read(args.capture8001) if args.capture8001 else None
    offline = read(args.offline86)["runs"]
    completed83 = read(ROOT / "83_S4780_NATIVE_B05_COMPLETION_2026-10-24/results.json")
    spec = importlib.util.spec_from_file_location("saved_metrics_reader", ROOT / "diagnostics/collect_ts_inventory_20261022.py")
    metric_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_reader)

    for key, source, batch, lineage in (("B11", "B01", "CB64", "HC"), ("B12", "B05", "CB64", "HC"),
                                        ("B13", "B05", "CB32", "HC"), ("B14", "B01", "CB64", "PH")):
        base = copy.deepcopy(next(b for b in bases if b["baseline_id"] == source))
        base.update(baseline_id=key, lineage=lineage, batch=batch, global_batch=int(batch[2:]),
                    local_batch=32, world_size=2 if batch == "CB64" else 1, steps_per_epoch=200,
                    epochs=300, total_updates=60000, workers=8, eval_batch_size=16,
                    rng_protocol="continuous original-derived streams; private seed/epoch/step class plans")
        base["name"] = base["name"].replace("SH32x307", batch + "x200")
        if key == "B14":
            base["name"] = base["name"].replace("HC-", "PH-", 1)
            base["base_losses"] = "CE + .01 sum_200_400_600(Rcontr_g+Rhier_g); shared encoder, four BN passes"
            base["negative_pairing"] = "global batch reversal for each of three part scales; local32 BN"
        bases.append(base)
    base_map = {b["baseline_id"]: b for b in bases}

    for run in runs:
        run["previous_display_name"] = run["display_name"]
        run.update(sample_K=20 if run["hier_enabled"] else None, proxy_K=20 if run["hier_enabled"] else None,
                   lambda_hier=.5 if run["hier_enabled"] else None, warmup_epochs=0 if run["hier_enabled"] else None,
                   record_kind="formal_training", in_v9_and_later=int(run["run_id"][2:]) >= 11)
        run["evidence_note"] = "75号原始已验收记录，精度与运行身份保留"
        if run["run_id"] == "TS15":
            run.update(status="unverified_after_saved_prefix", evidence_note="本机仅有10/22 e34定量快照；断电后尾段未知，8001本次拒绝连接。不能断言当前仍运行或已完成。")

    def new_run(identifier, legacy, base_id, component, ks=20, kp=20, weight=.5, recipe="A1"):
        base = base_map[base_id]
        run = dict(run_id=identifier, id=legacy, baseline_id=base_id, component=component, seed=4780,
                   replicate=None, selection="TS", train_count=9840, validation_count=0, test_count=2468,
                   status="unverified_after_saved_prefix", completed_epochs=None, planned_epochs=300,
                   best_epoch=None, best_test_oa=None, best_test_aa_same_epoch=None,
                   online_train_oa_at_best_test_epoch=None, end_test_oa=None, end_test_aa=None,
                   end_online_train_oa=None, end_clean_train_oa=None, latest_test_epoch=None, latest_test_oa=None,
                   latest_online_train_oa=None, latest_clean_train_epoch=None, latest_clean_train_oa=None,
                   last10_test_oa_mean=None, last10_test_oa_population_std=None,
                   hier_enabled=True, hier_c=.1, hier_view="direction-preserving fixed depth4.6", hier_GS_scope="one dedicated view GS, branched before base GS",
                   direct_radial_hier_gradient="removed at whole-coordinate interface; backbone remains coupled",
                   proxy_tangent_cap=2.3, proxy_GS=True,
                   proxy_optimizer=dict(name="AdamW", lr=.01 if recipe == "A0" else .005,
                                        eps=1e-8 if recipe == "A0" else 1e-4, min_lr=.0005, betas=[.9,.999], weight_decay=.01),
                   production_commit=None, source_document=None, metric_files=None, metric_files_contiguous_from_one=None,
                   source_precision="unrounded saved JSON scalars", sample_K=ks, proxy_K=kp, lambda_hier=weight,
                   warmup_epochs=0, record_kind="formal_training", in_v9_and_later=True)
        runs.append(run)
        return run

    for identifier, arm, base_id in (("TS16", "native", "B11"), ("TS17", "b05", "B12")):
        source = completed83["runs"][arm]
        run = new_run(identifier, source["run_id"], base_id, "Hshell-A1")
        run.update(status="completed", completed_epochs=300, best_epoch=source["best"]["epoch"],
                   best_test_oa=source["best"]["oa"], best_test_aa_same_epoch=source["best"]["aa"],
                   end_test_oa=source["e300"]["oa"], end_test_aa=source["e300"]["aa"],
                   end_online_train_oa=source["e300"]["online_train_oa"], end_clean_train_oa=source["e300"]["clean_train_oa"],
                   latest_test_epoch=300, latest_test_oa=source["e300"]["oa"],
                   latest_online_train_oa=source["e300"]["online_train_oa"], latest_clean_train_epoch=300,
                   latest_clean_train_oa=source["e300"]["clean_train_oa"],
                   last10_test_oa_mean=source["e291_300"]["oa_mean"], last10_test_oa_population_std=source["e291_300"]["oa_sd_population"],
                   production_commit=completed83["source_commit"], source_document="83_S4780_NATIVE_B05_COMPLETION_2026-10-24/results.json",
                   metric_files=300, metric_files_contiguous_from_one=True, finished_beijing=source["finished_beijing"],
                   evidence_note="83保存的两组完整验收及全精度统计；本次8001不可达，沿用已验收结果")
        if identifier == "TS16":
            run["online_train_oa_at_best_test_epoch"] = 96.461
            run["field_precision"] = {"online_train_oa_at_best_test_epoch": "83报告第89行保存三位小数，不反推更多精度"}

    part = new_run("TS18", "parthier_v1_full_checkpoint_q2048_20261024", "B14", "Hpart-v1", ks=20, kp=None)
    part.update(completed_epochs=1, hier_c=1., hier_view="actual parts200/400/600; Q2048 detached FIFO; no fixed whole shell",
                hier_GS_scope="no additional GS", direct_radial_hier_gradient="present; real-part/whole distance and depth objectives",
                proxy_tangent_cap=None, proxy_GS=False, proxy_optimizer=None, warmup_epochs=None,
                ord_chain_weight=.1, depth_margin=.2, fifo_Q=2048, production_commit="82478cd",
                source_document="80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md",
                evidence_note="10/24 09:50首轮200更新/test2468/BN800验收通过；后续完整状态与准确率未在本机保存，8001本次拒绝连接。不是8000 OOM短测。")

    offline_capture = []
    for identifier, variant, component in (("TS19", "a1", "Hshell-A1"), ("TS20", "a0", "Hshell-A0")):
        src = offline[variant]
        manifest = src["manifest"]
        run = new_run(identifier, "b05_cb32_shell_" + variant + "_20261025_r1", "B13", component, 10, 20, .5, variant.upper())
        best = metric_reader.compact_epoch(src["best_metrics"], "saved_best_metrics")
        last = metric_reader.compact_epoch(src["latest_metrics"], "metrics_epoch_300.json")
        run.update(status="completed_saved_then_owned_exit_cleanup", completed_epochs=300,
                   best_epoch=manifest["best"]["epoch"], best_test_oa=best["test"]["oa"], best_test_aa_same_epoch=best["test"]["aa"],
                   online_train_oa_at_best_test_epoch=best["online_train"]["oa"], end_test_oa=last["test"]["oa"],
                   end_test_aa=last["test"]["aa"], end_online_train_oa=last["online_train"]["oa"], end_clean_train_oa=last["clean_train"]["oa"],
                   latest_test_epoch=300, latest_test_oa=last["test"]["oa"], latest_online_train_oa=last["online_train"]["oa"],
                   latest_clean_train_epoch=300, latest_clean_train_oa=last["clean_train"]["oa"],
                   production_commit=manifest["commit"], legacy_canonical_name=manifest["canonical_name"],
                   source_document="86_B05_SINGLECARD_A0_A1_QUEUE_2026-10-25.md; source_capture_offline86.json",
                   metric_files=src["metric_files"], metric_files_contiguous_from_one=True,
                   finished_utc=manifest.get("finished_utc"),
                   evidence_note="本机独立SHA/CRC/CPU加载及60000更新/BN120000验收；保存完成后退出I/O阻塞已清理；末10轮JSON未本机备份")
        offline_capture.append(dict(run_id=identifier, canonical_name=manifest["canonical_name"], commit=manifest["commit"],
                                    status=manifest["status"], completed_epochs=300, metric_files=src["metric_files"],
                                    actual_config={k:manifest["training_config"].get(k) for k in ("sample_K","proxy_K","lambda_hier_after_warmup","proxy_optimizer","global_batch","seed","steps_per_epoch")},
                                    best=best, latest=last))

    for src in remote["rows"]:
        cfg = src["actual_config"]
        run = new_run(src["id"], src["canonical_name"], "B13", "Hshell-A1", cfg["sample_K"], cfg["proxy_K"], cfg["lambda_hier_after_warmup"])
        is_end = src["completed_epochs"] == 300
        best, latest = src["best"], src["latest"]
        run.update(status=src["status"], completed_epochs=src["completed_epochs"], best_epoch=src["best_epoch"],
                   best_test_oa=best["test"]["oa"], best_test_aa_same_epoch=best["test"]["aa"],
                   online_train_oa_at_best_test_epoch=best["online_train"]["oa"],
                   end_test_oa=latest["test"]["oa"] if is_end else None, end_test_aa=latest["test"]["aa"] if is_end else None,
                   end_online_train_oa=latest["online_train"]["oa"] if is_end else None,
                   end_clean_train_oa=latest["clean_train"]["oa"] if is_end else None,
                   latest_test_epoch=latest["completed_epoch"], latest_test_oa=latest["test"]["oa"],
                   latest_online_train_oa=latest["online_train"]["oa"],
                   last10_test_oa_mean=src["tail_test_oa_mean"] if is_end else None,
                   last10_test_oa_population_std=src["tail_test_oa_population_std"] if is_end else None,
                   production_commit=src["production_commit"], legacy_canonical_name=src["canonical_name"],
                   source_document=("91_B05_CB32_A1_K_WEIGHT_ABLATIONS_2026-10-26.md" if src["id"] in ("TS21","TS22") else "92_B05_CB32_A1_K10_WEIGHT_FOLLOWUP_2026-10-26.md"),
                   metric_files=src["metric_files"], metric_files_contiguous_from_one=src["metric_files_contiguous_from_one"],
                   snapshot_source="source_capture_8000.json", observed_utc=remote["observed_utc"],
                   evidence_note="本次8000只读采集，完成前缀冻结，运行中尾段不填e300")
        clean = src.get("latest_clean_train_record")
        if clean:
            run.update(latest_clean_train_epoch=clean["completed_epoch"], latest_clean_train_oa=clean["clean_train"]["oa"])

    if restored:
        by_id = {r["run_id"]: r for r in runs}
        for src in restored["rows"]:
            run = by_id[src["id"]]
            best, latest = src["best"], src["latest"]
            is_end = src["completed_epochs"] == 300 and src["status"] == "completed"
            interrupted = src["id"] == "TS15" and not is_end and src["matching_process_count"] == 0
            run.update(status="interrupted_saved_prefix" if interrupted else
                       "completed_saved_then_owned_exit_cleanup" if src["id"] in ("TS19","TS20") and is_end else src["status"],
                       completed_epochs=src["completed_epochs"], best_epoch=src["best_epoch"],
                       best_test_oa=best["test"]["oa"], best_test_aa_same_epoch=best["test"]["aa"],
                       online_train_oa_at_best_test_epoch=best["online_train"]["oa"],
                       end_test_oa=latest["test"]["oa"] if is_end else None,
                       end_test_aa=latest["test"]["aa"] if is_end else None,
                       end_online_train_oa=latest["online_train"]["oa"] if is_end else None,
                       end_clean_train_oa=latest["clean_train"]["oa"] if is_end else None,
                       latest_test_epoch=latest["completed_epoch"], latest_test_oa=latest["test"]["oa"],
                       latest_online_train_oa=latest["online_train"]["oa"],
                       last10_test_oa_mean=src["tail_test_oa_mean"] if is_end else None,
                       last10_test_oa_population_std=src["tail_test_oa_population_std"] if is_end else None,
                       production_commit=src["production_commit"], metric_files=src["metric_files"],
                       metric_files_contiguous_from_one=src["metric_files_contiguous_from_one"],
                       snapshot_source="source_capture_8001_restored.json", observed_utc=restored["observed_utc"],
                       finished_utc=src.get("finished"),
                       evidence_note="8001恢复后本次直接只读采集已有manifest、best/last轮及末10轮，未重新评估或训练")
            run.pop("field_precision", None)
            clean = src.get("latest_clean_train_record")
            if clean:
                run.update(latest_clean_train_epoch=clean["completed_epoch"], latest_clean_train_oa=clean["clean_train"]["oa"])
            if interrupted:
                run["evidence_note"] = "已保存54个连续完整轮次；manifest旧running标记未清除，但当前无对应任务进程。不恢复训练，不把54轮当300轮。"
            if src.get("canonical_name"):
                run["legacy_canonical_name"] = src["canonical_name"]

    for run in runs:
        base = base_map[run["baseline_id"]]
        run.update({k:base[k] for k in ("base_c","base_tangent_cap","base_GS","gamma","delta","batch","global_batch","local_batch","world_size","steps_per_epoch","total_updates","workers","eval_batch_size","rng_protocol")})
        run["proxy_initial_lr"] = run["proxy_optimizer"]["lr"] if run.get("proxy_optimizer") else None
        run["proxy_eps"] = run["proxy_optimizer"]["eps"] if run.get("proxy_optimizer") else None
        component = run["component"]
        if component not in ("H0", "Hpart-v1"):
            component += f"-Ks{run['sample_K']}-Kp{run['proxy_K']}-L{run['lambda_hier']:g}-W0"
        run["component_full"] = component
        suffix = f"S{run['seed']}" + (", " + run["replicate"] if run.get("replicate") else "")
        run["display_name"] = f"{base['name']} / {component} ({suffix})"

    gaps = []
    for run in runs:
        for field in ("best_test_oa", "best_test_aa_same_epoch", "online_train_oa_at_best_test_epoch", "end_test_oa", "end_online_train_oa", "end_clean_train_oa", "last10_test_oa_mean"):
            if run.get(field) is None:
                pending = run["status"] == "running" and field.startswith(("end_", "last10_"))
                reason = ("本次确认TS15只保存至e54且无对应进程，e300不存在；不是资料丢失" if run["status"] == "interrupted_saved_prefix" else
                          "PartHier既有训练协议未记录clean/eval训练OA；不追加新前向补造历史值" if restored and run["run_id"]=="TS18" and field=="end_clean_train_oa" else
                          "尚未训练到e300；不属于遗失数据" if pending else
                          "8001拒绝连接；仅有历史保存前缀，后续需读取原任务已有文件" if run["run_id"] in ("TS15","TS18") else
                          "8001拒绝连接，末10轮未本机备份；best/e300及完成验收已取得" if run["run_id"] in ("TS19","TS20") and field.startswith("last10") else
                          "83汇总未包含best同轮在线训练OA；8001不可达，待读取该轮已有JSON" if run["run_id"]=="TS17" and field=="online_train_oa_at_best_test_epoch" else
                          "75原始记录未包含此口径；未将在线训练OA或其他轮次代填")
                category = ("not_completed_epoch300" if run["status"]=="interrupted_saved_prefix" else
                            "pending_epoch300" if pending else
                            "not_recorded_by_protocol" if restored and (run["run_id"]=="TS18" or int(run["run_id"][2:])<=6) else "unavailable_evidence")
                gaps.append(dict(run_id=run["run_id"], field=field, category=category, reason=reason))

    diagnostics = [
        dict(record="D01", baseline_id="B09", component="Hshell-A0", seed=22, batch="CB64", sample_K=20, proxy_K=20, lambda_hier=.5, lr=.01, eps=1e-8, budget="10x200; schedule300", cap_e10=502, classification="train-only，无test", source="71_V9_PROXY_CAP_CAUSE_2026-10-22/report.md"),
        dict(record="D02", baseline_id="B09", component="Hshell-A1", seed=22, batch="CB64", sample_K=20, proxy_K=20, lambda_hier=.5, lr=.005, eps=1e-4, budget="10x200; schedule300", cap_e10=0, classification="train-only，无test", source="73_V9_LR_EPS_10E_DIAGNOSTIC_2026-10-22/report.md"),
        dict(record="D03", baseline_id="B12", component="Hshell-A2", seed=4780, batch="CB64", sample_K=20, proxy_K=20, lambda_hier=.5, lr=.005, eps=1e-8, budget="10x200; schedule300", cap_e10=500, classification="train-only，无test", source="88_B05_EPS_ONLY_10E_2026-10-25/report.md"),
        dict(record="D04", baseline_id="B12", component="Hshell-A1", seed=4780, batch="CB64", sample_K=20, proxy_K=20, lambda_hier=.5, lr=.005, eps=1e-4, budget="10x200; schedule300", cap_e10=0, classification="train-only，无test", source="88_B05_EPS_ONLY_10E_2026-10-25/report.md"),
        dict(record="D05", baseline_id="B13", component="Hshell-A0", seed=4780, batch="CB32", sample_K=10, proxy_K=20, lambda_hier=.5, lr=.01, eps=1e-8, budget="fresh10x200; schedule300", cap_e10=511, classification="保留原test/随机流；fresh复跑，不能当历史缺失轮恢复", source="90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/report.md"),
    ]
    for d in diagnostics:
        d["display_name"] = f"{base_map[d['baseline_id']]['name']} / {d['component']}-Ks{d['sample_K']}-Kp{d['proxy_K']}-L{d['lambda_hier']:g}-E10-SCHED300 (S{d['seed']})"
    d90 = read(args.diagnostic90)
    diagnostics[-1].update(best_epoch_in_first10=d90["best_epoch"], best_test_oa_in_first10=d90["best"]["test"]["oa"],
                           best_test_aa_same_epoch=d90["best"]["test"]["aa"], e10_test_oa=d90["latest"]["test"]["oa"],
                           e10_test_aa=d90["latest"]["test"]["aa"], e10_online_train_oa=d90["latest"]["online_train"]["oa"],
                           e10_clean_train_oa=d90["latest"]["clean_train"]["oa"])

    inventory.update(schema_version=4, naming_document="docs/research/93_V9_EXPERIMENT_ATLAS_2026-10-26/report.md",
                     scope="TS-only formal training: V8, original HyCoRe adaptations, V9 and PartHier; validation-selected runs excluded",
                     observed_utc=max(remote["observed_utc"],restored["observed_utc"]) if restored else remote["observed_utc"], formal_record_count=len(runs), mechanisms=diagnostics,
                     evidence_gaps=gaps, source_atlas75_sha256=hashlib.sha256((ROOT/"75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/tables/unified_inventory.json").read_bytes()).hexdigest(),
                     collection_policy="Read-only saved scalar evidence; no training/evaluation/checkpoint mutation; " + ("8001 restored and re-collected" if restored else "8001 refused connection"))
    save(tables / "unified_inventory.json", inventory)
    save(ROOT / "experiment_registry_ts_v4.json", inventory)
    save(tables / "source_capture_8000.json", remote)
    save(tables / "source_capture_offline86.json", dict(scope="Sanitized scalar excerpt of previously accepted local backup", source_backup_sha256=hashlib.sha256(args.offline86.read_bytes()).hexdigest(), rows=offline_capture))
    save(tables / "source_capture_diagnostic90.json", d90)
    if restored:
        save(tables / "source_capture_8001_restored.json", restored)
    dump_csv(tables / "all_experiments.csv", runs)
    dump_csv(tables / "name_mapping.csv", [{k:r.get(k) for k in ("run_id","id","baseline_id","display_name","previous_display_name","legacy_canonical_name","production_commit","source_document")} for r in runs])
    dump_csv(tables / "mechanism_experiments.csv", diagnostics)
    dump_csv(tables / "data_gaps.csv", gaps)

    assert [r["run_id"] for r in runs] == [f"TS{i:02d}" for i in range(1,25)]
    assert len(set(r["display_name"] for r in runs)) == len(runs)
    assert all(r["end_test_oa"] is not None for r in runs if r["status"].startswith(("completed", "historical_completed")))
    assert all(r["end_test_oa"] is None for r in runs if r["status"] == "running")
    for old_run, new in zip(old["runs"],runs):
        assert old_run["run_id"] == new["run_id"] and old_run["baseline_id"] == new["baseline_id"]
        if not restored or old_run["run_id"] != "TS15":
            assert old_run["best_test_oa"] == new["best_test_oa"] and old_run["end_test_oa"] == new["end_test_oa"]
    for src in remote["rows"] + (restored["rows"] if restored else []):
        assert abs(src["manifest_best_test_oa"] - src["best"]["test"]["oa"]) < 1e-10
        assert src["metric_files_contiguous_from_one"]
        cfg=src["actual_config"]
        if src in remote["rows"]:
            assert (cfg["seed"],cfg["global_batch"],cfg["steps_per_epoch"],cfg["proxy_optimizer"]["lr"],cfg["proxy_optimizer"]["eps"]) == (4780,32,200,.005,1e-4)
    for path in tables.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in ("/mnt/", "jhr@", "59.72.109.250", "PRIVATE KEY", "/tmp/hycore_")), path.name
    save(tables / "acceptance.json", dict(status="passed", formal_records=24, preserved_original_ids=15, new_formal_records=9,
         baseline_recipes=len(bases), mechanism_records=len(diagnostics), all_aliases_unique=True, old14_completed_metrics_unchanged=True,
         TS15_original_identity_preserved=True, TS15_saved_prefix_refreshed=bool(restored),
         best_metrics_match_saved_manifest=True, running_epoch300_not_fabricated=True, no_torch_or_gpu=True,
         unknown_or_missing_fields_explicit=True, private_paths_excluded_from_public_capture=True))

    def status(r):
        if r["status"] == "running": return f"进行中 e{r['completed_epochs']}/300"
        if r["status"] == "interrupted_saved_prefix": return f"已中断；保存至e{r['completed_epochs']}"
        if r["status"].startswith("unverified"): return f"尾段未知；已验e{r['completed_epochs']}"
        if r["run_id"] in ("TS19","TS20"): return "完成；退出残留已清理"
        return "完成；历史续训" if r["run_id"] == "TS01" else "完成"

    beijing=dt.datetime.fromisoformat(inventory["observed_utc"]).astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
    completed=sum(r["status"].startswith(("completed","historical_completed")) for r in runs)
    interrupted_count=sum(r["status"]=="interrupted_saved_prefix" for r in runs)
    running_count=sum(r["status"]=="running" for r in runs)
    unknown_count=sum(r["status"].startswith("unverified") for r in runs)
    sections=["# V9至今：TS实验统一命名、全量成绩与缺失数据补录", "",
              f"采集截止：北京时间**{beijing}**。正式记录24条（TS01–TS24），其中{completed}条已完成、{running_count}条进行中、{interrupted_count}条已中断、{unknown_count}条后续状态待核实；14种底座配方，5条独立十轮机制记录。V9及后续主范围为TS11–TS24共14条，TS01–TS10作为75原表的历史参照保留。", "",
              "本表接续[75原图谱](../75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md)：**保留TS01–TS15及B01–B10原编号**，新增TS16–TS24、B11–B14。旧服务器目录、日志和权重不改名；新增名称是展示别名。范围继续排除VS984验证选模实验。只读现有JSON、历史报告和已验收本机备份，不启动训练、模型前向或BN校准。", "",
              "## 1. 统一命名：底座、组件、采样与优化器分开", "",
              "完整格式：`{HC|V8|V9|PH}-C{1|01}-Cap{0|23}-GS{0|1}-M{1000_4|500_2}-{SH32|CB32|CB64|CY64}x{步数} / {组件}-Ks{样本K}-Kp{代理K}-L{lambda}-W0 (S种子[, HIST|R2])`。H0省略不存在的K/代理参数；PartHier使用Hpart-v1并单独列FIFO/排序项。seed只作后缀和身份元数据，不分配新底座编号。", "",
              "C01=c0.1；Cap23=切空间上限2.3；GS只指基础whole/part出口，HIER分支GS另列。M500_2指两个margin为500/N与2，两项基础损失权重仍各0.01。CB32每步16类×有放回2例、单卡；CB64每步32类×2例、双卡；CY64是循环队列且每轮300步；SH32是shuffle/drop-last每轮307步。", "",
              "**B05严格保留原义：HC、SH32×307。** 后续口头称B05的几何基底，改CB64后为B12，改CB32后为B13，不能继续把三种训练协议全部写成B05。B11与旧B09几何近似，但执行谱系、workers/eval及连续随机流不同，保留不同底座身份。PartHier三尺度基础项和四次BN另作B14，不冒充只替换了一个HIER损失。", "",
              table(["底座","统一底座全名","global/local/卡数","workers/eval","来源"],[[b["baseline_id"],b["name"],f"{b['global_batch']}/{b['local_batch']}/{b['world_size']}",f"{b['workers']}/{b['eval_batch_size']}","75原编号" if int(b["baseline_id"][1:])<=10 else {"B11":"原生c1改CB64，连续流","B12":"B05几何改CB64","B13":"B05几何改CB32","B14":"PartHier三尺度intra基底"}[b["baseline_id"]]] for b in bases]), "",
              "### 组件与AdamW", "",
              table(["组件","whole/候选视图","代理优化器"],[["H0","不构造HIER","无"],["Hshared-A0","基础共享cap/GS后的实际whole半径","0.01 / 1e-8"],["Hshell-A0","c0.1、固定whole深度4.6、专用GS","0.01 / 1e-8"],["Hshell-A1","同Hshell-A0","0.005 / 1e-4"],["Hshell-A2（仅D03）","同Hshell-A0","0.005 / 1e-8"],["Hpart-v1","c1真实part200/400/600＋Q2048，whole不固定壳","无自由代理/AdamW；共享RSGD"]]), "",
              "A0/A1/A2中的数值均为代理AdamW初始LR/eps；其余betas0.9/0.999、WD0.01、最低LR0.0005及300轮余弦。固定壳组件P512/D256、0.9倍高斯初始化、前向cap2.3与自身GS、无更新后投影；margin/tau0.1、T50、W0、自负k排除。实际Ks/Kp/lambda按行列出，不再默认全部K20/lambda0.5。", "",
              "## 2. 全部正式训练记录与分类成绩", "",
              "OA、AA单位均为%。AA取best OA同一轮，e300取末轮而非final_test中的best摘要。末10轮固定e291–300，SD为总体标准差，不是多seed误差条。运行中best只是完成前缀内best，不参与完成结果排名。", "",
              table(["记录/底座","实验全称（seed）","状态","best轮","best OA","同轮AA","e300 OA","e291–300 OA均值±SD"],[[r["run_id"]+" / "+r["baseline_id"],r["display_name"],status(r),r.get("best_epoch") or "—",number(r.get("best_test_oa")),number(r.get("best_test_aa_same_epoch")),number(r.get("end_test_oa")),"—" if r.get("last10_test_oa_mean") is None else number(r["last10_test_oa_mean"])+" ± "+number(r["last10_test_oa_population_std"])] for r in runs]), "",
              "TS01为含resume的历史锚点，TS02为fresh重跑；不合并。TS02–TS06原日志仅保留0.001个百分点，CSV不反推伪造精度。" +
              ("TS15现已核实54个连续完整轮次且无对应任务进程；manifest残留running不代表当前还在训练。TS18已从8001原始记录补齐300轮，不能继续写成首轮后未知。" if restored else "TS15仍仅有e34本机定量证据，不能把旧running状态当作当前运行；TS18只确认首轮通过，不填无来源的best或e300。") +
              "TS19/20训练保存完整但退出I/O阻塞，10/25已按用户授权清理对应残留，不能把队列旧running/failed误写成训练没完成。", "",
              "### V9及后续的开关核对表", "",
              table(["记录","底座c/cap/基础GS","batch×步","intra margin","HIER视图/分支GS","Ks/Kp","lambda","代理LR/eps","seed"],[[r["run_id"],f"{r['base_c']}/{r['base_tangent_cap'] or '无新增'}/{int(r['base_GS'])}",f"{r['batch']}×{r['steps_per_epoch']}",f"{r['gamma']}/N；{r['delta']}","无" if not r["hier_enabled"] else "真实多尺度part/c1/无新增GS" if r["component"]=="Hpart-v1" else "固定4.6/c0.1/独立GS",f"{r['sample_K'] or '—'}/{r['proxy_K'] or '—'}",number(r.get("lambda_hier")),"无" if r.get("proxy_optimizer") is None else f"{r['proxy_initial_lr']:g}/{r['proxy_eps']:g}",r["seed"]] for r in runs if r["in_v9_and_later"]]), "",
              "PartHier另外启用Q2048、排序/链项权重0.1、深度margin0.2；基础intra为三个尺度各0.01相加。与自由代理固定壳不是单变量替换。", "",
              "### 训练准确率：在线与clean/eval分开", "",
              "在线训练OA来自增强和均衡有放回抽样；clean/eval训练OA来自无训练增强、eval模式的完整9840对象。两者不能互换，缺失时不借用其他轮次或test OA。最新clean列显式标轮次，避免把中途值放进e300。", "",
              table(["记录","best OA同轮在线train OA","e300在线train OA","e300 clean/eval train OA","最近clean/eval（轮次）"],[[r["run_id"],number(r.get("online_train_oa_at_best_test_epoch")),number(r.get("end_online_train_oa")),number(r.get("end_clean_train_oa")),"—" if r.get("latest_clean_train_oa") is None else number(r["latest_clean_train_oa"])+f" (e{r['latest_clean_train_epoch']})"] for r in runs]), "",
              "## 3. 时间线与可比较的消融", "",
              table(["时间","记录","想验证什么 / 实际变化","已有结果或边界"],[["10/20–22","TS11/12/15","V9将原生主pipe与c0.1固定壳HIER分叉；A0→A1调整代理AdamW","A0末轮分类回落；A1完整尾段未取得"],["10/21–22","TS13/14","连续流循环64；加入固定壳HIER-A0","best和e300均约+0.486pp；不能单独归因循环优于有放回"],["10/23–24","TS16/17","原生c1及B05几何分别配CB64、新HIER-A1","代理回内已验收，但与旧SH32的94.0/94.3不是匹配H0"],["10/24","TS18","PartHier真实多尺度局部祖先＋历史FIFO，替代自由代理","已验首轮；完整尾段待8001恢复读取"],["10/25","TS19/20","B05几何改CB32；A1与A0组合对照","A0 best更高；LR和eps联合改变，不能拆成eps单因素"],["10/26凌晨","TS21/22","减半Ks/Kp，或lambda0.5→0.1","两组已完成；TS21同时改两种K，TS22为权重单变量"],["10/26上午","TS23/24","以TS22为基准，分别只改Kp20→10及lambda0.1→0.05","本次截取e17；尚无终点结论"]]), "",
              table(["对照","变量","best OA差（pp）","e300 OA差（pp）","解释范围"],[[a+" → "+b,what,number(next(r for r in runs if r["run_id"]==b)["best_test_oa"]-next(r for r in runs if r["run_id"]==a)["best_test_oa"]) if next(r for r in runs if r["run_id"]==b)["status"]!="running" else "未完成",number(next(r for r in runs if r["run_id"]==b)["end_test_oa"]-next(r for r in runs if r["run_id"]==a)["end_test_oa"]) if next(r for r in runs if r["run_id"]==b)["end_test_oa"] is not None else "未完成",limit] for a,b,what,limit in [("TS19","TS20","代理LR/eps A1→A0","联合优化器配方"),("TS19","TS21","Ks10/Kp20→Ks5/Kp10","同时调整两种K"),("TS19","TS22","lambda0.5→0.1","同协议权重单变量"),("TS22","TS23","仅Kp20→10","待完整训练"),("TS22","TS24","仅lambda0.1→0.05","待完整训练")]]), "",
              "上述均为单seed/单次轨迹的观测差异；参数改变会使后续训练路径分叉，不代表统计显著性。正确性修复、协议变化和层次目标效果分开归因。未把代理回缩写成独立形态层次证明。", "",
              "## 4. 十轮机制实验：另列，不伪装成300轮分类结果", "",
              table(["记录/底座","组件（seed）","batch/Ks/Kp/lambda","LR/eps","实际预算/调度","e10越cap","分类口径"],[[d["record"]+" / "+d["baseline_id"],d["component"]+f" (S{d['seed']})",f"{d['batch']}/{d['sample_K']}/{d['proxy_K']}/{d['lambda_hier']}",f"{d['lr']}/{d['eps']}",d["budget"],f"{d['cap_e10']}/512",d["classification"]] for d in diagnostics]), "",
              "D01/D02不续训成TS15；D03/D04不并入完整TS17；D05是fresh同配方复跑，与TS20原e1/e5辅助端点不同，不是恢复原历史未保存轮次。D03的A2只是本总账为LR0.005/eps1e-8新增的明确别名，不改写旧文件。", "",
              f"D05保留分类读出：前十轮best为e{d90['best_epoch']}，OA **{number(d90['best']['test']['oa'])}%**、同轮AA **{number(d90['best']['test']['aa'])}%**；e10 OA **{number(d90['latest']['test']['oa'])}%**，在线训练OA **{number(d90['latest']['online_train']['oa'])}%**、clean/eval训练OA **{number(d90['latest']['clean_train']['oa'])}%**。这仍是10轮结果，不与e300成绩混排。原始全精度值见机制CSV。", "",
              "### 其他已做工作与短测登记边界", "",
              table(["记录组","内容","是否正式分类训练"],[["71/72","径向梯度、夹角/测地中点、固定状态Adam一步重放","否；不把状态重放当新训练轨迹"],["73/79/86/88/90/91/92验收短测","独立train-only 2×2等GPU/保存合同，短测后fresh","否；与对应正式/机制记录关联，不重复计数"],["PartHier 8000普通四pass及CPU保存尝试","32GiB容量不足或连接中断；未放行为正式训练","否；失败预备尝试保留80记录"],["PartHier 8001 all / small_medium checkpoint","满Q2048容量、BN/RNG保持、保存回归；small_medium最终采用","否；不是两条完整PartHier消融"],["85/90冻结嵌入审查","读取已有权重，whole/代理几何及分布统计","否；数据源与训练记录绑定"],["均衡32的200/300步覆盖率重放","CPU重放采样；没有修改训练配置","否；没有新分类结果"]]), "",
              "## 5. 本次补录、剩余缺口与来源", "",
              "**补录完成：** 新增9条正式身份、4种底座和5条机制身份；补齐TS16/17、TS19/20、TS21/22的完整best OA/同轮AA/e300及可取得的训练OA；采集TS23/24的完成前缀。75中TS15的旧running状态改为尾段待核实，保留其原e34数值，不追溯改写75原报告。", "",
              "**8001当前拒绝连接，仍有明确缺口：** TS15的断电后尾段；TS18首轮后的进度、best及终点；TS19/20的末10轮均值/SD（本机只有best/末轮原始记录及300份文件存在的验收）；TS17的best同轮在线训练OA。TS16此项已从83报告补录为96.461%，仅保留报告三位小数，不反推更多精度。本次没有用其他seed、相似配置、最终best摘要或推算值代填。TS23/24尚未到e300属于正常待完成，不是资料丢失。", "",
              "所有空值和原因见[data_gaps.csv](tables/data_gaps.csv)。旧历史实验未记录clean/eval时保留空值，并与服务器不可达造成的缺失区分。未启动新训练或自动等待服务器恢复。", "",
              "- [完整配置/成绩CSV](tables/all_experiments.csv)：24条正式记录；精度保留原JSON或原日志。",
              "- [新旧命名对应CSV](tables/name_mapping.csv)：TS、底座、旧别名、源commit映射。",
              "- [十轮机制CSV](tables/mechanism_experiments.csv)、[统一JSON](tables/unified_inventory.json)、[TS注册表v4](../experiment_registry_ts_v4.json)。",
              "- [本次8000脱敏采集](tables/source_capture_8000.json)、[本机86脱敏证据](tables/source_capture_offline86.json)、[验收](tables/acceptance.json)。",
              "- 旧15条源：[75](../75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md)；双卡完整结果：[83](../83_S4780_NATIVE_B05_COMPLETION_2026-10-24/report.md)；PartHier：[80](../80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)；新单卡配置：[86](../86_B05_SINGLECARD_A0_A1_QUEUE_2026-10-25.md)、[91](../91_B05_CB32_A1_K_WEIGHT_ABLATIONS_2026-10-26.md)、[92](../92_B05_CB32_A1_K10_WEIGHT_FOLLOWUP_2026-10-26.md)。", ""]
    report = "\n".join(sections)
    if restored:
        report = report.replace("A0末轮分类回落；A1完整尾段未取得", "A0末轮分类回落；A1仅54轮后中断，无e300")
        report = report.replace("75中TS15的旧running状态改为尾段待核实，保留其原e34数值，不追溯改写75原报告。", "TS15从原e34快照补至已保存e54，并确认当前无对应进程、归为中断；75原报告仍保留为历史快照，不追溯改写。TS18已补入完整300轮结果。")
        report = report.replace("已验首轮；完整尾段待8001恢复读取", "完成300轮；best92.990/e30088.006，后期回落")
        current = {r["run_id"]:r for r in runs}
        report = report.replace("本次截取e17；尚无终点结论", f"本次分别截取e{current['TS23']['completed_epochs']}/e{current['TS24']['completed_epochs']}；尚无终点结论")
        old_gap = "**8001当前拒绝连接，仍有明确缺口：** TS15的断电后尾段；TS18首轮后的进度、best及终点；TS19/20的末10轮均值/SD（本机只有best/末轮原始记录及300份文件存在的验收）；TS17的best同轮在线训练OA。TS16此项已从83报告补录为96.461%，仅保留报告三位小数，不反推更多精度。本次没有用其他seed、相似配置、最终best摘要或推算值代填。TS23/24尚未到e300属于正常待完成，不是资料丢失。"
        new_gap = "**8001连接恢复后已补齐可恢复记录：** TS15从e34旧快照更新为完整保存至e54，前缀best e52 OA91.856%，当前无对应进程，归为已中断；TS18完整300轮的best/e300/末10轮/在线训练OA已补齐；TS19/20末10轮均值与SD已补齐；TS16/17 best同轮在线训练OA直接取原始轮JSON，替换旧报告精度或空值。\n\n剩余空值均有明确性质：TS15没有完成300轮，因此没有e300及e291–300统计；TS23/24仍在训练，终点尚未产生；PartHier协议未记录clean/eval训练OA，TS01–TS06历史记录也未提供该口径。此次没有新增前向去补造历史训练读出，也没有重启TS15或其他任务。"
        assert old_gap in report
        report = report.replace(old_gap,new_gap)
        report = report.replace("旧历史实验未记录clean/eval时保留空值，并与服务器不可达造成的缺失区分。未启动新训练或自动等待服务器恢复。", "旧历史实验未记录clean/eval时保留空值；与尚未完成的终点区分。8001已恢复，没有继续保留‘连接失败待补’的旧缺口。本次没有启动或停止训练。")
        report = report.replace("[本机86脱敏证据](tables/source_capture_offline86.json)、[验收](tables/acceptance.json)", "[本机86脱敏证据](tables/source_capture_offline86.json)、[8001恢复后的原始标量采集](tables/source_capture_8001_restored.json)、[验收](tables/acceptance.json)")
        report += "\n## 6. 8001恢复后的补录摘要\n\n" + table(["记录","核实状态","best OA / 同轮AA","e300 OA","e291–300均值±SD"], [[current[k]["run_id"],status(current[k]),number(current[k]["best_test_oa"])+" / "+number(current[k]["best_test_aa_same_epoch"]),number(current[k]["end_test_oa"]),"—" if current[k]["last10_test_oa_mean"] is None else number(current[k]["last10_test_oa_mean"])+" ± "+number(current[k]["last10_test_oa_population_std"])] for k in ("TS15","TS18","TS19","TS20")]) + "\n\nPartHier TS18于北京时间10月25日06:35:53完成，best e201；其e300在线训练OA99.515625%。best之后test回落约4.984个百分点，是已有训练记录的观察，不据此单独归因某一损失或结构。\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(out=str(out),records=len(runs),completed=completed,running=sum(r["status"]=="running" for r in runs),unknown=sum(r["status"].startswith("unverified") for r in runs),gaps=len(gaps)),ensure_ascii=False))


if __name__ == "__main__":
    main()
