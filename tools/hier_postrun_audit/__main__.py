"""One entry point for read-only HIER post-training reports."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess

from .adapters import normalize_run
from .inventory import inventory_run, resolve_run_dir


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def epoch_table(path, analyses):
    keys = sorted({key for item in analyses for row in item["epochs"] for key in row.get("metrics", {})})
    header = ["run", "epoch", "model_updates", "logged_model_updates", "proxy_updates",
              "logged_proxy_updates", "phase"] + keys
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for item in analyses:
            name = item.get("display_name", item.get("run_id", "run"))
            for row in item["epochs"]:
                writer.writerow({**{key: row.get(key) for key in header[:7]},
                                 "run": name, **row.get("metrics", {})})


def read_specs(path):
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    specs = value if isinstance(value, list) else value["snapshots"]
    for spec in specs:
        for key in ("cache", "checkpoint", "optimizer_checkpoint", "proxy_cache"):
            if spec.get(key):
                candidate = Path(spec[key])
                if not candidate.is_absolute():
                    spec[key] = str((source.parent / candidate).resolve())
    return specs


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="List saved data and missing capabilities without model loading")
    inventory.add_argument("--run-dir", type=Path, required=True)
    run = sub.add_parser("run", help="Produce a fresh report; CPU logs by default")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--compare-run", type=Path, action="append", default=[])
    run.add_argument("--name")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--snapshot-spec", type=Path)
    run.add_argument("--pointcloud-pool", type=Path, help="Optional existing NPZ with actual clouds/sample_ids/labels for galleries")
    run.add_argument("--epochs", help="Explicit saved epochs/aliases, e.g. 20,100,200,best,last; missing epochs are reported")
    run.add_argument("--extract", action="store_true", help="Explicitly export clean snapshots on one idle GPU")
    run.add_argument("--allow-incomplete", action="store_true", help="Watermark an incomplete source run")
    run.add_argument("--no-plots", action="store_true", help="Explicit numerical-only export; no HTML report")
    for command in (run,):
        command.add_argument("--gpu", type=int)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--population", choices=("panel", "full"), default="panel")
        command.add_argument("--per-class", type=int, default=32)
        command.add_argument("--checkpoint-policy", choices=("standard", "all"), default="standard")
        command.add_argument("--max-checkpoints", type=int, default=12)
        command.add_argument("--max-seconds", type=float, default=3600.)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--seed", type=int, default=22)
        command.add_argument("--with-pointclouds", action="store_true")
    probe = sub.add_parser("backbone", help="One explicitly selected checkpoint: frozen shared encoder gradients on an idle GPU")
    probe.add_argument("--snapshot-spec", type=Path, required=True)
    probe.add_argument("--epoch", type=int, required=True)
    probe.add_argument("--out-dir", type=Path, required=True)
    probe.add_argument("--data-dir", type=Path, required=True)
    probe.add_argument("--gpu", type=int, required=True)
    probe.add_argument("--seed", type=int, default=22)
    probe.add_argument("--microbatch-size", type=int, default=16)
    probe.add_argument("--max-seconds", type=float, default=600.)
    system = sub.add_parser("system", help="Join immutable exports, controlled mechanisms, shape and cross-version evidence")
    system.add_argument("--artifact-dir", type=Path, action="append", required=True)
    system.add_argument("--out-dir", type=Path, required=True)
    system.add_argument("--pointcloud-pool", type=Path)
    system.add_argument("--snapshot-spec", type=Path, action="append", default=[])
    system.add_argument("--mechanism-epochs", default="100,200,best,last")
    system.add_argument("--structure-epochs", default="100,200,best,last")
    system.add_argument("--skip-mechanisms", action="store_true")
    system.add_argument("--skip-structure", action="store_true")
    system.add_argument("--no-plots", action="store_true")
    system.add_argument("--query-count", type=int, default=64)
    system.add_argument("--noise-repeats", type=int, default=8)
    system.add_argument("--mechanism-seconds", type=float, default=1800.)
    system.add_argument("--structure-seconds", type=float, default=1800.)
    system.add_argument("--shape-points", type=int, default=64)
    system.add_argument("--shape-per-class", type=int, default=8)
    system.add_argument("--bootstrap", type=int, default=200)
    system.add_argument("--seed", type=int, default=22)
    delivery = sub.add_parser("delivery", help="Copy public immutable CPU evidence and completed real backbone probes into a fresh offline delivery")
    delivery.add_argument("--system-dir", type=Path, required=True)
    delivery.add_argument("--backbone-dir", type=Path, action="append", default=[])
    delivery.add_argument("--out-dir", type=Path, required=True)
    delivery.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def attach_export_usage(specs, analysis):
    lookup = {row["epoch"]: row for row in analysis.get("proxy_usage", [])
              if row.get("component") == "sample" and row.get("domain") == "noncollision" and row.get("role") == "combined"}
    for spec in specs:
        observed = lookup.get(spec.get("epoch"))
        if observed and spec.get("run_key") == analysis.get("run_id"):
            spec["proxy_activation_counts"] = observed["counts"]
            spec["activation_scope"] = "epoch training sample noncollision pair+triple; distinct from fixed-panel retrieval"
    return specs


def code_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = arguments()
    if args.command == "inventory":
        print(json.dumps(inventory_run(args.run_dir), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.command == "delivery":
        from .delivery import build_delivery
        result = build_delivery(args.system_dir, args.backbone_dir, args.out_dir,
                                make_plots=not args.no_plots)
        print(json.dumps({key: result[key] for key in ("status", "backbone_success_count", "sources_unchanged")}, ensure_ascii=False), flush=True)
        return
    if args.command == "backbone":
        from .backbone_probe import run_backbone_probe
        selected = [spec for spec in read_specs(args.snapshot_spec) if spec.get("epoch") == args.epoch]
        if len(selected) != 1:
            raise ValueError("Backbone probe requires exactly one explicit actual saved epoch in the supplied spec")
        result = run_backbone_probe(selected[0], args.out_dir, args.data_dir, args.gpu,
            seed=args.seed, microbatch_size=args.microbatch_size, max_seconds=args.max_seconds)
        print(json.dumps({"status":"completed","epoch":result["epoch"],"optimizer_updates":0},ensure_ascii=False),flush=True)
        return
    if args.command == "system":
        from .system import run_system_audit
        result = run_system_audit(args.artifact_dir, args.out_dir,
            pointcloud_pool=args.pointcloud_pool, snapshot_spec_files=args.snapshot_spec,
            mechanism_epochs=args.mechanism_epochs.split(","), structure_epochs=args.structure_epochs.split(","),
            do_mechanisms=not args.skip_mechanisms, do_structure=not args.skip_structure,
            make_plots=not args.no_plots, render=not args.no_plots,
            max_mechanism_snapshots=16, max_structure_snapshots=20,
            mechanism_options={"query_count":args.query_count,"noise_repeats":args.noise_repeats,
                               "max_seconds":args.mechanism_seconds,"seed":args.seed},
            structure_options={"shape_points":args.shape_points,"per_class":args.shape_per_class,
                               "bootstrap":args.bootstrap,"max_seconds":args.structure_seconds,"seed":args.seed})
        print(json.dumps({"status":result.get("status", result.get("manifest", {}).get("status")),"output_dir":str(args.out_dir)},ensure_ascii=False),flush=True)
        return
    if args.extract and (args.gpu is None or args.data_dir is None):
        raise ValueError("--extract requires explicit --gpu and --data-dir")
    if args.extract and args.snapshot_spec:
        raise ValueError("Choose existing snapshot spec or new extraction, not both")
    paths = [resolve_run_dir(path) for path in [args.run_dir, *args.compare_run]]
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError("A report requires a new output directory")
    if any(out == path or path in out.parents for path in paths):
        raise ValueError("Report output must be outside every source run directory")
    inventories = [inventory_run(path) for path in paths]
    for inventory in inventories:
        status = inventory.get("status", inventory.get("identity", {}).get("status"))
        if status in ("running", "starting", "failed") and not args.allow_incomplete:
            raise ValueError("Incomplete source: use --allow-incomplete for a clearly marked partial audit")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "hier-postrun-audit-v1", "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_run_ids": [path.name for path in paths],
                "read_only_source": True, "optimizer_updates": 0, "new_test_forward": False,
                "mode": "logs_and_snapshots" if args.extract or args.snapshot_spec else "logs",
                "explicit_gpu_export": args.extract, "allow_incomplete": args.allow_incomplete,
                "audit_code_commit": code_commit(), "source_statuses": [item.get("status") for item in inventories],
                "warnings": []}
    manifest_path = out / "audit_manifest.json"
    write_json(manifest_path, manifest)
    try:
        analyses = [normalize_run(path, name=args.name if index == 0 else None, inventory=inventories[index])
                    for index, path in enumerate(paths)]
        for item in analyses:
            item["run_id"] = item["audit_run_key"]
        write_json(out / "normalized_runs.json", analyses)
        epoch_table(out / "epochs.csv", analyses)
        snapshots = None
        pointcloud_pool = str(args.pointcloud_pool.resolve()) if args.pointcloud_pool else None
        if args.extract:
            from .extract import extract_features
            exported = extract_features(paths[0], out / "feature_exports", args.data_dir, args.gpu,
                policy=args.checkpoint_policy, maximum=args.max_checkpoints,
                population=args.population, per_class=args.per_class, seed=args.seed,
                batch_size=args.batch_size, max_seconds=args.max_seconds,
                with_pointclouds=args.with_pointclouds, epochs=args.epochs)
            specs = attach_export_usage(exported["snapshots"], analyses[0])
            manifest["export_status"] = exported["manifest"]["status"]
            pointcloud_pool = exported.get("pointcloud_pool")
        elif args.snapshot_spec:
            specs = read_specs(args.snapshot_spec)
        else:
            specs = []
        if specs:
            from .snapshots import analyze_snapshots
            snapshots = analyze_snapshots(specs, out / "snapshots")
            if pointcloud_pool:
                snapshots["pointcloud_pool"] = pointcloud_pool
        if args.no_plots:
            manifest["warnings"].append("Numerical-only mode was explicitly requested; no visual report")
            report = None
        else:
            from .report import render_report
            report = render_report(analyses, snapshots, out)
        partial = any(item.get("status") not in (None, "completed") for item in inventories) or manifest.get("export_status") == "partial_budget"
        if snapshots and any(not item.get("available") for item in snapshots.get("availability", [])):
            manifest["warnings"].append("Some requested snapshots were rejected or unavailable; see snapshot_summary.json")
            partial = True
        manifest["warnings"].extend((report or {}).get("warnings", []))
        manifest.update(status="partial" if partial else "completed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        logged_epoch_counts=[len(item["epochs"]) for item in analyses],
                        report=report, warnings=manifest["warnings"] + [
                            warning for item in analyses for warning in item.get("warnings", [])])
        write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "output_dir": str(out),
                          "logged_epochs": manifest["logged_epoch_counts"], "mode": manifest["mode"]},
                         ensure_ascii=False), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
