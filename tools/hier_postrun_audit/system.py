"""Fresh, read-only system audit of already exported HIER training artifacts.

No training, GPU/model/test forward or optimizer update is performed. Source
logs, caches, point clouds, checkpoints and earlier reports remain immutable.
"""
from __future__ import annotations
import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from .longitudinal import analyze_longitudinal, run_key
from .comparison import build_comparison


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _epoch(value):
    return int(value) if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and int(value) == value and value >= 0 else None


def _alias_epoch(value):
    if isinstance(value, dict):
        return next((_epoch(value.get(k)) for k in ("epoch", "completed_epochs", "selected_epoch") if _epoch(value.get(k)) is not None), None)
    return _epoch(value)


def _specs(path):
    value = _read(path)
    records = value if isinstance(value, list) else value.get("snapshots")
    if not isinstance(records, list):
        raise ValueError("Snapshot spec file must contain an explicit snapshots list")
    results = copy.deepcopy(records)
    for spec in results:
        for key in ("cache", "checkpoint", "proxy_cache", "optimizer_checkpoint"):
            if spec.get(key):
                candidate = Path(spec[key])
                spec[key] = str((Path(path).parent / candidate).resolve()) if not candidate.is_absolute() else str(candidate.resolve())
    return results


def _qualified(analysis, artifact):
    old = analysis.get("run_id")
    if not analysis.get("audit_run_key"):
        payload = "|".join((artifact.name, str(analysis.get("display_name")), str(old),
                            str(analysis.get("identity", {}).get("commit")), str(analysis.get("identity", {}).get("seed"))))
        stem = str(analysis.get("storage_run_id") or old or "run")
        analysis["audit_run_key"] = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:64] + "_" + hashlib.sha256(payload.encode()).hexdigest()[:8]
    analysis.setdefault("storage_run_id", old)
    analysis["run_id"] = analysis["audit_run_key"]
    return analysis


def _owner(spec, analyses):
    value = spec.get("run_key")
    exact = [r for r in analyses if value in (r.get("audit_run_key"), r.get("run_id"))]
    if len(exact) == 1:
        return exact[0]
    candidates = [r for r in analyses if value in (r.get("storage_run_id"), r.get("display_name"))]
    if len(candidates) == 1:
        return candidates[0]
    if not value and len(analyses) == 1:
        return analyses[0]
    raise ValueError("Snapshot run identity is ambiguous or absent from its normalized source artifact")


def _usage(spec, analysis, warnings):
    import numpy as np
    records = [row for row in analysis.get("proxy_usage", []) if row.get("epoch") == spec["epoch"]
               and row.get("component") == "sample" and row.get("domain") == "noncollision" and row.get("role") == "combined"]
    if not records:
        return spec
    if len(records) != 1:
        raise ValueError("Ambiguous saved sample noncollision usage for a snapshot epoch")
    record = records[0]
    with np.load(spec["cache"], allow_pickle=False) as archive:
        ids = archive["proxy_ids"].copy().reshape(-1) if "proxy_ids" in archive else spec.get("proxy_ids")
    if ids is None:
        warnings.append(spec["run_key"] + ": explicit proxy IDs missing; recorded counts were not positionally guessed")
        return spec
    ids = list(ids)
    if set(ids) != set(record.get("proxy_ids", [])):
        raise ValueError("Training proxy IDs and saved snapshot parameter-row IDs disagree")
    lookup = dict(zip(record["proxy_ids"], record["counts"]))
    spec["proxy_activation_counts"] = [lookup[value] for value in ids]
    spec["activation_scope"] = "epoch training sample noncollision pair+triple; distinct from fixed-panel retrieval"
    return spec


def _prepare_spec(spec, analyses, warnings):
    spec = copy.deepcopy(spec)
    analysis = _owner(spec, analyses)
    spec["run_key"] = run_key(analysis)
    spec["display_name"] = analysis.get("display_name", spec["run_key"])
    spec["storage_run_id"] = analysis.get("storage_run_id", spec.get("storage_run_id"))
    if spec.get("version") is None and analysis.get("identity", {}).get("version"):
        spec["version"] = str(analysis["identity"]["version"]).lower()
    if _epoch(spec.get("epoch")) is None:
        raise ValueError("Working snapshot spec needs a real saved integer epoch")
    spec["epoch"] = int(spec["epoch"])
    if not spec.get("cache") or not Path(spec["cache"]).is_file():
        raise ValueError("Explicit source cache is missing; the system runner does not export or invent embeddings")
    spec["equal_radius_fraction"] = .5
    return _usage(spec, analysis, warnings)


def _selection(specs, analyses, aliases, tokens, maximum):
    tokens = ("100", "200", "best", "last") if tokens is None else tokens
    if isinstance(tokens, str):
        tokens = [x.strip() for x in tokens.split(",") if x.strip()]
    selected, availability = [], []
    by_key = {}
    for spec in specs:
        by_key.setdefault(spec["run_key"], {})[spec["epoch"]] = spec
    for analysis in analyses:
        key = run_key(analysis)
        values = by_key.get(key, {})
        wanted = set()
        for token in tokens:
            text = str(token).lower()
            if text == "all":
                wanted.update(values)
                continue
            if text in ("best", "last"):
                epoch = _alias_epoch(aliases.get(key, {}).get(text))
                if epoch is None and text == "best":
                    epoch = _alias_epoch(analysis.get("identity", {}).get("best"))
                if epoch is None and text == "last":
                    epoch = _epoch(analysis.get("inventory", {}).get("manifest_completed_epochs"))
                if epoch is None:
                    availability.append({"run_key": key, "requested": text, "available": False,
                                         "reason": "Actual saved epoch alias is unavailable; log/cache maxima were not guessed"})
                    continue
            else:
                try:
                    epoch = int(text)
                except ValueError as exc:
                    raise ValueError("Epoch selectors are actual integer epochs, best, last or all") from exc
                if epoch < 0:
                    raise ValueError("Epoch selectors must be nonnegative")
            wanted.add(epoch)
            availability.append({"run_key": key, "requested": str(token), "epoch": epoch,
                                 "available": epoch in values,
                                 "reason": None if epoch in values else "Requested actual saved checkpoint/cache is not in these artifacts"})
        selected.extend(values[epoch] for epoch in sorted(wanted & set(values)))
    if len(selected) > maximum:
        raise ValueError("Selected snapshot count exceeds the explicit CPU audit budget")
    return selected, availability


def _consistent_reuse(summary, spec):
    if summary.get("source", {}).get("cache_sha256") != _sha(spec["cache"]):
        return False
    if summary.get("whole", {}).get("c") != float(spec["c"]):
        return False
    if summary.get("proxy", {}).get("mapped_available") and summary["proxy"].get("mapping") != spec.get("proxy_mapping"):
        return False
    if summary.get("source", {}).get("input_sha256") != spec.get("input_sha256"):
        return False
    if "proxy_activation_counts" in spec:
        activation = summary.get("proxy", {}).get("training_activation", {})
        ids = summary.get("retrieval", {}).get("proxy_ids", [])
        if len(ids) != len(spec["proxy_activation_counts"]):
            return False
        used = {pid for pid, count in zip(ids, spec["proxy_activation_counts"]) if count > 0}
        if activation.get("total_selection_slots") != sum(spec["proxy_activation_counts"]):
            return False
        if activation.get("scope") != spec.get("activation_scope"):
            return False
        if set(activation.get("used_proxy_ids", [])) != used:
            return False
    return True


def _equal_radius(summary, arrays, spec):
    """Equal-radius distance is monotone in cosine; reuse audited direction IDs.

    Selected distances are recomputed, avoiding the invalid nonlinear transform
    of mean/std quantiles. This control changes no production embedding/kernel.
    """
    import numpy as np
    from .geometry import ball_geometry, expmap0, quantiles
    direction = summary.get("retrieval", {}).get("direction", {})
    if not direction.get("available", summary.get("retrieval", {}).get("available", False)) or not direction.get("topk_sample_ids"):
        return
    with np.load(spec["cache"], allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    c = float(spec["c"])
    whole = values.get("mu", values.get("whole_mu"))
    proxy = values.get("proxy_ball")
    if proxy is None and values.get("proxy_tangent") is not None and spec.get("proxy_mapping"):
        proxy = expmap0(values["proxy_tangent"], c, **spec["proxy_mapping"])
    if proxy is None:
        return
    ids = values["sample_ids"].reshape(-1)
    lookup = {value: i for i, value in enumerate(ids.tolist())}
    tops = np.asarray(direction["topk_sample_ids"])
    rows = np.asarray([[lookup[value] for value in row] for row in tops.tolist()], dtype=int)
    proxy_ids = values.get("proxy_ids", np.asarray(spec.get("proxy_ids", []))).reshape(-1)
    declared_ids = summary.get("retrieval", {}).get("proxy_ids", [])
    if len(proxy_ids) != len(proxy) or set(proxy_ids.tolist()) != set(declared_ids):
        return
    proxy_order = {value: i for i, value in enumerate(proxy_ids.tolist())}
    proxy = proxy[[proxy_order[value] for value in declared_ids]]
    ug, vg = ball_geometry(proxy, c), ball_geometry(whole, c)
    if not ug["direction_defined"].all() or not vg["direction_defined"].all():
        return
    cosine = np.clip(np.einsum("pd,pkd->pk", ug["direction"], vg["direction"][rows]), -1., 1.)
    q = float(spec["equal_radius_fraction"])
    distances = 2 / np.sqrt(c) * np.arcsinh(np.sqrt(2 * q * q * (1 - cosine) / (1 - q * q) ** 2))
    control = copy.deepcopy(direction)
    control["available"] = True
    control["nearest_distance_quantiles"] = quantiles(distances)
    control["nearest_distance_statistics"] = {"mean": float(np.mean(distances)), "std": float(np.std(distances)), "count": int(distances.size)}
    control["control"] = {"equal_normalized_radius": q,
                          "ranking_identity": "Equal-radius hyperbolic distance is strictly monotone in 1-cos(theta); audited deterministic direction topk IDs reused",
                          "selected_distances": "Exact general-c values recomputed from actual original directions",
                          "counterfactual_only": True, "production_changed": False}
    # Equal-radius query and endpoint depths are identical by construction.
    control["proxy_neighbour_depth"] = {
        "proxy_shallower_than_all_topk_fraction": 0.,
        "proxy_shallower_than_any_topk_fraction": 0.,
        "neighbour_mean_depth_minus_proxy_depth_quantiles": quantiles(np.zeros(len(proxy))),
        "note": "Equal-radius counterfactual; depths are identical by construction, not trained ancestor depths"}
    if "proxy_used_mask" in arrays and len(arrays["proxy_used_mask"]) == len(proxy):
        for name, mask in (("used", arrays["proxy_used_mask"]), ("inactive", ~arrays["proxy_used_mask"])):
            if name in control.get("groups", {}):
                control["groups"][name]["nearest_distance_quantiles"] = quantiles(distances[mask])
    summary["retrieval"]["equal_radius_hyperbolic"] = control
    arrays["equal_radius_hyperbolic_topk_ids"] = tops
    if "direction_slot_counts" in arrays:
        arrays["equal_radius_hyperbolic_slot_counts"] = arrays["direction_slot_counts"].copy()
    if "direction_proxy_topk_purity" in arrays:
        arrays["equal_radius_hyperbolic_proxy_topk_purity"] = arrays["direction_proxy_topk_purity"].copy()


def _reuse_snapshot(summary, arrays_path, output, spec=None):
    import numpy as np
    from .geometry import pool_hash
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("Existing audited snapshot arrays contain nonfinite values")
    if not {"sample_ids", "labels", "whole_depth"}.issubset(arrays):
        raise ValueError("Existing snapshot arrays cannot verify actual objects/labels/radial observations")
    if pool_hash(arrays["sample_ids"]) != summary.get("source", {}).get("sample_pool_sha256"):
        raise ValueError("Existing snapshot arrays disagree with the audited object-pool identity")
    result = copy.deepcopy(summary)
    if spec:
        _equal_radius(result, arrays, spec)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", result["run_key"])[:70]
    name = stem + "_e" + str(result["epoch"]) + "_" + _sha(arrays_path)[:8] + "_arrays.npz"
    target = output / name
    if target.exists():
        raise ValueError("Snapshot array filename collision")
    np.savez_compressed(target, **arrays)
    result["arrays_file"] = name
    result["reuse"] = {"source_arrays_file": Path(arrays_path).name,
                      "source_arrays_sha256": _sha(arrays_path), "actual_arrays_loaded": True,
                      "source_report_measures_reused": True}
    return result


def _epoch_csv(path, analyses):
    fields = sorted({key for analysis in analyses for row in analysis.get("epochs", []) for key in row.get("metrics", {})})
    columns = ["run_key", "display_name", "epoch", "model_updates", "logged_model_updates", "proxy_updates", "phase"] + fields
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for analysis in analyses:
            for row in analysis.get("epochs", []):
                writer.writerow({"run_key": run_key(analysis), "display_name": analysis.get("display_name"),
                                 **{key: row.get(key) for key in columns[2:7]}, **row.get("metrics", {})})


def run_system_audit(artifact_dirs, output_dir, *, pointcloud_pool=None, snapshot_spec_files=None,
                     mechanism_epochs=None, structure_epochs=None, do_mechanisms=True, do_structure=True,
                     mechanism_options=None, structure_options=None, max_mechanism_snapshots=16,
                     max_structure_snapshots=20, make_plots=True, render=True, max_seconds=5400,
                     do_redundancy=True, redundancy_options=None):
    """Combine existing CLI exports into a new CPU system audit.

    Actual CLI filenames are used: normalized_runs.json,
    feature_exports/snapshot_spec.json, snapshots/snapshot_summary.json.
    Already audited arrays are genuinely loaded and rehomed, never flattened or
    pointed at a different source directory. Explicit specs supplement exports.
    """
    started = time.monotonic()
    artifacts = [Path(value).resolve() for value in artifact_dirs]
    out = Path(output_dir).resolve()
    if not artifacts or any(not path.is_dir() for path in artifacts):
        raise ValueError("Every artifact source must be an existing CLI audit directory")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("Artifact directories must be distinct")
    if out.exists():
        raise FileExistsError("System audit requires a fresh output directory")
    if any(out == path or path in out.parents or out in path.parents for path in artifacts):
        raise ValueError("Combined output and immutable source artifact trees must be disjoint")
    if max_seconds <= 0 or max_mechanism_snapshots < 1 or max_structure_snapshots < 1:
        raise ValueError("Invalid CPU audit budget")
    analyses, specs, reused, transitions, aliases, source_files, warnings, source_statuses = [], [], [], [], {}, set(), [], []
    artifact_analyses = []
    for artifact in artifacts:
        normalized = artifact / "normalized_runs.json"
        if not normalized.is_file():
            raise ValueError("CLI artifact lacks normalized_runs.json")
        source_files.add(normalized)
        rows = _read(normalized)
        if not isinstance(rows, list):
            raise ValueError("normalized_runs.json must be a list")
        local = [_qualified(copy.deepcopy(row), artifact) for row in rows]
        artifact_analyses.append(local)
        analyses.extend(local)
        manifest_path = artifact / "audit_manifest.json"
        if manifest_path.is_file():
            source_files.add(manifest_path)
            status = _read(manifest_path).get("status")
            if status in ("running", "failed"):
                raise ValueError("Cannot combine an actively changing or failed source audit artifact")
            source_statuses.append(status)
        spec_file = artifact / "feature_exports" / "snapshot_spec.json"
        if spec_file.is_file():
            source_files.add(spec_file)
            specs.extend(_prepare_spec(s, local, warnings) for s in _specs(spec_file))
        export_manifest = artifact / "feature_exports" / "export_manifest.json"
        if export_manifest.is_file():
            source_files.add(export_manifest)
            values = _read(export_manifest)
            if values.get("status") in ("running", "failed"):
                raise ValueError("Source clean export is still changing or failed")
            if len(local) == 1:
                aliases[run_key(local[0])] = values.get("resolved_checkpoint_aliases", {})
        summary_file = artifact / "snapshots" / "snapshot_summary.json"
        if summary_file.is_file():
            source_files.add(summary_file)
            summary = _read(summary_file)
            for row in summary.get("snapshots", []):
                owner = _owner(row, local)
                row = copy.deepcopy(row); row["run_key"] = run_key(owner)
                array_path = summary_file.parent / row["arrays_file"]
                if not array_path.is_file():
                    raise ValueError("Saved snapshot summary references a missing audited arrays file")
                source_files.add(array_path)
                reused.append((row, array_path))
            for transition in summary.get("transitions", []):
                owner = _owner(transition, local)
                transition = copy.deepcopy(transition); transition["run_key"] = run_key(owner)
                transitions.append(transition)
    if len({run_key(a) for a in analyses}) != len(analyses):
        raise ValueError("Repeated source run key: supply each underlying run only once")
    for source in snapshot_spec_files or []:
        source = Path(source).resolve(); source_files.add(source)
        specs.extend(_prepare_spec(s, analyses, warnings) for s in _specs(source))
    unique = {}
    for spec in specs:
        key = (spec["run_key"], spec["epoch"])
        if key in unique and Path(unique[key]["cache"]).resolve() != Path(spec["cache"]).resolve():
            raise ValueError("Two different caches claim the same source run and checkpoint epoch")
        unique[key] = spec
    specs = list(unique.values())
    for spec in specs:
        for field in ("cache", "checkpoint", "proxy_cache", "optimizer_checkpoint"):
            if spec.get(field):
                source = Path(spec[field]).resolve()
                if not source.is_file():
                    raise ValueError("Explicit snapshot/optimizer source file is missing")
                if out == source or out in source.parents:
                    raise ValueError("Source files cannot reside inside the fresh combined output")
                source_files.add(source)
    if pointcloud_pool is None:
        candidates = [path / "feature_exports" / "candidate_clouds.npz" for path in artifacts]
        pointcloud_pool = next((path for path in candidates if path.is_file()), None)
    pointcloud_pool = Path(pointcloud_pool).resolve() if pointcloud_pool else None
    if pointcloud_pool:
        if not pointcloud_pool.is_file() or out == pointcloud_pool or out in pointcloud_pool.parents:
            raise ValueError("Point-cloud source must exist outside the fresh output tree")
        source_files.add(pointcloud_pool)
    source_hashes = {path: _sha(path) for path in source_files}
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "hier-postrun-system-v1", "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "read_only_sources": True, "optimizer_updates": 0, "GPU_forwards": 0, "new_test_forwards": 0,
                "run_keys": [run_key(a) for a in analyses], "source_audit_statuses": source_statuses,
                "resolved_checkpoint_aliases": aliases,
                "checkpoint_alias_provenance": "feature_exports/export_manifest.json: actual exported checkpoint alias verification; missing aliases fall back only to saved identity/manifest epoch, never log/cache maxima",
                "source_identities": [{"file": path.name, "sha256": value} for path, value in source_hashes.items()],
                "budgets": {"max_seconds": max_seconds, "max_mechanism_snapshots": max_mechanism_snapshots,
                            "max_structure_snapshots": max_structure_snapshots},
                "warnings": warnings, "capabilities": {}}
    try:
        manifest["audit_code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["audit_code_commit"] = None
    _write(out / "system_manifest.json", manifest)
    try:
        _write(out / "normalized_runs.json", analyses)
        _write(out / "combined_snapshot_spec.json", {"scope": "Server-local working spec: paths are private, do not publish", "snapshots": specs})
        _epoch_csv(out / "epochs.csv", analyses)
        reuse_lookup = {(row["run_key"], row["epoch"]): (row, path) for row, path in reused}
        needs = [spec for spec in specs if (spec["run_key"], spec["epoch"]) not in reuse_lookup
                 or not _consistent_reuse(reuse_lookup[(spec["run_key"], spec["epoch"])][0], spec)]
        snapshot_dir = out / "snapshots"
        if needs:
            from .snapshots import analyze_snapshots
            snapshots = analyze_snapshots(needs, snapshot_dir)
        else:
            snapshot_dir.mkdir()
            snapshots = {"schema_version": "1", "snapshots": [], "transitions": [], "availability": [],
                         "interpretation": ["Existing audited arrays were actually loaded and rehomed; source objects/labels retained."],
                         "arrays_base_dir": str(snapshot_dir)}
        recalculated = {(row["run_key"], row["epoch"]) for row in snapshots["snapshots"]}
        requested = {(spec["run_key"], spec["epoch"]): spec for spec in specs}
        for row, arrays in reused:
            key = (row["run_key"], row["epoch"])
            if key in recalculated:
                continue
            if key in requested and not _consistent_reuse(row, requested[key]):
                continue
            snapshots["snapshots"].append(_reuse_snapshot(row, arrays, snapshot_dir, requested.get(key)))
            snapshots["availability"].append({"label": row.get("label"), "run_key": row["run_key"],
                                              "epoch": row["epoch"], "available": True, "reused_actual_arrays": True})
        snapshots["transitions"].extend(t for t in transitions
                                         if (t["run_key"], t["from_epoch"]) not in recalculated
                                         and (t["run_key"], t["to_epoch"]) not in recalculated)
        snapshots["snapshots"].sort(key=lambda row: (row["run_key"], row["epoch"]))
        if pointcloud_pool: snapshots["pointcloud_pool"] = str(pointcloud_pool)
        _write(snapshot_dir / "snapshot_summary.json", {key: value for key, value in snapshots.items()
                                                       if key not in ("arrays_base_dir", "pointcloud_pool")})
        lifetime = analyze_longitudinal(analyses, out, make_plots=make_plots)
        mechanisms, structure = None, None
        extension_figures = list(lifetime["figures"])
        remaining = max_seconds - (time.monotonic() - started)
        if do_mechanisms:
            selected, availability = _selection(specs, analyses, aliases, mechanism_epochs, max_mechanism_snapshots)
            import numpy as np
            active = []
            for spec in selected:
                owner = next(a for a in analyses if run_key(a) == spec["run_key"])
                if owner.get("identity", {}).get("config", {}).get("proxy_optimizer") is False:
                    continue
                with np.load(spec["cache"], allow_pickle=False) as archive:
                    if "proxy_tangent" in archive:
                        active.append(spec)
            manifest["capabilities"]["mechanisms"] = {"selected": len(active), "availability": availability}
            if active and remaining > 0:
                from .mechanisms import analyze_mechanisms
                options = {"query_count": 64, "noise_repeats": 8, "gradient_repeats": 1, "max_seconds": 1800}
                options.update(mechanism_options or {}); options["max_seconds"] = min(options["max_seconds"], remaining)
                mechanisms = analyze_mechanisms(active, out / "mechanisms", **options)
                if make_plots:
                    from .mechanisms_figures import save_mechanism_figures
                    rendered = save_mechanism_figures(mechanisms, out)
                    extension_figures.extend(rendered["figures"]); warnings.extend(rendered.get("warnings", []))
            else:
                warnings.append("Fixed-query mechanisms unavailable: no declared trainable proxy caches or system CPU budget exhausted.")
        remaining = max_seconds - (time.monotonic() - started)
        if do_structure:
            selected, availability = _selection(specs, analyses, aliases, structure_epochs, max_structure_snapshots)
            manifest["capabilities"]["structure"] = {"selected": len(selected), "availability": availability}
            if selected and pointcloud_pool and remaining > 0:
                from .structure import analyze_structure, save_structure_figures
                options = {"shape_points": 64, "per_class": 8, "bootstrap": 200, "max_seconds": 1800,
                           "max_objects": 400, "max_proxies": 64}
                options.update(structure_options or {}); options["max_seconds"] = min(options["max_seconds"], remaining)
                structure = analyze_structure(selected, pointcloud_pool, out / "structure", snapshots=snapshots, **options)
                if make_plots:
                    rendered = save_structure_figures(structure, out)
                    extension_figures.extend(rendered["figures"]); warnings.extend(rendered.get("warnings", []))
            else:
                warnings.append("Independent shape audit unavailable: explicit cache/cloud sources missing or system CPU budget exhausted.")
        redundancy = None
        remaining = max_seconds - (time.monotonic() - started)
        if do_redundancy:
            manifest["capabilities"]["redundancy"] = {"selected": len(specs)}
            if specs and remaining > 0:
                from .redundancy import analyze_redundancy, save_redundancy_figures
                options = {"chunk_size": 32, "max_seconds": 600}
                options.update(redundancy_options or {}); options["max_seconds"] = min(options["max_seconds"], remaining)
                redundancy = analyze_redundancy(specs, out / "redundancy", **options)
                if make_plots:
                    rendered = save_redundancy_figures(redundancy, out)
                    extension_figures.extend(rendered["figures"]); warnings.extend(rendered.get("warnings", []))
            else:
                warnings.append("Proxy redundancy/boundary audit unavailable: explicit cache specs missing or CPU budget exhausted.")
        comparison = build_comparison(analyses, snapshots, mechanisms, structure, out, make_plots=make_plots)
        extension_figures.extend(comparison["figures"])
        extensions = {"longitudinal": lifetime, "comparison": comparison,
                      "mechanisms": mechanisms, "structure": structure, "redundancy": redundancy, "figures": extension_figures}
        if render and make_plots:
            from .report import render_report
            report = render_report(analyses, snapshots, out, extensions=extensions)
            warnings.extend(report.get("warnings", []))
        else:
            report = None
        partial = any(value in ("partial", "partial_budget") for value in source_statuses)
        partial |= any(not row.get("available") for row in snapshots.get("availability", []))
        partial |= do_mechanisms and mechanisms is None or do_structure and structure is None or do_redundancy and redundancy is None
        for name in ("mechanisms", "structure", "redundancy"):
            module = locals()[name]
            if module and module.get("status") in ("partial_budget", "partial", "failed"): partial = True
            if module: warnings.extend(module.get("warnings", []))
        manifest.update(status="partial" if partial else "completed", report=report,
                        snapshots=len(snapshots["snapshots"]), source_files_unchanged=all(_sha(path) == expected for path, expected in source_hashes.items()),
                        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started)
        if not manifest["source_files_unchanged"]:
            raise RuntimeError("An immutable source artifact changed during system audit")
        _write(out / "system_manifest.json", manifest)
        return {"manifest": manifest, "snapshots": snapshots, "extensions": extensions, "report": report,
                "output_dir": str(out)}
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__,
                        elapsed_seconds=time.monotonic() - started,
                        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        _write(out / "system_manifest.json", manifest)
        raise
