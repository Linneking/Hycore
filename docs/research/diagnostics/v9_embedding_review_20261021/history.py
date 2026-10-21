"""Read-only CPU history extraction from existing scalar logs and fixed archives.

Usage: python history.py --spec PRIVATE_SPEC.json --out NEW_OUTPUT_DIRECTORY
Spec: {repo: ..., runs: [{key,run_dir,c_native,kind,label?,max_epoch?}]}.
No encoder construction, model forward, CUDA call, backward or optimizer update.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ARCHIVE_EPOCHS = (1, 5, 20, 40, 99, 100, 160, 200, 240, 300)
DOMAINS = ("all_draws", "noncollision", "active_noncollision")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signature(path):
    value = path.stat()
    return {"size": int(value.st_size), "mtime_ns": int(value.st_mtime_ns),
            "inode": int(value.st_ino)}


def read_json_unchanged(path):
    before = signature(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if signature(path) != before:
        raise RuntimeError("Source changed while reading: " + path.name)
    return value, before


def load_unchanged(path):
    import torch
    before = signature(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        value = torch.load(path, map_location="cpu", weights_only=False)
    if signature(path) != before:
        raise RuntimeError("Source archive changed while loading: " + path.name)
    return value, before


def epoch_of(value, source_name=None):
    if value.get("initialization_only") is True:
        if any(value.get(field) is not None and int(value[field]) != 0
               for field in ("completed_epochs", "epoch")):
            raise ValueError("Initialization-only checkpoint has a nonzero epoch")
        return 0
    if value.get("completed_epochs") is not None:
        return int(value["completed_epochs"])
    if value.get("epoch") is not None:
        return int(value["epoch"])
    # A filename alone is not evidence of initialization. Accept a format
    # explicitly identifying an initial state, never a generic model format.
    if source_name == "initialization_e0.pth":
        fmt = str(value.get("format", "")).lower()
        if any(marker in fmt for marker in ("initialization", "initial-state", "initial_state", "init-e0", "init_e0")):
            return 0
        raise ValueError("Named E0 archive lacks initialization_only or an initialization-specific format")
    return -1


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def depth(value, c):
    points = array(value)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite matrix coordinates")
    q = np.sqrt(c) * np.linalg.norm(points, axis=1)
    if np.any(q >= 1) or np.any(q < 0):
        raise ValueError("History coordinate outside declared geometry domain")
    return 2 / np.sqrt(c) * np.arctanh(q)


def usage_counts(structure, component, domain):
    """Accept both raw first-batch structure and pooled epoch structure."""
    report = structure.get(component, {}) if isinstance(structure, dict) else {}
    raw, pooled = report.get("selected_proxy_counts", {}), report.get("selected_ancestor_usage", {})
    parts = []
    for role in ("pair", "triple"):
        value = raw.get(domain + "/" + role)
        if value is None:
            value = pooled.get(domain + "/" + role, {}).get("counts")
        if value is None:
            return None
        value = np.asarray(value, dtype=np.int64).reshape(-1)
        if np.any(value < 0):
            raise ValueError("Negative selection counts")
        parts.append(value)
    if parts[0].shape != parts[1].shape:
        raise ValueError("Pair/triple proxy counts have different shape")
    return parts[0] + parts[1]


def usage_summary(counts):
    if counts is None:
        return None
    total = int(counts.sum())
    p = counts[counts > 0] / total if total else np.empty(0)
    return {"used": int(np.count_nonzero(counts)), "selections": total,
            "effective": float(np.exp(-(p * np.log(p)).sum())) if total else 0.,
            "maximum_share": float(counts.max() / total) if total else None}


def usages(structure, include_counts=False):
    result = {}
    for domain in DOMAINS:
        selected = {component: usage_counts(structure, component, domain)
                    for component in ("sample", "proxy")}
        selected["combined"] = (selected["sample"] + selected["proxy"]
                                if all(value is not None for value in selected.values()) else None)
        result[domain] = {}
        for component, counts in selected.items():
            summary = usage_summary(counts)
            if summary is not None and include_counts:
                summary["counts"] = counts.tolist()
            result[domain][component] = summary
    return result


def moments(value):
    if not isinstance(value, dict):
        return None
    result = {k: value[k] for k in ("count", "mean", "std", "min", "p10", "median", "p90", "max",
                                   "batch_size_weighted_mean", "at_forward_depth_cap_fraction")
              if isinstance(value.get(k), (int, float)) or value.get(k) is None and k in value}
    return result or None


def scalar_summaries(telemetry):
    result = {}
    for key, value in telemetry.get("scalars", {}).items():
        if key.startswith("proxy_") or key.startswith("hier_whole_depth") or key in ("whole_depth", "part_depth"):
            if isinstance(value, dict):
                result[key] = moments(value)
            elif isinstance(value, (int, float)):
                result[key] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    repo = Path(spec["repo"])
    sys.path.insert(0, str(repo))
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
    training, snapshots, proxies, inventory = [], [], [], []
    for run in spec["runs"]:
        run_dir = Path(run["run_dir"])
        c_native, c_hier = float(run["c_native"]), .1
        maximum = int(run.get("max_epoch", 10 ** 9))
        base = {k: run.get(k) for k in ("key", "label", "kind")}
        base.update(c_native=c_native, c_hier=c_hier)
        metrics = {}
        for path in sorted(run_dir.glob("metrics_epoch_*.json")):
            saved, proof = read_json_unchanged(path)
            epoch = epoch_of(saved, path.name)
            if epoch < 0 or epoch > maximum:
                continue
            metrics[epoch] = saved
            telemetry = saved.get("telemetry", {})
            geometry = telemetry.get("geometry", {})
            point_moments = {}
            for role in ("whole", "part"):
                entry = geometry.get(role, {})
                point_moments[role] = {"depth": moments(entry.get("depth")),
                                      "at_cap_fraction": entry.get("at_forward_depth_cap_fraction"),
                                      "positions": entry.get("positions")}
            report = {**base, "epoch": epoch, "scope": "actual_train_epoch_pooled_augmented_positions",
                      "geometry": point_moments, "usage": usages(saved.get("structure", {})),
                      "scalars": scalar_summaries(telemetry), "source_archive": path.name,
                      "source_stat": proof}
            batch_geometry = geometry.get("hier_reported_per_batch") or {}
            report["hier_reported_batch_scalar_moments"] = {key: moments(value) for key, value in batch_geometry.items()
                                                          if any(term in key for term in ("depth", "radius", "cap"))}
            training.append(report)
        snapshot_count, archive_count = 0, 0
        for path in sorted(run_dir.glob("actual_training_epoch_*.pth")):
            # Path epoch is available before any potentially large loading.
            path_epoch = int(path.stem.rsplit("_", 1)[1])
            if path_epoch > maximum:
                continue
            saved, proof = load_unchanged(path)
            epoch = epoch_of(saved, path.name)
            if saved.get("stage") != "actual_training_pre_optimizer" or int(saved.get("step", -1)) != 1:
                raise ValueError("Unexpected snapshot stage or step")
            if not math.isclose(float(saved.get("native_c", c_native)), c_native, rel_tol=1e-6):
                raise ValueError("Snapshot native curvature disagrees with run spec")
            report = {**base, "epoch": epoch, "step": 1,
                      "scope": "actual_train_pre_optimizer_first_batch",
                      "population_note": "global64 sampled positions; not full unique-object pool",
                      "source_archive": path.name, "source_stat": proof,
                      "whole_count": saved.get("whole_count"), "child_count": saved.get("child_count"),
                      "usage": usages(saved.get("structure", {}), include_counts=True)}
            for field, c in (("native_mu", c_native), ("native_nu", c_native), ("hier_mu", c_hier)):
                report[field + "_depth"] = depth(saved[field], c).tolist() if field in saved else None
            report["positions"] = len(saved["native_mu"]) if "native_mu" in saved else None
            report["proxy_depth"] = depth(saved["proxy_ball"], c_hier).tolist() if "proxy_ball" in saved else None
            report["proxy_tangent_norm"] = np.linalg.norm(array(saved["proxy_tangent"]), axis=1).tolist() if "proxy_tangent" in saved else None
            if signature(path) != proof:
                raise RuntimeError("Source snapshot changed during extraction")
            snapshots.append(report); snapshot_count += 1
            del saved
        choices = [("initialization_e0.pth", 0, "initialization")]
        choices.extend((f"checkpoint_epoch_{epoch:03d}.pth", epoch, "archive") for epoch in ARCHIVE_EPOCHS if epoch <= maximum)
        choices.append(("best.pth", None, "best"))
        absent = []
        for filename, expected, stage in choices:
            path = run_dir / filename
            if not path.exists():
                absent.append(filename)
                continue
            saved, proof = load_unchanged(path)
            epoch = epoch_of(saved, path.name)
            if expected is not None and epoch != expected:
                raise ValueError("Fixed checkpoint epoch mismatch: " + filename)
            if epoch > maximum:
                del saved
                continue
            mapping = saved.get("proxy")
            if not isinstance(mapping, dict) or not mapping:
                del saved
                continue
            tangent_values = [value for key, value in mapping.items() if key.split(".")[-1] == "tangent_proxies"]
            if len(tangent_values) != 1:
                raise ValueError("Missing or ambiguous hierarchy tangent proxy state")
            tangent = tangent_values[0].detach().cpu().float()
            if tangent.ndim != 2 or tangent.shape[0] != 512:
                raise ValueError("Expected 512 hierarchy proxies")
            with torch.no_grad():
                ball = to_ball(tangent, c=c_hier, clip_r=2.3, hook=False)
            proxy_depth = depth(ball, c_hier)
            tangent_norm = np.linalg.norm(array(tangent), axis=1)
            report = {**base, "epoch": epoch, "stage": stage,
                      "scope": "fixed_checkpoint_proxy_forward_cpu_official_mapping",
                      "source_archive": filename, "source_stat": proof,
                      "proxy_depth": proxy_depth.tolist(), "proxy_tangent_norm": tangent_norm.tolist(),
                      "proxy_cap_fraction": float(np.mean(np.abs(proxy_depth - 4.6) <= 1e-3)),
                      "proxy_tangent_over_cap_count": int(np.count_nonzero(tangent_norm > 2.3 + 1e-6)),
                      "usage_source_epoch": epoch, "usage": usages(metrics.get(epoch, {}).get("structure", {}), include_counts=True)}
            if signature(path) != proof:
                raise RuntimeError("Source fixed checkpoint changed during extraction")
            proxies.append(report); archive_count += 1
            del saved, mapping, tangent_values, tangent, ball
        inventory.append({**base, "training_rows": len(metrics), "snapshots": snapshot_count,
                          "proxy_checkpoint_rows": archive_count, "max_epoch": run.get("max_epoch"),
                          "missing_named_archives": absent})
        print(json.dumps({"key": run["key"], "metrics": len(metrics), "snapshots": snapshot_count,
                          "proxy_checkpoints": archive_count}), flush=True)
    common = {"format": "v9-cpu-readonly-training-history-v1", "source_files_unchanged": True,
              "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "optimizer_updates": 0, "network_forward_calls": 0, "CUDA_calls": 0,
              "sample_identifiers_or_embeddings_exported": False,
              "proxy_mapping": "official V8/HIER to_ball CPU FP32 c=.1 tangent cap2.3; no GS needed for value-only readout",
              "usage_definition": "pair+triple counts for sample, proxy, combined; all_draws/noncollision/active_noncollision; missing is null"}
    write(args.out / "training_rows.json", {**common, "rows": training})
    write(args.out / "longitudinal.json", {**common, "rows": snapshots})
    write(args.out / "proxy_checkpoint_history.json", {**common, "rows": proxies})
    write(args.out / "history_manifest.json", {**common, "runs": inventory})


if __name__ == "__main__":
    main()
