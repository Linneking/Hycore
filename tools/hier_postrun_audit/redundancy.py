"""CPU proxy redundancy: nearest partners and explicitly labelled tolerances.

Near-coincident coordinates, small geodesic separation and similar directions
have different meanings/units. No threshold here defines a normal hierarchy.
The pairwise computation retains at most chunk_size x P distances, never P x P x D.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .geometry import align_rows, ball_geometry, expmap0, matrix, poincare_distance, pool_hash, quantiles, unique_ids


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _load(spec):
    if not spec.get("run_key") or spec.get("epoch") is None:
        raise ValueError("Redundancy requires actual epoch and run_key identities")
    arrays, metadata, source = {}, {}, {}
    path = Path(spec["cache"]).resolve() if spec.get("cache") else None
    if path:
        before = _hash(path)
        with np.load(path, allow_pickle=False) as archive:
            # Do not decompress the full candidate whole/logit/cloud pool.
            for key in ("proxy_ball", "proxy_tangent", "tangent_proxies", "proxy_ids", "c", "curvature", "metadata_json"):
                if key in archive:
                    arrays[key] = archive[key].copy()
        if _hash(path) != before:
            raise RuntimeError("Source cache changed while reading")
        source = {"cache_file": path.name, "cache_sha256": before}
        if "metadata_json" in arrays:
            metadata = json.loads(str(np.asarray(arrays["metadata_json"]).reshape(())))
    c = spec.get("c", arrays.get("c", arrays.get("curvature", metadata.get("c"))))
    if c is None:
        raise ValueError("Redundancy needs explicit curvature")
    c = float(c)
    tangent = spec.get("proxy_tangent", arrays.get("proxy_tangent", arrays.get("tangent_proxies")))
    coords = spec.get("proxy_coords", spec.get("explicit_proxy_coords", spec.get("proxy_ball", arrays.get("proxy_ball"))))
    mapping = spec.get("proxy_mapping", metadata.get("proxy_mapping"))
    if tangent is not None:
        tangent = matrix(tangent, "stored proxy tangent parameters").copy()
        if mapping is not None:
            expected = expmap0(tangent, c, **mapping)
            if coords is None:
                coords = expected
            elif not np.allclose(coords, expected, atol=float(spec.get("mapping_atol", 2e-6)), rtol=2e-6):
                raise ValueError("Declared proxy map disagrees with explicit coordinates")
        elif coords is None:
            raise ValueError("Stored tangent parameters require an explicit forward mapping")
    identity = {"run_key": str(spec["run_key"]), "epoch": int(spec["epoch"]),
                "display_name": spec.get("display_name", spec.get("label", spec["run_key"])),
                "version": spec.get("version"), "storage_run_id": spec.get("storage_run_id"),
                "source": source, "c": c}
    if coords is None:
        return {**identity, "available": False, "reason": "No proxy parameters/coordinates; baseline is not a bank of dead proxies"}, None, path
    coords = matrix(coords, "proxy ball coordinates").copy()
    geometry = ball_geometry(coords, c)
    declared_ids = spec.get("proxy_ids", arrays.get("proxy_ids"))
    stable = declared_ids is not None
    ids = unique_ids(np.arange(len(coords)) if declared_ids is None else declared_ids, "proxy_ids")
    if len(ids) != len(coords):
        raise ValueError("Proxy coordinates and IDs differ")
    order = np.argsort(ids, kind="stable")
    coords, ids = coords[order], ids[order]
    tangent = tangent[order] if tangent is not None else None
    if tangent is not None and len(tangent) != len(coords):
        raise ValueError("Stored tangent and coordinate rows differ")
    return {**identity, "available": True, "proxy_ids_stable": stable, "proxy_id_policy": spec.get("proxy_id_policy"),
            "proxy_id_pool_sha256": pool_hash(ids), "dimension": coords.shape[1], "proxy_count": len(ids),
            "proxy_mapping": mapping,
            "stored_tangent_norm": quantiles(np.linalg.norm(tangent, axis=1)) if tangent is not None else None}, (coords, ids), path


def _one(row, private, chunk_size, deadline, coordinate_tolerance, geodesic_tolerance, angular_tolerance):
    points, ids = private
    count, c = len(points), row["c"]
    geometry = ball_geometry(points, c)
    nearest_d, nearest_rows = np.full(count, np.nan), np.full(count, -1, np.int64)
    nearest_a, angular_rows = np.full(count, np.nan), np.full(count, -1, np.int64)
    pair_counts = {"coordinate_near_pair_count": 0, "geodesic_near_pair_count": 0, "direction_near_pair_count": 0}
    valid_direction = geometry["direction_defined"]
    for start in range(0, count, chunk_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("Proxy redundancy wall-time budget reached")
        stop = min(start + chunk_size, count)
        distances = poincare_distance(points[start:stop], points, c)
        all_rows = np.arange(start, stop)
        upper = np.arange(count)[None, :] > all_rows[:, None]
        # Invert the same stable asinh formula to recover coordinate separation
        # from the exact squared-difference accumulation, avoiding cancellation
        # in ||x||² + ||y||² - 2<x,y>.
        denominator = ((1 - geometry["normalized_radius"][start:stop] ** 2)[:, None] *
                       (1 - geometry["normalized_radius"] ** 2)[None, :])
        coordinate_distance = np.sinh(np.sqrt(c) * distances / 2.) * np.sqrt(denominator / c)
        pair_counts["coordinate_near_pair_count"] += int(np.sum(upper & (coordinate_distance <= coordinate_tolerance)))
        pair_counts["geodesic_near_pair_count"] += int(np.sum(upper & (distances <= geodesic_tolerance)))
        distances[np.arange(stop-start), all_rows] = np.inf
        if count > 1:
            partners = distances.argmin(axis=1)
            nearest_rows[start:stop] = partners
            nearest_d[start:stop] = distances[np.arange(stop-start), partners]
        cosine = np.clip(geometry["direction"][start:stop] @ geometry["direction"].T, -1., 1.)
        angles = np.arccos(cosine)
        direction_pair_valid = valid_direction[start:stop, None] & valid_direction[None, :]
        pair_counts["direction_near_pair_count"] += int(np.sum(upper & direction_pair_valid & (angles <= angular_tolerance)))
        angles[~direction_pair_valid] = np.inf
        angles[np.arange(stop-start), all_rows] = np.inf
        if count > 1:
            partners = angles.argmin(axis=1)
            best = angles[np.arange(stop-start), partners]
            available = np.isfinite(best)
            angular_rows[start:stop][available] = partners[available]
            nearest_a[start:stop][available] = best[available]
    partners = [None if index < 0 else _scalar(ids[index]) for index in nearest_rows]
    angular_partners = [None if index < 0 else _scalar(ids[index]) for index in angular_rows]
    row.update(
        proxy_ids=ids.tolist(), proxy_depth=geometry["depth"].tolist(),
        depth_quantiles=quantiles(geometry["depth"]), radius_quantiles=quantiles(geometry["radius"]),
        normalized_radius_quantiles=quantiles(geometry["normalized_radius"]),
        zero_direction_count=int((~valid_direction).sum()),
        nearest_geodesic={"available_proxy_count": int(np.isfinite(nearest_d).sum()),
                          "distance_quantiles": quantiles(nearest_d[np.isfinite(nearest_d)]),
                          "partner_ids": partners, "distances": [float(value) if np.isfinite(value) else None for value in nearest_d]},
        nearest_direction={"available_proxy_count": int(np.isfinite(nearest_a).sum()),
                           "angle_rad_quantiles": quantiles(nearest_a[np.isfinite(nearest_a)]),
                           "partner_ids": angular_partners,
                           "angles_rad": [float(value) if np.isfinite(value) else None for value in nearest_a],
                           "reason_if_missing": "Origins have no direction; a single valid direction has no eligible nonself partner"},
        near_pairs={**pair_counts, "total_unordered_pairs": count * (count - 1) // 2,
                    "valid_direction_unordered_pairs": int(valid_direction.sum() * (valid_direction.sum() - 1) // 2),
                    "tolerances": {"coordinate_distance": coordinate_tolerance, "geodesic_distance": geodesic_tolerance,
                                   "angular_radians": angular_tolerance},
                    "meaning": "Explicit numerical closeness descriptions only; no normality/semantic hierarchy threshold"})
    return row


def _transition(first, second):
    row = {"run_key": second["run_key"], "from_epoch": first["epoch"], "to_epoch": second["epoch"], "available": False}
    if not first.get("available") or not second.get("available"):
        row["reason"] = "Mapped proxies unavailable"
        return row
    if not first["proxy_ids_stable"] or not second["proxy_ids_stable"]:
        row["reason"] = "Explicit stable proxy identities missing"
        return row
    if first["c"] != second["c"] or first["dimension"] != second["dimension"]:
        row["reason"] = "Curvature or embedding dimension changed"
        return row
    if first["proxy_id_pool_sha256"] != second["proxy_id_pool_sha256"]:
        row["reason"] = "Proxy identity pools differ"
        return row
    alignment = align_rows(np.asarray(first["proxy_ids"]), np.asarray(second["proxy_ids"]))
    row["available"] = True
    for key in ("nearest_geodesic", "nearest_direction"):
        old = first[key]["partner_ids"]
        new = [second[key]["partner_ids"][i] for i in alignment]
        valid = [(a, b) for a, b in zip(old, new) if a is not None and b is not None]
        row[key + "_partner_retention"] = {"proxy_count": len(valid),
            "fraction": float(np.mean([a == b for a, b in valid])) if valid else None}
    row["note"] = "Same saved proxy IDs within one run across actual checkpoints; nearest partners are not semantic ancestors."
    return row


def analyze_redundancy(specs, output_dir, chunk_size=32, max_seconds=600,
                       coordinate_tolerance=1e-8, geodesic_tolerance=1e-6, angular_tolerance=1e-6):
    """Read explicit cache/coordinates into a new bounded, CPU-only audit."""
    if not specs or not 1 <= int(chunk_size) <= 128 or max_seconds <= 0:
        raise ValueError("Invalid redundancy snapshot/chunk/time budget")
    for value in (coordinate_tolerance, geodesic_tolerance, angular_tolerance):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Duplicate tolerances must be finite and nonnegative")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("Redundancy output must be a new directory")
    paths = [Path(spec["cache"]).resolve() for spec in specs if spec.get("cache")]
    if any(path.parent == out or path.parent in out.parents for path in paths):
        raise ValueError("Redundancy output must be outside source cache directories")
    hashes = {path: _hash(path) for path in paths}
    out.mkdir(parents=True)
    start, rows, transitions = time.monotonic(), [], []
    summary = {"format": "hier-proxy-redundancy-v1", "status": "running",
               "config": {"chunk_size": int(chunk_size), "max_seconds": float(max_seconds),
                          "coordinate_tolerance": coordinate_tolerance,
                          "geodesic_tolerance": geodesic_tolerance, "angular_tolerance_rad": angular_tolerance},
               "snapshots": rows, "transitions": transitions, "warnings": [],
               "scope": "Proxy geometry and nearest nonself partners; no sample/proxy activation, optimizer updates or semantic hierarchy claim",
               "units": {"coordinate_distance": "Native Euclidean ball-coordinate units",
                         "geodesic_distance": "Full unsquared hyperbolic distance, same units as margin",
                         "angular_radians": "Angle between directions, excluding origins"}}
    def save():
        (out / "redundancy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        deadline = start + max_seconds
        for spec in sorted(specs, key=lambda item: (str(item.get("run_key")), int(item.get("epoch", -1)))):
            if time.monotonic() >= deadline:
                raise TimeoutError("Proxy redundancy budget reached before next snapshot")
            row, private, _ = _load(spec)
            if private is not None:
                row = _one(row, private, int(chunk_size), deadline, coordinate_tolerance, geodesic_tolerance, angular_tolerance)
            previous = next((old for old in reversed(rows) if old["run_key"] == row["run_key"]), None)
            if previous:
                transitions.append(_transition(previous, row))
            rows.append(row)
            save()
        summary["status"] = "completed"
    except TimeoutError as exc:
        summary["status"] = "partial_budget"
        summary["warnings"].append(str(exc))
    except Exception as exc:
        summary["status"], summary["error"] = "failed", type(exc).__name__ + ": " + str(exc)
        save()
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        summary["sources_unchanged"] = all(_hash(path) == digest for path, digest in hashes.items())
        if not summary["sources_unchanged"]:
            summary["status"], summary["error"] = "failed", "Source cache changed during redundancy audit"
        save()
    if not summary["sources_unchanged"]:
        raise RuntimeError(summary["error"])
    return summary


def save_redundancy_figures(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _slug
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures, groups = [], {}
    for row in summary.get("snapshots", []):
        if row.get("available"):
            groups.setdefault(row["run_key"], []).append(row)
    for run, rows in groups.items():
        rows.sort(key=lambda row: row["epoch"])
        epochs, name = [row["epoch"] for row in rows], rows[0].get("display_name", run)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        def plot(axis, values, label, color):
            y = [float(value) if value is not None else float("nan") for value in values]
            if any(np.isfinite(value) for value in y):
                axis.plot(epochs, y, "o-", label=label, color=color)
        plot(axes[0, 0], [row["nearest_geodesic"]["distance_quantiles"]["p50"] for row in rows], "median", "#2563eb")
        plot(axes[0, 0], [row["nearest_geodesic"]["distance_quantiles"]["p05"] for row in rows], "p05", "#d97706")
        plot(axes[0, 1], [None if row["nearest_direction"]["angle_rad_quantiles"]["p50"] is None else np.degrees(row["nearest_direction"]["angle_rad_quantiles"]["p50"]) for row in rows], "median", "#2563eb")
        plot(axes[0, 1], [None if row["nearest_direction"]["angle_rad_quantiles"]["p05"] is None else np.degrees(row["nearest_direction"]["angle_rad_quantiles"]["p05"]) for row in rows], "p05", "#d97706")
        transitions = [row for row in summary.get("transitions", []) if row["run_key"] == run]
        for key, color in (("nearest_geodesic", "#2563eb"), ("nearest_direction", "#d97706")):
            values = [row.get(key + "_partner_retention", {}).get("fraction") if row.get("available") else None for row in transitions]
            if any(value is not None for value in values):
                axes[1, 0].plot([row["to_epoch"] for row in transitions],
                               [float(value) if value is not None else float("nan") for value in values], "o-", color=color, label=key)
        for key, label, color in (("coordinate_near_pair_count", "coordinate closeness", "#2563eb"),
                                  ("geodesic_near_pair_count", "geodesic closeness", "#d97706"),
                                  ("direction_near_pair_count", "direction closeness", "#16a34a")):
            plot(axes[1, 1], [row["near_pairs"][key] for row in rows], label, color)
        axes[0, 0].set(title="Nearest nonself geodesic distance", ylabel="Full hyperbolic distance")
        axes[0, 1].set(title="Nearest nonself direction angle", ylabel="Degrees; origins excluded")
        axes[1, 0].set(title="Nearest partner ID retention", ylabel="Same-ID fraction", ylim=(0, 1.03))
        axes[1, 1].set(title="Numerically close unordered pairs", ylabel="Pair count")
        for axis in axes.flat:
            axis.set_xlabel("Actual saved checkpoint epoch")
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=8)
        cfg = summary["config"]
        fig.suptitle(f"{name}: proxy redundancy descriptions\nTolerances: coordinate {cfg['coordinate_tolerance']:g}, geodesic {cfg['geodesic_tolerance']:g}, angle {cfg['angular_tolerance_rad']:g} rad; no normality threshold", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .91))
        stem = _slug(run) + "_proxy_redundancy"
        png, svg = out / (stem + ".png"), out / (stem + ".svg")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        figures.append({"run_id": run, "kind": "proxy_redundancy", "title": "Proxy nearest partners and numerical closeness",
                        "note": "Full geodesic and angle nearest partners exclude self, with deterministic true-ID ties. Origins have no direction. Explicit coordinate/geodesic/angular tolerances describe numerical closeness only; nearest partners are not HIER ancestors or semantic redundancy labels.",
                        "png": "figures/" + png.name, "svg": "figures/" + svg.name})
    return {"figures": figures, "warnings": summary.get("warnings", [])}
