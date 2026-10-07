"""Bounded galleries of actual candidate point clouds, never embedding points."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .plots import _slug


def representative_proxy_rows(snapshot, depth=None, used_mask=None, maximum=4):
    """Choose a repeated raw tuple plus shallow/deep/used representatives."""
    raw = snapshot.get("retrieval", {}).get("hyperbolic", {}).get("topk_sample_ids", [])
    if not raw or maximum < 1:
        return []
    chosen = []
    def add(index, reason):
        if index is not None and 0 <= int(index) < len(raw) and int(index) not in [i for i, _ in chosen]:
            chosen.append((int(index), reason))
    groups = Counter(tuple(sorted(map(str, row))) for row in raw)
    hot = max(groups, key=groups.get)
    add(next(i for i, row in enumerate(raw) if tuple(sorted(map(str, row))) == hot), "most repeated raw top-k set")
    if depth is not None and len(depth) == len(raw):
        order = sorted(range(len(depth)), key=lambda i: float(depth[i]))
        add(order[0], "shallowest proxy")
        add(order[-1], "deepest proxy")
        active = [i for i in order if used_mask is not None and len(used_mask) == len(raw) and bool(used_mask[i])]
        add(active[len(active) // 2] if active else order[len(order) // 2], "median-depth selected proxy" if active else "median-depth proxy")
    for index in range(len(raw)):
        if len(chosen) >= maximum:
            break
        add(index, "additional proxy row")
    return chosen[:maximum]


def save_object_galleries(snapshots, output_dir, max_snapshots=2, max_proxies=4, max_points=1024):
    """Render at most two checkpoint galleries with fixed camera/common scale.

    A gallery consists of raw and direction top-k neighbours for one explicit
    proxy row. The figures show original 3D point-cloud coordinates. No point
    cloud is invented from an embedding; missing object IDs stay unavailable.
    """
    figures, warnings = [], []
    data = snapshots or {}
    rows = data.get("snapshots", [])
    if not rows:
        return {"figures": figures, "warnings": warnings}
    source = data.get("pointcloud_pool")
    if not source:
        return {"figures": [], "warnings": ["Object gallery unavailable: actual point-cloud pool was not supplied."]}
    source = Path(source)
    if not source.is_file():
        return {"figures": [], "warnings": ["Object gallery unavailable: point-cloud pool file is missing."]}
    if max_points < 1 or max_points > 1024 or max_proxies < 1 or max_proxies > 4 or max_snapshots < 1 or max_snapshots > 2:
        raise ValueError("Gallery CPU budget is 1–2 checkpoints, 1–4 proxies each and at most 1024 points per object.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Object galleries require matplotlib and numpy.") from exc
    with np.load(source, allow_pickle=False) as archive:
        if not {"sample_ids", "clouds"}.issubset(archive.files):
            return {"figures": [], "warnings": ["Object gallery unavailable: point-cloud pool needs explicit sample_ids and clouds arrays."]}
        ids, clouds = np.asarray(archive["sample_ids"]).reshape(-1), np.asarray(archive["clouds"])
        gold = np.asarray(archive["labels"]).reshape(-1) if "labels" in archive else None
        if clouds.ndim != 3 or clouds.shape[0] != len(ids) or clouds.shape[2] != 3:
            raise ValueError("Actual point-cloud pool must have shape candidates × points × 3 with matching object IDs.")
        if len(set(map(str, ids))) != len(ids):
            raise ValueError("Point-cloud object IDs must be unique.")
        if gold is not None and len(gold) != len(ids):
            raise ValueError("Point-cloud label/ID lengths differ.")
        lookup = {str(value): i for i, value in enumerate(ids)}
        eligible = [row for row in rows if row.get("retrieval", {}).get("hyperbolic", {}).get("topk_sample_ids")]
        by_run = {}
        for row in eligible:
            by_run.setdefault(row.get("run_key", row.get("run_id", "snapshot")), []).append(row)
        output = Path(output_dir) / "figures"
        output.mkdir(parents=True, exist_ok=True)
        arrays_dir = data.get("arrays_base_dir")
        for run_id, candidates in by_run.items():
            candidates.sort(key=lambda row: row.get("epoch", 0))
            selected = candidates[:1] if max_snapshots == 1 or len(candidates) == 1 else [candidates[0], candidates[-1]]
            for row in selected:
                depth, used = None, None
                if arrays_dir and row.get("arrays_file"):
                    file = Path(arrays_dir) / row["arrays_file"]
                    if file.is_file():
                        with np.load(file, allow_pickle=False) as values:
                            depth = np.asarray(values["proxy_depth"]).copy() if "proxy_depth" in values else None
                            used = np.asarray(values["proxy_used_mask"]).copy() if "proxy_used_mask" in values else None
                selected_proxies = representative_proxy_rows(row, depth, used, max_proxies)
                methods = [(name, row.get("retrieval", {}).get(name, {}).get("topk_sample_ids", [])) for name in ("hyperbolic", "direction")]
                proxy_ids = row.get("retrieval", {}).get("proxy_ids", [])
                for proxy_row, reason in selected_proxies:
                    object_ids = [str(value) for _, nearest in methods if proxy_row < len(nearest) for value in nearest[proxy_row][:4]]
                    present = [lookup[value] for value in object_ids if value in lookup]
                    if not present:
                        warnings.append(f"{run_id}: e{row.get('epoch')} gallery skipped: selected object IDs are absent from the actual point-cloud pool.")
                        continue
                    limit = max(float(np.max(np.abs(clouds[index, :max_points]))) for index in present) * 1.05
                    if not np.isfinite(limit) or limit <= 0:
                        raise ValueError("Selected actual point clouds contain nonfinite or degenerate coordinates.")
                    fig = plt.figure(figsize=(12, 6.4))
                    proxy_id = proxy_ids[proxy_row] if proxy_row < len(proxy_ids) else f"unverified row {proxy_row}"
                    for method_index, (method, nearest) in enumerate(methods):
                        values = nearest[proxy_row][:4] if proxy_row < len(nearest) else []
                        for position in range(4):
                            ax = fig.add_subplot(2, 4, method_index * 4 + position + 1, projection="3d")
                            if position < len(values) and str(values[position]) in lookup:
                                index = lookup[str(values[position])]
                                points = clouds[index, :max_points]
                                if not np.isfinite(points).all():
                                    raise ValueError("Selected actual point cloud contains nonfinite points.")
                                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.3, color="#2563eb", alpha=0.7, rasterized=True)
                                label = f"{method}: ID {values[position]}"
                                if gold is not None:
                                    label += f" · class {gold[index]}"
                                ax.set_title(label, fontsize=8)
                            else:
                                ax.text2D(0.1, 0.45, "Actual object unavailable", transform=ax.transAxes, fontsize=8)
                            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
                            ax.set_box_aspect((1, 1, 1))
                            ax.view_init(elev=20, azim=35)
                            ax.set_axis_off()
                    fig.suptitle(f"{run_id} · e{row.get('epoch')} · proxy {proxy_id}\nActual point clouds: {reason}; fixed camera and common scale", fontsize=12)
                    fig.tight_layout(rect=(0, 0, 1, 0.93))
                    stem = _slug(run_id) + f"_e{row.get('epoch')}_proxyrow{proxy_row}_objects"
                    png, svg = output / (stem + ".png"), output / (stem + ".svg")
                    fig.savefig(png, dpi=150, bbox_inches="tight")
                    fig.savefig(svg, bbox_inches="tight")
                    plt.close(fig)
                    figures.append({"run_id": str(run_id), "kind": "object_gallery",
                                    "title": f"e{row.get('epoch')} proxy {proxy_id}: actual object gallery",
                                    "note": f"Original candidate point clouds, at most {max_points} points each; fixed elev20/azim35 and a common scale within this gallery. Rows compare raw hyperbolic versus direction top-k retrieval. These are not embedding coordinates or training descendants.",
                                    "png": "figures/" + png.name, "svg": "figures/" + svg.name})
    return {"figures": figures, "warnings": warnings}
