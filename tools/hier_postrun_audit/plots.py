"""Scientific figures for saved HIER telemetry and frozen cache snapshots.

All plots describe observations; no curve interpolates missing epochs and no
low-dimensional projection is presented as the original hyperbolic geometry.
Matplotlib is imported only when a report is rendered.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _slug(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:60] or "run"
    return label + "_" + hashlib.sha256(str(value).encode("utf8")).hexdigest()[:8]


def _metrics(row):
    return row.get("metrics", {})


def _series(rows, key):
    return [float(_metrics(row)[key]) if _number(_metrics(row).get(key)) else float("nan") for row in rows]


def _has(rows, key):
    return any(_number(_metrics(row).get(key)) for row in rows)


def _gapped_rows(rows):
    """Insert plot-only NaN separators, never invented scalar observations."""
    result = []
    for row in rows:
        if result and _number(row.get("epoch")) and _number(result[-1].get("epoch")) and row["epoch"] - result[-1]["epoch"] > 1:
            old = result[-1]
            updates = ((old["model_updates"] + row["model_updates"]) / 2
                       if _number(old.get("model_updates")) and _number(row.get("model_updates")) else None)
            result.append({"epoch": (old["epoch"] + row["epoch"]) / 2,
                           "model_updates": updates, "metrics": {}, "phase": None})
        result.append(row)
    return result


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def usage_transition(previous, current):
    """JSD in bits and tie-aware Spearman, only for identical proxy identity sets.

    Empty draws are unavailable, not a zero-divergence observation. A flat
    rank vector has undefined Spearman. Counts are aligned by explicit ID.
    """
    old_ids, new_ids = previous.get("proxy_ids", []), current.get("proxy_ids", [])
    old_counts, new_counts = previous.get("counts", []), current.get("counts", [])
    if len(set(map(str, old_ids))) != len(old_ids) or len(set(map(str, new_ids))) != len(new_ids):
        return {"available": False, "reason": "duplicate proxy IDs"}
    if set(map(str, old_ids)) != set(map(str, new_ids)) or not old_ids:
        return {"available": False, "reason": "proxy identity sets differ or are missing"}
    if len(old_counts) != len(old_ids) or len(new_counts) != len(new_ids):
        return {"available": False, "reason": "proxy count/ID lengths differ"}
    old = dict(zip(map(str, old_ids), old_counts))
    new = dict(zip(map(str, new_ids), new_counts))
    ids = sorted(old)
    a, b = [old[i] for i in ids], [new[i] for i in ids]
    if any(not _number(v) or v < 0 for v in a + b) or sum(a) == 0 or sum(b) == 0:
        return {"available": False, "reason": "nonfinite, negative or empty proxy draws"}
    p, q = [v / sum(a) for v in a], [v / sum(b) for v in b]
    midpoint = [(x + y) / 2 for x, y in zip(p, q)]
    jsd = sum(0.5 * x * math.log2(x / m) for x, m in zip(p, midpoint) if x > 0)
    jsd += sum(0.5 * y * math.log2(y / m) for y, m in zip(q, midpoint) if y > 0)
    ra, rb = _average_ranks(a), _average_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    old_used, new_used = {i for i in ids if old[i] > 0}, {i for i in ids if new[i] > 0}
    return {"available": True, "jsd_bits": max(0.0, min(1.0, jsd)),
            "rank_spearman": numerator / denominator if denominator else None,
            "used_set_jaccard": len(old_used & new_used) / len(old_used | new_used),
            "newly_used_count": len(new_used - old_used), "became_inactive_count": len(old_used - new_used),
            "previous_used_count": len(old_used), "current_used_count": len(new_used)}


def _decorate(ax, rows, metadata, ylabel):
    ax.set_xlabel("Saved epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    epochs = [r.get("epoch", i) for i, r in enumerate(rows)]
    if len(epochs) > 1 and epochs[0] != epochs[-1]:
        updates = [r.get("model_updates") for r in rows]
        if all(_number(v) for v in updates):
            # The second axis names the cumulative training timeline. The input
            # adapter separately records logged updates for inherited prefixes.
            secondary = ax.secondary_xaxis("top")
            selected = sorted(set([0, len(rows) // 2, len(rows) - 1]))
            secondary.set_xticks([epochs[i] for i in selected])
            secondary.set_xticklabels([f"{int(updates[i]):,}" for i in selected])
            secondary.set_xlabel("Model updates (protocol timeline)")
    warmup = metadata.get("warmup_epochs")
    if _number(warmup) and warmup > 0 and epochs and min(epochs) <= warmup <= max(epochs):
        ax.axvline(warmup, color="#6b7280", linestyle="--", linewidth=1, label="HIER warmup ends")
    old_phase = None
    for row in rows:
        phase = row.get("phase")
        if old_phase is not None and phase is not None and phase != old_phase:
            ax.axvline(row["epoch"], color="#9ca3af", linestyle=":", linewidth=1)
        if phase is not None:
            old_phase = phase


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Keep explanatory labels outside the observations, particularly when
        # many source-defined hierarchy fields share a panel.
        compact = []
        for label in labels:
            label = label.replace("active noncollision", "live hinge").replace("all draws", "all")
            label = label.replace("effective proxy count", "effective IDs").replace("used proxy count", "used IDs")
            label = label.replace("positive fraction", "+ fraction").replace("negative fraction", "− fraction")
            compact.append(label)
        ax.legend(handles, compact, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                  ncol=2 if len(handles) > 1 else 1, frameon=False)


def save_figures(analyses, snapshots, output_dir):
    """Write independent PNG and SVG plots; return public relative filenames."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("HIER audit figures require matplotlib and numpy; install the audit requirements before rendering. No graphless report was generated.") from exc
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures, warnings = [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "figure.dpi": 110, "savefig.dpi": 160, "svg.fonttype": "none"})

    def save(fig, run_id, stem, title, note):
        base = _slug(run_id) + "_" + stem
        fig.tight_layout()
        png, svg = figure_dir / (base + ".png"), figure_dir / (base + ".svg")
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        figures.append({"run_id": str(run_id), "title": title, "note": note,
                        "png": "figures/" + png.name, "svg": "figures/" + svg.name})

    for analysis in analyses:
        run_id = analysis.get("run_id", "unknown")
        display = analysis.get("display_name", run_id)
        metadata = analysis.get("metadata", {})
        rows = _gapped_rows(sorted(analysis.get("epochs", []), key=lambda row: row.get("epoch", 0)))
        if not rows:
            warnings.append(f"{display}: no saved epoch scalar records; epoch plots unavailable.")
            continue
        epochs = [r.get("epoch", i) for i, r in enumerate(rows)]

        def lines(stem, title, candidates, ylabel, note):
            keys = [(key, label) for key, label in candidates if _has(rows, key)]
            if not keys:
                return
            fig, ax = plt.subplots(figsize=(9, 4.4))
            for key, label in keys:
                values = _series(rows, key)
                sparse = any(not math.isfinite(v) for v in values)
                ax.plot(epochs, values, label=label, linewidth=1.25, marker="." if sparse else None, markersize=4)
            ax.set_title(f"{display}: {title}")
            _decorate(ax, rows, metadata, ylabel)
            _legend(ax)
            save(fig, run_id, stem, title, note)

        lines("classification", "Classification trajectory", [
            ("train_aug_oa_pct", "Augmented training OA"), ("clean_train_oa_pct", "Clean training OA"),
            ("val_oa_pct", "Validation OA"), ("val_aa_pct", "Validation AA")], "Accuracy (%)",
            "Sparse clean-train evaluations remain gaps. Official test results are not used as a training trajectory.")
        gap_rows = []
        for row in rows:
            clean, val = _metrics(row).get("clean_train_oa_pct"), _metrics(row).get("val_oa_pct")
            if _number(clean) and _number(val):
                gap_rows.append((row["epoch"], clean - val))
        if gap_rows:
            fig, ax = plt.subplots(figsize=(9, 4.0))
            ax.scatter([v[0] for v in gap_rows], [v[1] for v in gap_rows], s=14, label="Clean train OA − validation OA")
            ax.axhline(0, color="#6b7280", linewidth=0.8)
            ax.set_title(f"{display}: Generalization gap")
            _decorate(ax, rows, metadata, "Percentage points")
            _legend(ax)
            save(fig, run_id, "generalization_gap", "Generalization gap", "Only matching saved clean-train/validation epochs are compared; gaps are not causal evidence for HIER.")
        lines("loss_components", "Objective components", [
            ("loss", "Total loss"), ("base_loss", "Base loss"), ("ce_loss", "Cross entropy"),
            ("hier_loss", "HIER unweighted"), ("sample_hier_loss", "Sample–proxy unweighted"),
            ("proxy_hier_loss", "Proxy–proxy unweighted"), ("weighted_hier_loss", "Weighted HIER")], "Saved loss value",
            "Component scales follow their logged definition. Unweighted component magnitudes are not gradient contributions.")
        for component in ("whole", "proxy"):
            keys = [component + "_depth_" + suffix for suffix in ("p10", "median", "mean", "p90", "max")]
            if any(_has(rows, key) for key in keys):
                fig, ax = plt.subplots(figsize=(9, 4.3))
                lower, upper = component + "_depth_p10", component + "_depth_p90"
                if _has(rows, lower) and _has(rows, upper):
                    ax.fill_between(epochs, _series(rows, lower), _series(rows, upper), alpha=0.2, label="Saved p10–p90 summaries")
                for suffix in ("median", "mean", "max"):
                    key = component + "_depth_" + suffix
                    if _has(rows, key):
                        ax.plot(epochs, _series(rows, key), label="Saved " + suffix, linewidth=1.2)
                ax.set_title(f"{display}: {component} origin depth")
                _decorate(ax, rows, metadata, "Hyperbolic distance to origin")
                _legend(ax)
                save(fig, run_id, component + "_depth", component.title() + " depth",
                     "Saved training-forward summaries include sampled repeats, augmentation and BN effects. Proxy per-batch summaries must not be read as pooled clean-dataset quantiles. Origin depth is not Euclidean coordinate radius.")
        c = metadata.get("curvature")
        for component in ("whole", "proxy"):
            selected = []
            for suffix in ("p10", "median", "mean", "p90"):
                qkey, rkey = component + "_q_" + suffix, component + "_radius_" + suffix
                if _has(rows, qkey):
                    selected.append((suffix, _series(rows, qkey)))
                elif _has(rows, rkey) and _number(c) and c > 0:
                    selected.append((suffix, [v * math.sqrt(c) for v in _series(rows, rkey)]))
            if component == "proxy" and not selected and _has(rows, "proxy_radius_median_mean") and _number(c) and c > 0:
                selected.append(("mean of batch medians", [v * math.sqrt(c) for v in _series(rows, "proxy_radius_median_mean")]))
            if selected:
                fig, ax = plt.subplots(figsize=(9, 4.0))
                for label, values in selected:
                    ax.plot(epochs, values, label=label)
                ax.axhline(1, color="#6b7280", linestyle="--", linewidth=0.8, label="Mathematical ball boundary")
                ax.set_title(f"{display}: {component} normalized coordinate radius")
                _decorate(ax, rows, metadata, "q = √c · Euclidean ball radius")
                _legend(ax)
                save(fig, run_id, component + "_normalized_radius", component.title() + " normalized radius",
                     "The unit boundary is a coordinate limit with infinite hyperbolic origin distance. Curvature normalization is performed only when c is recorded.")
        metric_keys = sorted({key for row in rows for key, value in _metrics(row).items() if _number(value)})
        for stem, predicate, title, ylabel, note in [
            ("boundary_fractions", lambda k: any(w in k for w in ("boundary", "saturation", "projection", "over_cap", "overcap", "cap_hit")) and any(w in k for w in ("fraction", "share", "ratio")), "Boundary / projection fractions", "Recorded fraction", "Coordinate boundary proximity, forward projection and parameter cap events are different mechanisms; field names retain these definitions."),
            ("boundary_counts", lambda k: any(w in k for w in ("boundary", "saturation", "projection", "over_cap", "overcap", "cap_hit")) and any(w in k for w in ("count", "events")), "Boundary / constraint event counts", "Recorded count", "Repeated per-step cap events can exceed the number of distinct proxies. A missing count is never interpreted as zero."),
            ("proxy_usage_fractions", lambda k: "proxy" in k and "used" in k and "fraction" in k, "Proxy usage fractions", "Fraction of proxy IDs", "Fractions retain their all-draw/noncollision/live-hinge selection definition; they are not gradient-active fractions."),
            ("triplet_health", lambda k: (k.startswith("sample_") or k.startswith("proxy_")) and any(w in k for w in ("collision_fraction", "active_fraction", "active_noncollision_fraction", "self_k_fraction", "repeat_fraction", "cross_class_j_fraction")), "Triplet validity and activation", "Recorded fraction", "Proxy collision, equal object identity and a positive hinge are distinct conditions. A zero collision-masked loss does not imply a satisfied hierarchy."),
            ("relation_coverage", lambda k: (k.startswith("sample_") or k.startswith("proxy_")) and any(w in k for w in ("anchor_coverage", "candidate_triplet_coverage", "any_role_position_coverage", "eligible_training_fraction")), "Training relation coverage", "Recorded fraction", "Training role coverage describes sampled relations; repeated augmentation or class-balanced draws do not imply unique full-dataset coverage."),
            ("optimization_norms", lambda k: any(w in k for w in ("grad", "update", "displacement")) and any(w in k for w in ("norm", "median", "mean", "max")) and "count" not in k, "Gradient and update diagnostics", "Recorded norm / displacement", "Gradient magnitude is not AdamW displacement. Ball gradients, tangent parameter gradients and actual optimizer steps must not be interchanged."),
            ("clipping", lambda k: "clip" in k and "norm" not in k, "Gradient clipping diagnostics", "Recorded value", "Only saved clipping fields are plotted; no clipping history is inferred from loss magnitude."),
        ]:
            keys = [k for k in metric_keys if predicate(k)]
            # Avoid an unreadable figure when a source exports dozens of fields.
            for offset in range(0, len(keys), 9):
                subset = keys[offset:offset + 9]
                lines(stem + ("_" + str(offset // 9 + 1) if len(keys) > 9 else ""), title,
                      [(k, k.replace("_", " ")) for k in subset], ylabel, note)
        for component in ("sample", "proxy"):
            canonical = []
            for domain, label, color in (("all_draws", "All hard selections", "#2563eb"),
                                         ("noncollision", "Noncollision", "#dc2626"),
                                         ("active_noncollision", "Live hinge", "#059669")):
                for measure, suffix, style in (("used_proxy_count", "used IDs", "-"),
                                               ("effective_proxy_count", "effective IDs", "--")):
                    key = component + "_" + domain + "_" + measure
                    if _has(rows, key):
                        canonical.append((key, label + ": " + suffix, color, style))
            if canonical:
                fig, ax = plt.subplots(figsize=(9, 5.1))
                for key, label, color, style in canonical:
                    values = _series(rows, key)
                    ax.plot(epochs, values, label=label, color=color, linestyle=style, linewidth=1.3,
                            marker="." if any(not math.isfinite(v) for v in values) else None, markersize=4)
                ax.set_title(f"{display}: {component} ancestor proxy activation")
                _decorate(ax, rows, metadata, "Proxy IDs / effective number")
                _legend(ax)
                save(fig, run_id, component + "_proxy_activation", component.title() + " ancestor proxy activation",
                     "Solid: distinct hard-selected proxy IDs. Dashed: exp(entropy) effective number from the same domain's counts. All draws, noncollision draws and positive-hinge noncollision draws are separate definitions. Missing domains are omitted, not zero; duplicate all-draw aliases are excluded.")
            else:
                fallback = [(component + "_used_proxy_count", "Stored used-ID count"),
                            (component + "_effective_proxy_count", "Stored effective count")]
                lines(component + "_proxy_activation_stored", component.title() + " proxy activation (stored definition)",
                      fallback, "Stored count / effective number",
                      "Canonical all-draw/noncollision/live-hinge counts were not saved. These fallback fields retain their stored definition; they are not assigned a missing activation domain.")
        lines("proxy_parameters", "Saved proxy tangent parameters", [("proxy_tangent_norm_mean", "Mean parameter norm"), ("proxy_tangent_norm_max", "Maximum parameter norm"),
              ("proxy_parameter_median_tangent_norm_mean", "Mean of post-cap batch medians"), ("proxy_parameter_max_tangent_norm_mean", "Mean of post-cap batch maxima"),
              ("proxy_parameter_max_tangent_norm", "Maximum observed post-cap parameter norm")], "Euclidean tangent parameter norm",
              "Stored parameter norms are separate from mapped ball radius and hyperbolic depth; projected outputs can conceal parameter overshoot.")
        selected_depth = [k for k in metric_keys if (k.startswith("sample_") or k.startswith("proxy_")) and any(name in k for name in ("_pair_depth_mean", "_triple_depth_mean"))]
        for offset in range(0, len(selected_depth), 9):
            lines("selected_ancestor_depth_" + str(offset // 9 + 1), "Depth of actually selected ancestors", [(key, key.replace("_", " ")) for key in selected_depth[offset:offset + 9]], "Hyperbolic origin depth",
                  "Selection-weighted ancestor depth differs from the distribution of all proxies. Each role and all-draw/noncollision/live-hinge domain retains its saved definition.")
        ancestor_gaps = [k for k in metric_keys if "endpoint_gap" in k and any(w in k for w in ("positive_fraction", "negative_fraction"))]
        for offset in range(0, len(ancestor_gaps), 9):
            lines("ancestor_depth_gap_" + str(offset // 9 + 1), "Selected ancestor / endpoint depth ordering", [(key, key.replace("_", " ")) for key in ancestor_gaps[offset:offset + 9]], "Recorded signed-gap fraction",
                  "Signed gap conventions follow the source field. Population depth ordering is a diagnostic, not an explicit constraint guaranteed by the HIER triangle loss.")
        lines("model_clipping_norms", "Model gradient clipping", [("model_preclip_norm", "Pre-clip norm"), ("model_postclip_norm", "Post-clip norm")], "Saved parameter gradient norm",
              "Global model clipping can couple HIER and base gradients. Proxy AdamW updates have a separate optimizer and constraint.")
        lines("learning_rates", "Saved optimizer learning rates", [("model_lr", "Model LR"), ("proxy_lr", "Proxy LR")], "Learning rate",
              "Learning-rate schedules and inherited prefixes are protocol factors and must be matched in cross-version comparisons.")

        usage_groups = {}
        for record in analysis.get("proxy_usage", []):
            key = (record.get("component", "sample"), record.get("domain", "all_draws"), record.get("role", "combined"))
            if key[2] == "combined":
                usage_groups.setdefault(key, []).append(record)
        for key, records in sorted(usage_groups.items()):
            records.sort(key=lambda r: r.get("epoch", 0))
            if not records:
                continue
            ids = list(map(str, records[0].get("proxy_ids", [])))
            if not ids or len(set(ids)) != len(ids) or any(set(map(str, r.get("proxy_ids", []))) != set(ids) or len(set(map(str, r.get("proxy_ids", [])))) != len(r.get("proxy_ids", [])) for r in records):
                warnings.append(f"{display}: {key[0]}/{key[1]} usage heatmap unavailable: stable proxy ID sets are missing.")
                continue
            if any(len(r.get("counts", [])) != len(ids) for r in records):
                warnings.append(f"{display}: usage count/ID shape mismatch; heatmap not drawn.")
                continue
            matrix = []
            for record in records:
                counts = dict(zip(map(str, record["proxy_ids"]), record["counts"]))
                matrix.append([math.log10(1 + counts[i]) if _number(counts[i]) and counts[i] >= 0 else float("nan") for i in ids])
            fig, ax = plt.subplots(figsize=(10, 5.0))
            artist = ax.imshow(np.array(matrix).T, interpolation="nearest", aspect="auto", origin="lower", cmap="viridis")
            ticks = sorted(set([0, len(records) // 2, len(records) - 1]))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(records[i].get("epoch", "?")) for i in ticks])
            ytick = sorted(set([0, len(ids) // 2, len(ids) - 1]))
            ax.set_yticks(ytick)
            ax.set_yticklabels([ids[i] for i in ytick])
            ax.set_xlabel("Saved epoch (columns equally spaced)")
            ax.set_ylabel("Persistent proxy ID")
            ax.set_title(f"{display}: {key[0]} / {key[1]} hard-selection usage")
            fig.colorbar(artist, ax=ax, label="log10(1 + number of draws)")
            stem = "usage_" + "_".join(key)
            save(fig, run_id, stem, f"{key[0]} {key[1]} proxy-ID usage",
                 "Rows retain explicit proxy identities. Equal-width saved-epoch columns do not imply that every intermediate epoch was recorded. These are stochastic training selections, not nearest-neighbor descendants.")
            transitions = []
            for previous, current in zip(records, records[1:]):
                result = usage_transition(previous, current)
                if result.get("available"):
                    transitions.append({"epoch": current.get("epoch"), **result})
            if transitions:
                fig, axes = plt.subplots(4, 1, figsize=(9, 8.8), sharex=True)
                for ax, field, ylabel in zip(axes[:3], ("jsd_bits", "rank_spearman", "used_set_jaccard"), ("JSD (bits)", "Usage rank Spearman", "Used-ID Jaccard")):
                    ax.plot([r["epoch"] for r in transitions], [r[field] if _number(r.get(field)) else float("nan") for r in transitions], marker=".", linewidth=1)
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.2)
                axes[0].set_title(f"{display}: consecutive saved {key[0]}/{key[1]} usage stability")
                axes[-1].set_xlabel("Later saved epoch")
                axes[0].set_ylim(-0.03, 1.03)
                axes[1].set_ylim(-1.03, 1.03)
                axes[2].set_ylim(-0.03, 1.03)
                for field, label in (("newly_used_count", "Newly used"), ("became_inactive_count", "Became inactive")):
                    axes[3].plot([r["epoch"] for r in transitions], [r[field] for r in transitions], label=label)
                axes[3].set_ylabel("Proxy IDs")
                axes[3].grid(True, alpha=0.2)
                _legend(axes[3])
                save(fig, run_id, stem + "_stability", f"{key[0]} {key[1]} usage stability",
                     "Within-run consecutive saved records, identical proxy IDs, normalized counts. Empty draws and constant ranks are undefined. Variation includes batch composition and Gumbel noise; it is not a controlled temporal relation test.")

    snapshot_data = snapshots or {}
    snapshot_rows = snapshot_data.get("snapshots", []) if isinstance(snapshot_data, dict) else []
    grouped = {}
    for row in snapshot_rows:
        grouped.setdefault(row.get("run_key", row.get("run_id", "snapshot")), []).append(row)
    base_dir = snapshot_data.get("arrays_base_dir") if isinstance(snapshot_data, dict) else None
    for run_key, rows in grouped.items():
        rows.sort(key=lambda r: r.get("epoch", 0))
        x = [r.get("epoch", i) for i, r in enumerate(rows)]
        for component in ("whole", "proxy"):
            points = []
            for row in rows:
                quantiles = row.get(component, {}).get("depth_quantiles", {})
                if component == "proxy" and not quantiles:
                    quantiles = row.get("proxy", {}).get("all", {}).get("depth_quantiles", {})
                points.append(quantiles)
            if any(_number(q.get("p50")) for q in points):
                fig, ax = plt.subplots(figsize=(9, 4.2))
                value = lambda key: [q.get(key) if _number(q.get(key)) else float("nan") for q in points]
                ax.fill_between(x, value("p05"), value("p95"), alpha=0.2, label="Pooled p05–p95")
                ax.plot(x, value("p50"), marker="o", label="Pooled median")
                ax.set(xlabel="Checkpoint epoch", ylabel="Hyperbolic origin depth", title=f"{run_key}: fixed-pool {component} depth")
                ax.grid(True, alpha=0.2)
                _legend(ax)
                save(fig, run_key, "snapshot_" + component + "_depth", "Fixed-pool " + component + " depth",
                     "Checkpoint observations on explicitly identified frozen pools. Pooled quantiles are distinct from training-forward batch summaries. Missing checkpoints are not reconstructed.")
        for row in rows:
            filename = row.get("arrays_file")
            if filename and base_dir:
                path = Path(base_dir) / filename
                if path.exists():
                    with np.load(path, allow_pickle=False) as arrays:
                        if "whole_depth" in arrays and "proxy_depth" in arrays:
                            fig, ax = plt.subplots(figsize=(8.2, 4.1))
                            for key, label in (("whole_depth", "Whole fixed pool"), ("proxy_depth", "All proxies")):
                                values = np.asarray(arrays[key]).reshape(-1)
                                values = values[np.isfinite(values)]
                                if len(values):
                                    ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.4, label=f"{label} (n={len(values)})")
                            if "proxy_used_mask" in arrays:
                                mask = np.asarray(arrays["proxy_used_mask"]).astype(bool)
                                depth = np.asarray(arrays["proxy_depth"])
                                if len(mask) == len(depth):
                                    for selected, label in ((mask, "Training-selected proxies"), (~mask, "Not selected in recorded window")):
                                        values = depth[selected]
                                        values = values[np.isfinite(values)]
                                        if len(values):
                                            ax.hist(values, bins=40, density=True, histtype="step", linestyle="--", linewidth=1.1, label=f"{label} (n={len(values)})")
                            ax.set(xlabel="Hyperbolic origin depth", ylabel="Density", title=f"{run_key}: e{row.get('epoch')} actual depth distribution")
                            _legend(ax)
                            save(fig, run_key, f"snapshot_e{row.get('epoch')}_depth_distribution", f"e{row.get('epoch')} depth distribution",
                                 "Actual cache arrays, independently normalized densities. Used/inactive groups are shown only when an explicit training-selection window is supplied. Proxy and object populations differ in size and role.")
                            purity_series = []
                            for method in ("hyperbolic", "direction"):
                                purity = row.get("retrieval", {}).get(method, {}).get("proxy_topk_purity", [])
                                if len(purity) == len(arrays["proxy_depth"]):
                                    purity_series.append((method, purity))
                            if purity_series:
                                fig, ax = plt.subplots(figsize=(8.2, 4.1))
                                for method, purity in purity_series:
                                    ax.scatter(arrays["proxy_depth"], purity, alpha=0.38, s=12, label=method)
                                ax.set(xlabel="Proxy hyperbolic origin depth", ylabel="Top-k majority-class purity", ylim=(-0.03, 1.03), title=f"{run_key}: e{row.get('epoch')} depth and class purity")
                                _legend(ax)
                                save(fig, run_key, f"snapshot_e{row.get('epoch')}_depth_purity", f"e{row.get('epoch')} depth / class purity",
                                     "Every point represents one supplied proxy row. High class purity does not establish an intra-class morphology hierarchy; radial-bias controls are reported separately.")
            retrieval = row.get("retrieval", {})
            methods = [(key, retrieval[key]) for key in ("hyperbolic", "direction", "equal_radius_hyperbolic") if isinstance(retrieval.get(key), dict)]
            if methods:
                coverage, hotspot, labels = [], [], []
                for method, values in methods:
                    labels.append(method.replace("_", " "))
                    coverage.append(values.get("coverage", {}).get("unique_fraction", float("nan")))
                    hotspot.append(values.get("slot_concentration", {}).get("top4_sample_slot_share", float("nan")))
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
                axes[0].bar(labels, coverage, color="#2563eb")
                axes[1].bar(labels, hotspot, color="#dc2626")
                for ax, title in zip(axes, ("Unique object coverage", "Top-4 object slot concentration")):
                    ax.set(title=title, ylabel="Fraction", ylim=(0, 1))
                    ax.tick_params(axis="x", labelrotation=20)
                fig.suptitle(f"{run_key}: e{row.get('epoch')} frozen retrieval controls")
                save(fig, run_key, f"snapshot_e{row.get('epoch')}_retrieval", f"e{row.get('epoch')} raw / direction retrieval",
                     "Raw hyperbolic nearest objects and direction controls are frozen retrieval observations. They are not the objects selected as HIER training ancestors. Purity alone does not establish a morphology hierarchy.")
    transitions = snapshot_data.get("transitions", []) if isinstance(snapshot_data, dict) else []
    transition_groups = {}
    for row in transitions:
        if row.get("available"):
            transition_groups.setdefault(row.get("run_key", "snapshot"), []).append(row)
    for run_key, rows in transition_groups.items():
        rows.sort(key=lambda r: r.get("to_epoch", 0))
        fig, ax = plt.subplots(figsize=(9, 4.2))
        drew = False
        for method in ("hyperbolic", "direction"):
            values = [r.get("retrieval", {}).get(method, {}).get("mean_topk_jaccard") for r in rows]
            if any(_number(v) for v in values):
                ax.plot([r.get("to_epoch") for r in rows], [v if _number(v) else float("nan") for v in values], marker="o", label=method)
                drew = True
        if drew:
            ax.set(xlabel="Later checkpoint epoch", ylabel="Mean per-proxy top-k Jaccard", ylim=(-0.03, 1.03), title=f"{run_key}: identity-aligned retrieval stability")
            ax.grid(True, alpha=0.2)
            _legend(ax)
            save(fig, run_key, "snapshot_retrieval_stability", "Fixed-pool retrieval stability",
                 "Explicit same-run proxy IDs and same candidate object pool are required. Snapshot spacing and relation definition are preserved.")
        else:
            plt.close(fig)
        indexed = {r.get("epoch"): r for r in grouped.get(run_key, [])}
        for method in ("hyperbolic", "direction"):
            columns, labels, stable_ids = [], [], None
            for transition in rows:
                old, new = indexed.get(transition.get("from_epoch")), indexed.get(transition.get("to_epoch"))
                if not old or not new or not transition.get("retrieval", {}).get(method, {}).get("available", True):
                    continue
                old_ids = list(map(str, old.get("retrieval", {}).get("proxy_ids", [])))
                new_ids = list(map(str, new.get("retrieval", {}).get("proxy_ids", [])))
                old_top = old.get("retrieval", {}).get(method, {}).get("topk_sample_ids", [])
                new_top = new.get("retrieval", {}).get(method, {}).get("topk_sample_ids", [])
                if not old.get("proxy", {}).get("proxy_ids_stable") or not new.get("proxy", {}).get("proxy_ids_stable"):
                    continue
                if not old_ids or set(old_ids) != set(new_ids) or len(old_top) != len(old_ids) or len(new_top) != len(new_ids):
                    continue
                if stable_ids is None:
                    stable_ids = old_ids
                if set(stable_ids) != set(old_ids):
                    continue
                a, b = dict(zip(old_ids, old_top)), dict(zip(new_ids, new_top))
                column = []
                for proxy_id in stable_ids:
                    before, after = set(map(str, a[proxy_id])), set(map(str, b[proxy_id]))
                    column.append(len(before & after) / len(before | after) if before | after else float("nan"))
                columns.append(column)
                labels.append(f"{transition.get('from_epoch')}→{transition.get('to_epoch')}")
            if columns:
                fig, ax = plt.subplots(figsize=(9, 5.0))
                artist = ax.imshow(np.array(columns).T, interpolation="nearest", aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
                ticks = sorted(set([0, len(columns) // 2, len(columns) - 1]))
                ax.set_xticks(ticks)
                ax.set_xticklabels([labels[i] for i in ticks])
                ytick = sorted(set([0, len(stable_ids) // 2, len(stable_ids) - 1]))
                ax.set_yticks(ytick)
                ax.set_yticklabels([stable_ids[i] for i in ytick])
                ax.set(xlabel="Validated checkpoint transition", ylabel="Persistent proxy ID", title=f"{run_key}: {method} per-proxy top-k stability")
                fig.colorbar(artist, ax=ax, label="Top-k object-ID Jaccard")
                save(fig, run_key, "snapshot_" + method + "_proxy_topk_jaccard", method + " per-proxy retrieval stability",
                     "Each cell aligns the same explicit proxy ID on a validated same-object-pool checkpoint transition. No cross-run proxy identity matching is assumed.")
    from .gallery import save_object_galleries
    gallery = save_object_galleries(snapshots, output_dir)
    figures.extend(gallery["figures"])
    warnings.extend(gallery["warnings"])
    return {"figures": figures, "warnings": warnings}
