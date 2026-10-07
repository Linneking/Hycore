"""Observed proxy usage lifetimes, with missing epochs and optimization separated.

These records count saved hard-Gumbel ancestor selections. They do not imply
nonzero gradients, parameter displacement, semantic proxies or object retention.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def run_key(analysis):
    return str(analysis.get("audit_run_key") or analysis.get("identity", {}).get("audit_run_key")
               or analysis.get("display_name") or analysis.get("run_id", "unknown"))


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def concentration(counts):
    """Zero selections have undefined concentration, rather than perfect equality."""
    if not counts or any(not _finite(x) or x < 0 for x in counts):
        raise ValueError("Usage counts must be a nonempty list of finite nonnegative numbers")
    total = float(sum(counts))
    used = sum(x > 0 for x in counts)
    if total == 0:
        return {"selection_count": 0., "used_proxy_count": 0, "effective_proxy_count": 0.,
                "gini": None, "top1_share": None, "top10_share": None}
    ordered = sorted(float(x) for x in counts)
    n = len(ordered)
    gini = 2 * sum((i + 1) * x for i, x in enumerate(ordered)) / (n * total) - (n + 1) / n
    probabilities = [x / total for x in counts if x > 0]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return {"selection_count": total, "used_proxy_count": used,
            "effective_proxy_count": math.exp(entropy), "gini": max(0., min(1., gini)),
            "top1_share": max(counts) / total, "top10_share": sum(ordered[-10:]) / total}


def optimization_state(row):
    """Classify recorded optimization, never infer it just from phase names."""
    if row is None:
        return "unknown"
    weight = row.get("metrics", {}).get("lambda_hier")
    steps = row.get("proxy_steps")
    if _finite(weight) and weight == 0 or _finite(steps) and steps == 0:
        return "monitor_only"
    if _finite(steps) and steps > 0:
        return "optimized"
    return "unknown"


def _validated_records(records, epoch_rows):
    values, seen = [], set()
    for record in records:
        epoch = record.get("epoch")
        ids, counts = record.get("proxy_ids"), record.get("counts")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Every usage record needs an actual integer epoch")
        if epoch in seen:
            raise ValueError("Ambiguous duplicate usage records for the same group and epoch")
        seen.add(epoch)
        if not isinstance(ids, list) or not ids or len(ids) != len(counts or []):
            raise ValueError("Usage records require explicit matching proxy_ids and counts")
        if len(set(ids)) != len(ids):
            raise ValueError("Proxy IDs must be unique within each usage record")
        concentration(counts)
        values.append({**record, "optimization_state": optimization_state(epoch_rows.get(epoch))})
    values.sort(key=lambda record: record["epoch"])
    if values:
        expected = set(values[0]["proxy_ids"])
        if any(set(record["proxy_ids"]) != expected for record in values):
            raise ValueError("Proxy ID pools change; lifetime records cannot assume persistent parameter rows")
    return values


def _summarize_group(records, view):
    records = [r for r in records if view == "all_observed" or r["optimization_state"] == "optimized"]
    if not records:
        return {"available": False, "reason": "No saved counts in this observation/optimization view"}
    ids = list(records[0]["proxy_ids"])
    aligned = [dict(zip(row["proxy_ids"], row["counts"])) for row in records]
    segments = []
    for index, row in enumerate(records):
        if index == 0 or row["epoch"] != records[index - 1]["epoch"] + 1:
            segments.append([])
        segments[-1].append(index)
    summaries, running_used, first_observed = [], set(), set()
    previous, previous_epoch = None, None
    for index, row in enumerate(records):
        active = {proxy for proxy, count in aligned[index].items() if count > 0}
        first = active - first_observed
        first_observed |= active
        running_used |= active
        consecutive = previous_epoch is not None and row["epoch"] == previous_epoch + 1
        summaries.append({"epoch": row["epoch"], "model_updates": row.get("model_updates"),
                          "optimization_state": row["optimization_state"],
                          **concentration(list(aligned[index].values())),
                          "cumulative_observed_used_count": len(running_used),
                          "first_observed_use_count": len(first),
                          "became_inactive_count": len(previous - active) if consecutive else None,
                          "newly_active_count": len(active - previous) if consecutive else None,
                          "transition_observed": consecutive,
                          "previous_observed_epoch": previous_epoch})
        previous, previous_epoch = active, row["epoch"]
    proxy_summaries = []
    for proxy in ids:
        counts = [row[proxy] for row in aligned]
        active_indices = [i for i, count in enumerate(counts) if count > 0]
        spans, reactivations = [], 0
        for segment in segments:
            previously_active, start = False, None
            for offset, index in enumerate(segment):
                if counts[index] == 0 and start is None:
                    start = offset
                if counts[index] > 0:
                    if start is not None:
                        span = segment[start:offset]
                        spans.append({"from_epoch": records[span[0]]["epoch"],
                                      "to_epoch": records[span[-1]]["epoch"],
                                      "observed_zero_epochs": len(span),
                                      "left_censored": start == 0,
                                      "right_censored": False})
                        if previously_active:
                            reactivations += 1
                        start = None
                    previously_active = True
            if start is not None:
                span = segment[start:]
                spans.append({"from_epoch": records[span[0]]["epoch"], "to_epoch": records[span[-1]]["epoch"],
                              "observed_zero_epochs": len(span), "left_censored": start == 0,
                              "right_censored": True})
        proxy_summaries.append({"proxy_id": proxy, "cumulative_observed_selections": sum(counts),
                                "observed_used_epochs": len(active_indices),
                                "observed_zero_epochs": len(records) - len(active_indices),
                                "first_observed_use_epoch": records[active_indices[0]]["epoch"] if active_indices else None,
                                "last_observed_use_epoch": records[active_indices[-1]]["epoch"] if active_indices else None,
                                "observed_reactivation_count": reactivations,
                                "max_contiguous_observed_zero_epochs": max((s["observed_zero_epochs"] for s in spans), default=0),
                                "zero_spans": spans})
    cumulative = concentration([p["cumulative_observed_selections"] for p in proxy_summaries])
    return {"available": True, "view": view, "observed_epochs": [r["epoch"] for r in records],
            "observed_epoch_count": len(records),
            "segments": [{"from_epoch": records[s[0]]["epoch"], "to_epoch": records[s[-1]]["epoch"],
                          "observed_epoch_count": len(s)} for s in segments],
            "unknown_optimization_epoch_count": sum(r["optimization_state"] == "unknown" for r in records),
            "proxy_count": len(ids), "cumulative_concentration": cumulative,
            "never_used_in_observed_window_count": sum(p["observed_used_epochs"] == 0 for p in proxy_summaries),
            "reactivated_proxy_count": sum(p["observed_reactivation_count"] > 0 for p in proxy_summaries),
            "per_epoch": summaries, "proxies": proxy_summaries,
            "interpretation": "Censored zero spans describe only saved, consecutive epochs. An observation gap cannot establish continued dormancy or reactivation."}


def _write_csv(path, records):
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _figures(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    for item in results:
        groups = [g for g in item["groups"] if g.get("available") and g["role"] == "combined" and g["component"] == "sample"
                  and g["views"]["optimization_observed"].get("available")]
        if not groups:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        for group in groups:
            view = group["views"]["optimization_observed"]
            if not view.get("available"):
                continue
            rows = view["per_epoch"]
            x = np.array([r["epoch"] for r in rows])
            gap = np.r_[False, np.diff(x) != 1]
            for ax, field, label in ((axes[0], "gini", "Cumulative observed use / concentration"),
                                     (axes[1], "cumulative_observed_used_count", "Cumulative observed proxy IDs")):
                y = np.array([np.nan if r[field] is None else r[field] for r in rows], dtype=float)
                # Insert a NaN rather than draw a line through an unobserved gap.
                xx, yy = [], []
                for epoch, value, missing in zip(x, y, gap):
                    if missing:
                        xx.append(np.nan); yy.append(np.nan)
                    xx.append(epoch); yy.append(value)
                ax.plot(xx, yy, label=group["domain"])
                ax.set(xlabel="Actual saved epoch (optimized only)", ylabel=field.replace("_", " "))
                ax.grid(alpha=.22)
        axes[0].set_ylim(-.02, 1.02)
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle(item["run_key"] + " · actual sample ancestor usage histories")
        fig.tight_layout()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", item["run_key"])[:85] + "_" + hashlib.sha256(item["run_key"].encode()).hexdigest()[:8] + "_lifetime"
        for extension in ("png", "svg"):
            fig.savefig(output / (stem + "." + extension), dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append({"run_id": item["run_key"], "kind": "usage_lifetime",
                        "title": "Observed proxy usage concentration and lifetime coverage",
                        "note": "Actual hard-Gumbel selections, optimized epochs only. Gaps are not connected; all_draws, noncollision and active_noncollision remain separate. Cumulative coverage does not show object-neighbour retention.",
                        "png": "figures/" + stem + ".png", "svg": "figures/" + stem + ".svg"})
    return figures


def analyze_longitudinal(analyses, output_dir=None, make_plots=True):
    """Return and optionally save JSON/CSV/PNG/SVG usage histories.

    output_dir is an already-created audit directory. Dedicated filenames never
    overwrite source data; existing longitudinal output files are rejected.
    """
    results, warnings, proxy_csv, epoch_csv = [], [], [], []
    for analysis in analyses:
        key = run_key(analysis)
        epoch_rows = {r["epoch"]: r for r in analysis.get("epochs", [])}
        groups = defaultdict(list)
        for row in analysis.get("proxy_usage", []):
            groups[(row.get("component"), row.get("domain"), row.get("role"))].append(row)
        run = {"run_key": key, "display_name": analysis.get("display_name", key), "available": bool(groups),
               "groups": [], "reason": None if groups else "Per-proxy ID counts not saved; lifetime history is unavailable, not all-zero."}
        for identity, records in sorted(groups.items()):
            try:
                records = _validated_records(records, epoch_rows)
                views = {view: _summarize_group(records, view) for view in ("all_observed", "optimization_observed")}
            except ValueError as exc:
                warnings.append(key + ": " + str(exc))
                run["groups"].append({"component": identity[0], "domain": identity[1], "role": identity[2],
                                      "available": False, "reason": str(exc)})
                continue
            group = dict(zip(("component", "domain", "role"), identity))
            group.update(available=True, views=views)
            run["groups"].append(group)
            for view_name, view in views.items():
                prefix = {"run_key": key, **dict(zip(("component", "domain", "role"), identity)), "view": view_name}
                for proxy in view.get("proxies", []):
                    proxy_csv.append({**prefix, **{k: v for k, v in proxy.items() if k != "zero_spans"}})
                epoch_csv.extend({**prefix, **r} for r in view.get("per_epoch", []))
        results.append(run)
    result = {"schema_version": 1, "runs": results, "warnings": warnings, "figures": [],
              "definitions": ["An observed first/last use is bounded by saved history.",
                              "Zero-count epochs count ancestor selections, not nonzero gradients or parameter movements.",
                              "Monitor-only/warmup records are excluded from the optimization view.",
                              "Proxy IDs have meaning only within their own run; rows are never paired across versions."]}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("longitudinal_summary.json", "proxy_lifetimes.csv", "proxy_lifetime_epochs.csv"):
            if (output / name).exists():
                raise FileExistsError("Existing longitudinal audit outputs cannot be overwritten")
        if make_plots:
            result["figures"] = _figures(results, output)
        (output / "longitudinal_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_csv(output / "proxy_lifetimes.csv", proxy_csv)
        _write_csv(output / "proxy_lifetime_epochs.csv", epoch_csv)
    return result
