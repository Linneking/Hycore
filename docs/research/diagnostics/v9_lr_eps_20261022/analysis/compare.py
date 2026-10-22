"""CPU-only anonymous comparison of the LR/eps diagnostic with report71.

Reads private step arrays but exports aggregates only. Does not load an encoder,
dataset, optimizer or checkpoint; no model forward/backward or GPU operation.
The imported report71 analyzer provides cap transitions and exact Adam accounting.
Training losses are monitored on changing batches, not fixed evaluation losses.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parents[1] / "v9_proxy_cap_cause_20261021" / "analyze.py"
REPORT71 = HERE.parents[2] / "71_V9_PROXY_CAP_CAUSE_2026-10-22" / "summary.json"
COUNTS = ("triplets", "collisions", "noncollision_triplets", "active_triplets",
          "active_i_hinges", "active_j_hinges", "active_k_hinges")
SCALARS = ("loss", "ce", "base", "intra_contrastive", "intra_radial",
           "hier_monitored_loss", "proxy_gradient_norm", "whole_depth", "part_depth")


def load_legacy():
    spec = importlib.util.spec_from_file_location("v9_lr_eps_report71_analyzer", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A live writer may not have finished its last line yet.
                if not line.endswith("\n") and not stream.read(1):
                    break
                raise


def finite(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Nonfinite scalar in training telemetry")
    return value


def aggregate_learning(path, maximum_key=None):
    """Pool actual counts, never means of unequal per-step hinge fractions."""
    epochs = {}
    previous = (0, 0)
    for row in jsonl(path):
        key = int(row["epoch"]), int(row["step"])
        if key <= previous:
            raise ValueError("Training rows duplicated or not chronological")
        previous = key
        if maximum_key is not None and key > maximum_key:
            break
        e, step = key
        dest = epochs.setdefault(e, {"epoch": e, "steps": 0, "first_step": step,
            "scalar_sums": defaultdict(float), "scalar_counts": defaultdict(int),
            "components": {c: {"loss_sum": 0., "loss_steps": 0,
                               "counts": defaultdict(int)} for c in ("sample", "proxy")}})
        dest["steps"] += 1
        dest["last_step"] = step
        for name in SCALARS:
            if name in row:
                dest["scalar_sums"][name] += finite(row[name])
                dest["scalar_counts"][name] += 1
        hier = row["hier"]
        for name, data in dest["components"].items():
            data["loss_sum"] += finite(hier[name + "_loss"])
            data["loss_steps"] += 1
            source = hier[name]
            for count in COUNTS:
                raw = source[count]
                value = int(raw)
                if value < 0 or value != raw:
                    raise ValueError("Invalid hierarchy observation count")
                data["counts"][count] += value
    result = []
    for _, dest in sorted(epochs.items()):
        dest["scalar_means"] = {k: v / dest["scalar_counts"][k]
                                for k, v in dest.pop("scalar_sums").items()}
        for data in dest["components"].values():
            counts = dict(data["counts"])
            data["counts"] = counts
            data["loss_mean_over_updates"] = data.pop("loss_sum") / data["loss_steps"]
            n, valid, active = counts["triplets"], counts["noncollision_triplets"], counts["active_triplets"]
            if n != valid + counts["collisions"] or active > valid:
                raise ValueError("Hierarchy count conservation failed")
            if any(counts[k] > valid for k in ("active_i_hinges", "active_j_hinges", "active_k_hinges")):
                raise ValueError("Active hinge count exceeds noncollision draws")
            data["collision_fraction"] = counts["collisions"] / n if n else None
            data["active_triplet_fraction_noncollision"] = active / valid if valid else None
            data["active_hinge_fraction_noncollision"] = sum(counts[k] for k in
                ("active_i_hinges", "active_j_hinges", "active_k_hinges")) / (3 * valid) if valid else None
            data["active_hinge_fraction_all_draws"] = sum(counts[k] for k in
                ("active_i_hinges", "active_j_hinges", "active_k_hinges")) / (3 * n) if n else None
        result.append(dest)
    if not result:
        raise ValueError("No complete training telemetry rows")
    return result


def validate_baseline(trajectory):
    got = {r["epoch"]: r["end_over_cap"] for r in trajectory["epochs"]}
    if any(got.get(e) != count for e, count in ((1, 229), (5, 444), (10, 502))):
        raise ValueError("Reference is not report71 fresh baseline (229/444/502 at e1/5/10)")
    if trajectory["proxy_count"] != 512 or trajectory["paired_steps"] != 2000:
        raise ValueError("Report71 baseline size/horizon mismatch")


def trajectory_rows(label, data):
    rows = []
    for e in data["epochs"]:
        row = {"condition": label, **{k: e[k] for k in ("epoch", "steps", "end_over_cap",
            "first_passages", "crossing_events", "return_events", "cumulative_ever_over_cap",
            "cumulative_ever_returned", "end_mapped_cap_count")}}
        for field in ("end_norm", "end_mapped_depth"):
            row.update({field + "_" + key: val for key, val in e[field].items()})
        for cohort, stats in e["cohorts"].items():
            for name, value in stats["means"].items():
                row[cohort + "_" + name] = value
            for name, value in stats["fractions"].items():
                row[cohort + "_fraction_" + name] = value
        rows.append(row)
    return rows


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"baseline": "#A66230", "lr_eps": "#247D91"}
    labels = {"baseline": "Report71: LR .01 / eps 1e-8", "lr_eps": "New: LR .005 / eps 1e-4"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for condition, item in summary["conditions"].items():
        trajectory = item["trajectory"]
        es = trajectory["epochs"]
        x = [r["epoch"] for r in es]
        style = dict(color=colors[condition], marker="o", markersize=3, label=labels[condition])
        for ax, field, title in ((axes[0, 0], "end_over_cap", "Outside raw cap / 512"),
                                 (axes[0, 1], "end_norm", "Raw proxy norm (median)"),
                                 (axes[0, 2], "end_mapped_depth", "Mapped depth (median)")):
            ax.plot(x, [r[field]["median"] if isinstance(r[field], dict) else r[field] for r in es], **style)
            ax.set_title(title)
        axes[1, 0].plot(x, [r["cohorts"]["all"]["means"]["exact_norm_change"] for r in es], **style)
        learning = item["learning"]
        if learning is not None:
            lx = [r["epoch"] for r in learning]
            for c, dash in (("sample", "-"), ("proxy", "--")):
                kw = dict(color=colors[condition], linestyle=dash, label=labels[condition] + " / " + c)
                axes[1, 1].plot(lx, [r["components"][c]["loss_mean_over_updates"] for r in learning], **kw)
                axes[1, 2].plot(lx, [r["components"][c]["active_hinge_fraction_noncollision"] for r in learning], **kw)
    axes[0, 0].set_ylim(-5, 525)
    axes[0, 1].axhline(2.3, color="#777777", lw=1, ls=":")
    axes[0, 2].axhline(4.6, color="#777777", lw=1, ls=":")
    axes[1, 0].axhline(0, color="#777777", lw=1)
    axes[1, 0].set_title("Exact norm change / update (all proxies)")
    axes[1, 1].set_title("HIER training loss (unweighted)")
    axes[1, 2].set_title("Active hinges / noncollision hinges")
    axes[1, 2].set_ylim(0, 1)
    for ax in axes.flat:
        ax.set_xlabel("Completed epoch")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 1].legend(fontsize=7, frameon=False)
    fig.suptitle("V9 bounded 10-epoch optimization diagnostic", fontsize=14)
    fig.text(.055, .014, "Epoch means use changing training batches. Smaller loss/cap is not independent hierarchy or classification evidence.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for extension in ("png", "svg"):
        fig.savefig(out / ("comparison." + extension), dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, default=REPORT71)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--baseline-run-dir", type=Path,
                           help="Optional private report71 main directory for full-step loss/hinge comparison")
    reference.add_argument("--baseline-learning-json", type=Path,
                           help="Anonymous epoch list exported by aggregate_learning from report71 private steps.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Use a new analysis output directory")
    legacy = load_legacy()
    legacy.torch.set_num_threads(2)
    legacy.torch.set_num_interop_threads(1)
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))["trajectory"]
    validate_baseline(baseline)
    new = legacy.aggregate_steps(args.run_dir)
    last = new["epochs"][-1]
    learning = aggregate_learning(args.run_dir / "steps.jsonl", (last["epoch"], last["last_step"]))
    if sum(e["steps"] for e in learning) != new["paired_steps"]:
        raise ValueError("Optimizer and learning telemetry have different complete step counts")
    baseline_learning = aggregate_learning(args.baseline_run_dir / "steps.jsonl", (10, 200)) if args.baseline_run_dir else None
    if args.baseline_learning_json:
        baseline_learning = json.loads(args.baseline_learning_json.read_text(encoding="utf-8"))
        if not isinstance(baseline_learning, list):
            raise ValueError("Baseline learning export must be an aggregate epoch list")
    if baseline_learning is not None and sum(e["steps"] for e in baseline_learning) != 2000:
        raise ValueError("Incomplete report71 learning reference")
    summary = {"method": "CPU aggregate comparison; no model/optimizer execution; no GPU",
        "comparison_reference": "Report71 fresh 10x200 diagnostic, not historical report69 V9",
        "variant": "Only proxy initial LR .01 -> .005 and eps 1e-8 -> 1e-4; scheduler minimum .0005 unchanged",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "legacy_analyzer_sha256": hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
        "baseline_summary_sha256": hashlib.sha256(args.baseline_summary.read_bytes()).hexdigest(),
        "baseline_learning_json_sha256": hashlib.sha256(args.baseline_learning_json.read_bytes()).hexdigest() if args.baseline_learning_json else None,
        "missing_baseline_learning": baseline_learning is None,
        "limitations": ["Independent fresh training comparison; not exact historical bitwise replay",
            "Hinge rates pool actual counts with collisions excluded in the stated denominator",
            "Training losses use changing batches, changing relation mining and Gumbel choices",
            "No validation/test evaluation and no claim of classification or morphology improvement"],
        "conditions": {"baseline": {"trajectory": baseline, "learning": baseline_learning},
                       "lr_eps": {"trajectory": new, "learning": learning}}}
    args.out.mkdir(parents=True)
    rows, losses, accounting = [], [], []
    for label, item in summary["conditions"].items():
        rows.extend(trajectory_rows(label, item["trajectory"]))
        for e in item["trajectory"]["epochs"]:
            for cohort, data in e["optimizer_components"].items():
                accounting.append({"condition": label, "epoch": e["epoch"], "cohort": cohort,
                    "observations": data["proxy_update_observations"], **data["means"]})
        for e in item["learning"] or []:
            for component, data in e["components"].items():
                losses.append({"condition": label, "epoch": e["epoch"], "steps": e["steps"],
                    "component": component, **e["scalar_means"], **data["counts"],
                    **{k: v for k, v in data.items() if k != "counts"}})
    csv_write(args.out / "trajectory_comparison.csv", rows)
    csv_write(args.out / "learning_comparison.csv", losses)
    csv_write(args.out / "optimizer_comparison.csv", accounting)
    (args.out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        allow_nan=False), encoding="utf-8")
    plot(summary, args.out)
    print(json.dumps({"steps": new["paired_steps"], "last_epoch": last["epoch"],
        "new_over_cap": last["end_over_cap"], "new_ever_returned": new["ever_returned"],
        "baseline_learning_present": baseline_learning is not None}))


if __name__ == "__main__":
    main()
