"""Offline CPU aggregation of the private bounded V9 proxy-cause observation.

Only anonymous aggregate JSON/CSV are written. No encoder, data loader, GPU,
Gumbel replay, optimizer update, parameter export, or sample-ID export is used.
Dynamic cap cohorts are pooled by proxy-update observation count, not by taking
an unweighted mean of step means. Sparse fixed-choice gradients use the recorded
hard choices AND recorded hinge activity to avoid CPU/GPU threshold flips.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance, to_ball

DEFAULT_SNAPSHOTS = "e001_s001,e001_s005,e001_s020,e001_s100,e001_s200,e005_s200,e010_s200"
STEP_COMPONENTS = (
    "history_radial_m_step", "history_tangent_m_step", "current_radial_g_step",
    "current_tangent_g_step", "weight_decay_step", "finite_step_excess",
    "actual_radial_step", "exact_norm_change", "prediction_error_norm",
)
ARRAY_METRICS = (
    "signed_radial_gradient", "gradient_norm", "tangent_gradient_norm",
    "actual_radial_step", "actual_angular_step", "exact_norm_change", "finite_step_excess",
)
CAP = 2.3


def describe(value):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(a):
        return {"n": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite input statistic")
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def jsonl(path):
    """Ignore an unfinished final write while a diagnostic is still running."""
    with path.open(encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n") and not stream.read(1):
                    break
                raise


class Pool:
    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)
        self.flags = defaultdict(int)

    def arrays(self, values, mask, gradient_tol=1e-10, step_tol=1e-12):
        n = int(mask.sum())
        self.n += n
        if not n:
            return
        for key in ARRAY_METRICS:
            self.sums[key] += float(values[key][mask].sum())
        g = values["signed_radial_gradient"][mask]
        actual = values["exact_norm_change"][mask] > step_tol
        radial = values["actual_radial_step"][mask] > step_tol
        wants = g < -gradient_tol
        inward = g > gradient_tol
        flags = {"gradient_outward": wants, "gradient_inward": inward,
                 "gradient_near_zero": ~(wants | inward), "actual_norm_outward": actual,
                 "actual_radial_outward": radial,
                 "gradient_outward_and_norm_outward": wants & actual,
                 "gradient_not_outward_and_norm_outward": ~wants & actual,
                 "gradient_outward_and_norm_not_outward": wants & ~actual,
                 "gradient_not_outward_and_norm_not_outward": ~wants & ~actual}
        for key, value in flags.items():
            self.flags[key] += int(value.sum())

    def report(self):
        return {"proxy_update_observations": self.n,
                "means": {key: value / self.n for key, value in self.sums.items()} if self.n else {},
                "counts": dict(self.flags),
                "fractions": {key: value / self.n for key, value in self.flags.items()} if self.n else {}}


def aggregate_steps(run_dir, gradient_tol=1e-10, step_tol=1e-12):
    steps = {(int(r["epoch"]), int(r["step"])): r for r in jsonl(run_dir / "proxy_cause_steps.jsonl")}
    epochs, pools, components = {}, defaultdict(Pool), {}
    ever_cap = ever_return = last_after = None
    total_events = defaultdict(int)
    last_key = (0, 0)
    matched = 0
    for raw in jsonl(run_dir / "proxy_cause_per_proxy.jsonl"):
        key = (int(raw["epoch"]), int(raw["step"]))
        if key not in steps:
            continue
        if key <= last_key:
            raise ValueError("Proxy rows are duplicated or not chronological")
        last_key = key
        values = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()
                  if isinstance(v, list)}
        before, after = values["before_norm"], values["after_norm"]
        if before.ndim != 1 or not len(before) or any(v.shape != before.shape for v in values.values()):
            raise ValueError("Per-proxy row shape differs from the runner schema")
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite per-proxy row")
        if ever_cap is None:
            ever_cap = before > CAP
            ever_return = np.zeros(len(before), dtype=bool)
            initial_cap = int(ever_cap.sum())
        if len(before) != len(ever_cap):
            raise ValueError("Proxy count changed")
        if last_after is not None and not np.allclose(last_after, before, atol=2e-6, rtol=0):
            raise ValueError("Adjacent actual proxy norms do not match")
        last_after = after.copy()
        e = key[0]
        record = epochs.setdefault(e, {"epoch": e, "steps": 0,
            "first_step": key[1], "first_passages": 0, "crossing_events": 0,
            "return_events": 0, "initial_over_cap": int((before > CAP).sum()),
            "start_norm": describe(before), "start_mapped_depth": describe(values["mapped_depth_before"]),
            "maximum_after_over_cap": 0, "accounting_max_coordinate_error": 0.,
            "unique_returned_proxies": set(), "unique_crossing_proxies": set()})
        crossing = (before <= CAP) & (after > CAP)
        returns = (before > CAP) & (after <= CAP)
        first = ~ever_cap & (after > CAP)
        ever_cap |= after > CAP
        ever_return |= returns
        counts = {"first_passages": int(first.sum()), "crossing_events": int(crossing.sum()),
                  "return_events": int(returns.sum())}
        for name, count in counts.items():
            record[name] += count
            total_events[name] += count
        record["unique_returned_proxies"].update(np.flatnonzero(returns).tolist())
        record["unique_crossing_proxies"].update(np.flatnonzero(crossing).tolist())
        record.update(steps=record["steps"] + 1, last_step=key[1],
            end_norm=describe(after), end_mapped_depth=describe(values["mapped_depth_after"]),
            end_over_cap=int((after > CAP).sum()),
            end_mapped_cap_count=int((values["mapped_depth_after"] >= 4.5999).sum()),
            cumulative_ever_over_cap=int(ever_cap.sum()), cumulative_ever_returned=int(ever_return.sum()))
        record["maximum_after_over_cap"] = max(record["maximum_after_over_cap"], record["end_over_cap"])
        report = steps[key]
        record["accounting_max_coordinate_error"] = max(record["accounting_max_coordinate_error"],
            report["all"]["prediction_error_max_coordinate"]["max"])
        masks = {"all": np.ones(len(before), dtype=bool), "inside": before <= CAP, "outside": before > CAP}
        for cohort, mask in masks.items():
            for epoch_key in (e, 0):
                pools[(epoch_key, cohort)].arrays(values, mask, gradient_tol, step_tol)
                accumulator = components.setdefault((epoch_key, cohort), {"n": 0, "sums": defaultdict(float)})
                cohort_summary = report["all"] if cohort == "all" else report["by_before_cap"][cohort]
                n = len(before) if cohort == "all" else cohort_summary["count"]
                if n != int(mask.sum()):
                    raise ValueError("Step summary cohort count differs from per-proxy rows")
                accumulator["n"] += n
                if n:
                    for name in STEP_COMPONENTS:
                        accumulator["sums"][name] += cohort_summary[name]["mean"] * n
        matched += 1
    if not matched:
        raise ValueError("No paired complete proxy observation rows")
    output_epochs = []
    for e, row in sorted(epochs.items()):
        for name in ("unique_returned_proxies", "unique_crossing_proxies"):
            row[name] = len(row[name])
        row["cohorts"] = {c: pools[(e, c)].report() for c in ("all", "inside", "outside")}
        row["optimizer_components"] = component_report(components, e)
        output_epochs.append(row)
    return {"paired_steps": matched, "step_summaries_read": len(steps),
            "unpaired_step_summaries": len(steps) - matched, "proxy_count": len(ever_cap),
            "initial_over_cap": initial_cap, "ever_over_cap": int(ever_cap.sum()),
            "ever_returned": int(ever_return.sum()), **dict(total_events), "epochs": output_epochs,
            "pooled_cohorts": {c: pools[(0, c)].report() for c in ("all", "inside", "outside")},
            "pooled_optimizer_components": component_report(components, 0)}


def component_report(components, epoch):
    result = {}
    for cohort in ("all", "inside", "outside"):
        item = components.get((epoch, cohort), {"n": 0, "sums": {}})
        n = item["n"]
        result[cohort] = {"proxy_update_observations": n,
            "means": {k: v / n for k, v in item["sums"].items()} if n else {}}
    return result


def gradient_report(p, gradient, actual, mask, gradient_tol, step_tol):
    p, gradient = p.double(), gradient.double()
    u = p / p.norm(dim=-1, keepdim=True)
    radial = (gradient * u).sum(-1)
    tangential = gradient - radial[:, None] * u
    n = int(mask.sum())
    if not n:
        return {"n": 0}
    wants = radial < -gradient_tol
    norms_out = actual["exact_norm_change"].double() > step_tol
    radial_out = actual["actual_radial_step"].double() > step_tol
    counts = {"gradient_outward": int((wants & mask).sum()),
              "gradient_inward": int(((radial > gradient_tol) & mask).sum()),
              "gradient_near_zero": int(((radial.abs() <= gradient_tol) & mask).sum()),
              "actual_norm_outward": int((norms_out & mask).sum()),
              "actual_radial_outward": int((radial_out & mask).sum()),
              "gradient_outward_and_norm_outward": int((wants & norms_out & mask).sum()),
              "gradient_outward_and_norm_not_outward": int((wants & ~norms_out & mask).sum()),
              "gradient_not_outward_and_norm_outward": int((~wants & norms_out & mask).sum()),
              "gradient_not_outward_and_norm_not_outward": int((~wants & ~norms_out & mask).sum())}
    return {"n": n, "signed_radial": describe(radial[mask].numpy()),
            "angular_norm": describe(tangential[mask].norm(dim=-1).numpy()),
            "gradient_norm": describe(gradient[mask].norm(dim=-1).numpy()),
            "actual_exact_norm_change": describe(actual["exact_norm_change"][mask].numpy()),
            "actual_radial_step": describe(actual["actual_radial_step"][mask].numpy()),
            "counts": counts, "fractions": {key: value / n for key, value in counts.items()}}


def fixed_choice_loss(distances, recorded):
    """Same selected IDs and actual hinge activity; no routing derivatives."""
    triplets = recorded["triplets"].long()
    if not len(triplets):
        return distances.sum() * 0, distances.sum() * 0, 0
    i, j, k = triplets.unbind(-1)
    detail = recorded["details"]
    a, b = detail["pair_proxy_idx"].long(), detail["triple_proxy_idx"].long()
    margin = float(recorded["margin"])
    raw = torch.stack((distances[i, a] - distances[i, b] + margin,
                       distances[j, a] - distances[j, b] + margin,
                       distances[k, b] - distances[k, a] + margin), dim=-1)
    active = detail["masked_hinges"] > 0
    valid = a != b
    reconstructed_activity = (raw.detach() > 0) & valid[:, None]
    disagreements = int((active != reconstructed_activity).sum())
    fixed = (raw * active.to(raw)).sum(-1).mean()
    relu_scalar = (raw.relu().sum(-1) * valid.to(raw)).mean()
    return fixed, relu_scalar, disagreements


def analyze_snapshot(path, gradient_tol=1e-10, step_tol=1e-12):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    same, actual = payload["same_forward"], payload["per_proxy"]
    raw = same["proxy_before"].detach()
    parameter_before = payload["actual_optimizer_step"]["parameter"]
    if not torch.equal(raw, parameter_before):
        raise ValueError("Sparse same-forward proxies differ from actual optimizer pre-state")
    p = raw.clone().requires_grad_(True)
    radii = p.detach().double().norm(dim=-1)
    masks = {"all": torch.ones(len(p), dtype=torch.bool), "inside": radii <= CAP, "outside": radii > CAP}
    result = {"epoch": int(payload["epoch"]), "step": int(payload["step"]),
              "proxy_count": len(p), "parameter_norm": describe(radii.numpy()),
              "hier_weight": float(same["hier_weight"]),
              "recorded_component_sum_vs_actual_synced_gradient_max_abs":
              float(same["weighted_component_sum_vs_actual_synced_gradient_max_abs"]),
              "components": {}}
    for name in ("sample", "proxy"):
        proxies = to_ball(p)
        mu = same["hier_mu"].detach() if name == "sample" else proxies
        distances = official_distance(mu, proxies)
        recorded = same[name + "_ghhc"]
        saved_distance = recorded["distances"]
        difference = (distances.detach().double() - saved_distance.double()).abs()
        tolerance = 2e-4 + 2e-4 * saved_distance.double().abs()
        fixed, relu_scalar, activity_differences = fixed_choice_loss(distances, recorded)
        weighted_fixed = float(same["hier_weight"]) * fixed
        g_fixed, = torch.autograd.grad(weighted_fixed, p)
        g_observed = same["weighted_component_gradients"][name + "_loss"].detach()
        g_routing = g_observed - g_fixed.detach()
        gradients = {"full_st": g_observed, "fixed_choice": g_fixed.detach(), "routing_residual": g_routing}
        flat_fixed, flat_observed = g_fixed.detach().double().flatten(), g_observed.double().flatten()
        denominator = flat_fixed.norm() * flat_observed.norm()
        row = {"distances": {"max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose_atol_2e4_rtol_2e4": bool((difference <= tolerance).all()),
            "entry_count": difference.numel()},
            "unweighted_recorded_loss": float(recorded["loss"]),
            "unweighted_fixed_active_scalar": float(fixed.detach()),
            "unweighted_recomputed_relu_scalar": float(relu_scalar.detach()),
            "hinge_activity_disagreements_due_to_recomputation": activity_differences,
            "fixed_vs_full_cosine": float(torch.dot(flat_fixed, flat_observed) / denominator) if float(denominator) else None,
            "fixed_vs_full_norm_ratio": float(flat_fixed.norm() / flat_observed.norm()) if float(flat_observed.norm()) else None,
            "gradient_decomposition_residual_max_abs": float((g_observed - g_fixed.detach() - g_routing).abs().max()),
            "gradients": {key: {cohort: gradient_report(p.detach(), gradient, actual, mask,
                gradient_tol, step_tol) for cohort, mask in masks.items()} for key, gradient in gradients.items()}}
        result["components"][name] = row
        del distances, fixed, weighted_fixed, g_fixed, proxies
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_csvs(out, summary):
    epochs, cohorts, components, sparse = [], [], [], []
    for row in summary["trajectory"]["epochs"]:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for key in ("start_norm", "end_norm", "start_mapped_depth", "end_mapped_depth"):
            flat.update({key + "_" + k: v for k, v in row[key].items()})
        epochs.append(flat)
        for cohort, data in row["cohorts"].items():
            cohorts.append({"epoch": row["epoch"], "cohort": cohort,
                "observations": data["proxy_update_observations"], **data["means"], **data["fractions"]})
        for cohort, data in row["optimizer_components"].items():
            components.append({"epoch": row["epoch"], "cohort": cohort,
                "observations": data["proxy_update_observations"], **data["means"]})
    for row in summary["sparse_snapshots"]:
        for component, data in row["components"].items():
            for kind, groups in data["gradients"].items():
                for cohort, group in groups.items():
                    flat = {"epoch": row["epoch"], "step": row["step"], "component": component,
                            "gradient_kind": kind, "cohort": cohort, "n": group["n"]}
                    if group["n"]:
                        flat.update({key + "_mean": group[key]["mean"] for key in
                            ("signed_radial", "angular_norm", "gradient_norm", "actual_exact_norm_change", "actual_radial_step")})
                        flat.update(group["fractions"])
                    sparse.append(flat)
    for filename, rows in (("epochs.csv", epochs), ("cohorts.csv", cohorts),
                           ("optimizer_components.csv", components), ("sparse_gradients.csv", sparse)):
        write_csv(out / filename, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS,
                        help="Comma-separated e001_s001 labels; use 'none' to skip sparse backward")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-10)
    parser.add_argument("--step-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 2:
        parser.error("CPU analysis allows at most two threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.out.exists():
        raise FileExistsError("Output directory already exists; use a new destination")
    args.out.mkdir(parents=True)
    labels = [] if args.snapshots.lower() == "none" else args.snapshots.split(",")
    if any(not re.fullmatch(r"e\d{3}_s\d{3}", label) for label in labels):
        parser.error("Snapshot labels must have form e001_s001")
    trajectory = aggregate_steps(args.run_dir, args.gradient_tolerance, args.step_tolerance)
    summary = {"method": "Offline CPU; aggregate exports only; no encoder or optimizer updates",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "cpu_threads": args.threads, "gpu_used": False,
        "cap_parameter_norm": CAP, "mapped_cap_depth_threshold": 4.5999,
        "gradient_sign_tolerance": args.gradient_tolerance, "step_sign_tolerance": args.step_tolerance,
        "cohort_semantics": "Inside <=2.3 and outside >2.3 before each actual update; count-weighted dynamic cohorts",
        "component_semantics": "Algebraic contributions share actual Adam second moment; finite_step_excess bridges radial displacement and exact norm change; not independent interventions",
        "sparse_semantics": "Observed full-ST component minus reconstructed fixed-choice/recorded-activity gradient; weight=.5 from snapshot; no routing replay",
        "trajectory": trajectory, "sparse_snapshots": [], "unavailable_snapshots": []}
    for label in labels:
        path = args.run_dir / ("proxy_cause_" + label + ".pth")
        if path.exists():
            summary["sparse_snapshots"].append(analyze_snapshot(path, args.gradient_tolerance, args.step_tolerance))
        else:
            summary["unavailable_snapshots"].append(label)
    export_csvs(args.out, summary)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        allow_nan=False), encoding="utf-8")
    print(json.dumps({"paired_steps": trajectory["paired_steps"], "completed_epoch_observations":
        len(trajectory["epochs"]), "sparse_snapshots": len(summary["sparse_snapshots"]),
        "anonymous_files_written": 5}, allow_nan=False))


if __name__ == "__main__":
    main()
