"""Export anonymous full-epoch actual-Gumbel ancestor statistics, using CPU only.

Reads existing metrics_epoch_*.json['structure']; no model, data, training log,
checkpoint, GPU, or new observation. Per-proxy count arrays are checked then
discarded. Positive endpoint gap means shallower ancestor; positive pair-minus-
triple depth means triple shallower than pair. These are strict zero thresholds,
not independent semantics evidence or margin=.1 satisfaction rates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

DOMAINS = {"all_draws": "draw_count", "noncollision": "noncollision_count",
           "active_noncollision": "active_noncollision_count"}
QUANTITIES = ("pair_depth", "triple_depth", "pair_minus_triple_depth",
              "pair_i_endpoint_gap", "pair_j_endpoint_gap", "triple_i_endpoint_gap",
              "triple_j_endpoint_gap", "triple_k_endpoint_gap", "pair_min_endpoint_gap",
              "triple_min_endpoint_gap")


def integer(value, label):
    result = int(value)
    if result != value or result < 0:
        raise ValueError("Invalid nonnegative integer: " + label)
    return result


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite scalar: " + label)
    return result


def epoch_summary(payload):
    epoch = integer(payload["epoch"], "epoch")
    usage_rows, depth_rows, components = [], [], {}
    structure = payload["structure"]
    for component in ("sample", "proxy"):
        report = structure[component]
        proxy_count = integer(report["proxy_count"], "proxy_count")
        if proxy_count < 1:
            raise ValueError("No ancestor proxies")
        counts = {key: integer(report[key], key) for key in
                  ("steps", "draw_count", "noncollision_count", "active_noncollision_count", "self_k_count")}
        if not counts["active_noncollision_count"] <= counts["noncollision_count"] <= counts["draw_count"]:
            raise ValueError("Invalid nested Gumbel cohort counts")
        base = {"epoch": epoch, "component": component, "proxy_count": proxy_count,
                "steps": counts["steps"]}
        for domain, count_key in DOMAINS.items():
            expected = counts[count_key]
            for role in ("pair", "triple"):
                raw = report["selected_ancestor_usage"][domain + "/" + role]
                per_proxy = [integer(v, "proxy_selection_count") for v in raw["counts"]]
                total = sum(per_proxy)
                if len(per_proxy) != proxy_count or total != expected or total != raw["selection_count"]:
                    raise ValueError("Ancestor usage count conservation failed")
                used = sum(v > 0 for v in per_proxy)
                entropy = -sum((v / total) * math.log(v / total) for v in per_proxy if v) if total else None
                effective = math.exp(entropy) if total else None
                if used != raw["used_proxy_count"]:
                    raise ValueError("Saved used_proxy_count disagrees with actual counts")
                if total and not math.isclose(effective, raw["effective_proxy_count"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Saved effective_proxy_count disagrees with actual entropy")
                usage_rows.append({**base, "domain": domain, "role": role,
                    "selection_count": total, "used_proxy_count": used,
                    "used_proxy_fraction": used / proxy_count,
                    "effective_proxy_count": effective})
            for quantity in QUANTITIES:
                raw = report["depth_distributions"][domain + "/" + quantity]
                n = integer(raw["count"], "moment_count")
                positive = integer(raw["positive_count"], "positive_count")
                negative = integer(raw["negative_count"], "negative_count")
                if n != expected or positive + negative > n:
                    raise ValueError("Depth moment cohort count conservation failed")
                if n:
                    mean = finite(raw["sum"], "sum") / n
                    variance = max(0., finite(raw["sum_squares"], "sum_squares") / n - mean * mean)
                    std = math.sqrt(variance)
                    fraction = positive / n
                    for value, key in ((mean, "mean"), (std, "std"), (fraction, "positive_fraction")):
                        if not math.isclose(value, finite(raw[key], key), rel_tol=1e-8, abs_tol=1e-8):
                            raise ValueError("Saved depth distribution disagrees with its moments: " + key)
                else:
                    mean = std = fraction = None
                depth_rows.append({**base, "domain": domain, "quantity": quantity,
                    "count": n, "mean": mean, "std": std, "positive_fraction": fraction})
        # Explicit whitelist: no 512 counts array or private metadata survives.
        components[component] = {"selection": str(report["selection"]),
                                 "proxy_count": proxy_count, **counts}
    return {"epoch": epoch, "components": components,
            "ancestor_usage": usage_rows, "depth_distributions": depth_rows}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    files = sorted(p for p in args.run_dir.glob("metrics_epoch_*.json")
                   if re.fullmatch(r"metrics_epoch_\d{3}\.json", p.name))
    if not files:
        raise ValueError("No completed epoch metrics found")
    epochs, inputs = [], []
    previous = 0
    for path in files:
        raw = path.read_bytes()
        item = epoch_summary(json.loads(raw))
        if item["epoch"] <= previous or item["epoch"] != int(path.stem.rsplit("_", 1)[1]):
            raise ValueError("Duplicate/out-of-order/misnamed epoch")
        previous = item["epoch"]
        epochs.append(item)
        inputs.append({"epoch": item["epoch"], "sha256": hashlib.sha256(raw).hexdigest()})
    result = {"method": "CPU extraction of saved whole-epoch actual independent hard-Gumbel ancestor statistics",
        "new_training_observations": 0, "gpu_used": False,
        "depth_units": "origin hyperbolic distance at c=.1; source already applies curvature scaling",
        "gap_sign": "Endpoint minus ancestor: positive means ancestor shallower. Pair minus triple: positive means triple shallower.",
        "positive_threshold": 0.,
        "limitation": "Exact positive fractions include floating-point near-zero differences; not margin=.1 satisfaction or independent morphology evidence.",
        "usage_semantics": "Effective proxy count = exp(entropy of pooled epoch selection counts); not mean per-step effective count.",
        "per_proxy_arrays_exported": False,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_epoch_sha256": inputs, "epochs": epochs}
    args.out.mkdir(parents=True)
    (args.out / "structure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
        allow_nan=False), encoding="utf-8")
    write_csv(args.out / "ancestor_usage.csv", [r for e in epochs for r in e["ancestor_usage"]])
    write_csv(args.out / "depth_distributions.csv", [r for e in epochs for r in e["depth_distributions"]])
    print(json.dumps({"epochs": len(epochs), "last_epoch": previous,
        "usage_rows": sum(len(e["ancestor_usage"]) for e in epochs),
        "depth_rows": sum(len(e["depth_distributions"]) for e in epochs),
        "per_proxy_arrays_exported": False}))


if __name__ == "__main__":
    main()
