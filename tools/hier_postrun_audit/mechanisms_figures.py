"""Scientific views of fixed-query diagnostics, never optimizer/model gradients.

Noise-pair bands are descriptive quantiles, not confidence intervals. Every
gradient panel names its cached-feature/parameter partial derivative scope.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _average(values):
    values = [float(value) for value in values if _finite(value)]
    return float(np.mean(values)) if values else None


def _noise_value(snapshot, component, field):
    return snapshot.get("components", {}).get(component, {}).get("noise_summary", {}).get(
        "independent_noise_pair_agreement", {}).get(field, {})


def _temporal_value(transition, component, field):
    if not transition.get("available"):
        return None
    return _average(item.get(field) for item in transition.get("same_query_same_noise", {}).get(component, [])
                    if item.get("available"))


def partial_stats(snapshot, component, target):
    """Use unweighted partials for geometry and lambda-weighted norm separately.

    Zero norms/undefined radial directions yield unavailable energy ratios; they
    do not masquerade as entirely angular gradients. Proxy loss has no whole
    dependency, so that component is explicitly not applicable for whole_ball.
    """
    source = snapshot.get("gradient_summaries", {}).get(component, [])
    if not source:
        source = snapshot.get("components", {}).get(component, {}).get("gradient_repeats", [])
    if target == "whole_ball" and source and not any(item.get("whole_dependency") for item in source):
        return {"available": False, "reason": "Proxy-only objective has no cached whole dependency",
                "not_applicable": True}
    energies, signed, norms, counts, undefined = [], [], [], [], []
    for row in source:
        unweighted = row.get("unweighted_" + target)
        # Older detail records can still be displayed but must be labelled with
        # their actual weighted identity instead of silently removing lambda.
        if not isinstance(unweighted, dict):
            continue
        total, radial = unweighted.get("gradient_norm"), unweighted.get("radial_norm")
        if _finite(total) and total > 0 and _finite(radial) and not unweighted.get("radial_direction_undefined_count", 0):
            energies.append(float(radial * radial / (total * total)))
        signed.append(unweighted.get("signed_radial_mean"))
        weighted = row.get(target, {})
        norms.append(weighted.get("gradient_norm"))
        counts.append(unweighted.get("above_tolerance_row_count"))
        undefined.append(unweighted.get("radial_direction_undefined_count"))
    return {"available": bool(source), "gradient_repeat_count": len(source),
            "radial_energy_fraction": _average(energies),
            "unweighted_signed_radial_mean": _average(signed),
            "lambda_weighted_partial_norm": _average(norms),
            "unweighted_nonzero_rows_above_tolerance": _average(counts),
            "radial_direction_undefined_rows": _average(undefined),
            "semantics": "Frozen cached whole-ball/proxy-tangent Euclidean partials before clipping. Positive signed radial derivative gives inward plain-GD displacement; no AdamW/shared-encoder/base-gradient direction is inferred."}


def _line(axis, x, y, label, color, style="-", marker="o"):
    if any(_finite(value) for value in y):
        axis.plot(list(x), [float(value) if _finite(value) else float("nan") for value in y],
                  linestyle=style, marker=marker, label=label, color=color, markersize=4)


def save_mechanism_figures(mechanisms, output_dir):
    """Return report-compatible {figures,warnings} from real mechanism summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _slug
    output = Path(output_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures, warnings, groups = [], list((mechanisms or {}).get("warnings", [])), {}
    for row in (mechanisms or {}).get("snapshots", []):
        groups.setdefault(row["run_key"], []).append(row)

    def save(fig, run, suffix, kind, title, note):
        stem = _slug(run) + "_" + suffix
        png, svg = output / (stem + ".png"), output / (stem + ".svg")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        figures.append({"run_id": run, "kind": kind, "title": title, "note": note,
                        "png": "figures/" + png.name, "svg": "figures/" + svg.name})

    for run, rows in groups.items():
        rows.sort(key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in rows]
        name = rows[0].get("display_name") or run
        transitions = [row for row in (mechanisms or {}).get("transitions", []) if row.get("run_key") == run]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
        drawn = False
        for axis, component in zip(axes, ("sample", "proxy")):
            for field, noun, color in (("pair_proxy_ids_agreement", "pair ancestor ID", "#2563eb"),
                                       ("triple_proxy_ids_agreement", "triple ancestor ID", "#d97706")):
                noise = [_noise_value(row, component, field) for row in rows]
                means = [row.get("mean") for row in noise]
                if any(_finite(value) for value in means):
                    drawn = True
                    _line(axis, epochs, means, "Different Gumbel: " + noun, color)
                    valid = [(epoch, row.get("quantiles", {})) for epoch, row in zip(epochs, noise)
                             if _finite(row.get("mean")) and _finite(row.get("quantiles", {}).get("p05"))
                             and _finite(row.get("quantiles", {}).get("p95"))]
                    if valid:
                        axis.fill_between([x for x, _ in valid], [q["p05"] for _, q in valid],
                                          [q["p95"] for _, q in valid], color=color, alpha=.12)
                tx = [row["to_epoch"] for row in transitions]
                ty = [_temporal_value(row, component, field) for row in transitions]
                _line(axis, tx, ty, "Same Gumbel, next checkpoint: " + noun, color, "--", "s")
            axis.set(title=component + " fixed-query selections", xlabel="Actual saved checkpoint epoch",
                     ylabel="Per-query selected ancestor ID agreement", ylim=(0, 1.03))
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(loc="lower left", fontsize=7)
        if drawn:
            fig.suptitle(f"{name}: query/noise versus checkpoint stability\nSame-weight noise baseline and same-query/same-noise coarse checkpoint transitions", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .9))
            save(fig, run, "fixed_query_selection_stability", "controlled_stability",
                 "Fixed-query Gumbel and checkpoint stability",
                 "Solid lines vary Gumbel noise at fixed weights; bands are descriptive noise-pair 5–95% quantiles, not confidence intervals. Dashed lines preserve query IDs and noise between actually saved checkpoints. Hard ancestor IDs are distinct from collision masks or nearest-object top-k.")
        else:
            plt.close(fig)
            warnings.append(run + ": no eligible fixed-query noise agreement, stability figure unavailable.")

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        any_gradients = False
        for ri, target in enumerate(("proxy_tangent", "whole_ball")):
            for component, color in (("sample", "#2563eb"), ("proxy", "#d97706")):
                summaries = [partial_stats(row, component, target) for row in rows]
                if any(item.get("available") for item in summaries):
                    any_gradients = True
                for axis, field in zip(axes[ri], ("radial_energy_fraction", "lambda_weighted_partial_norm",
                                                 "unweighted_signed_radial_mean")):
                    _line(axis, epochs, [item.get(field) for item in summaries], component + " objective", color)
            titles = ("Unweighted radial energy fraction", "Lambda-weighted partial gradient norm",
                      "Unweighted signed radial derivative mean")
            for axis, title in zip(axes[ri], titles):
                axis.set(title=target + "\n" + title, xlabel="Actual saved checkpoint epoch")
                axis.grid(alpha=.2)
                if axis.lines:
                    axis.legend(fontsize=8)
            axes[ri, 0].set_ylabel("||g_radial||² / ||g||²")
            axes[ri, 0].set_ylim(-.02, 1.02)
            axes[ri, 1].set_yscale("symlog", linthresh=1e-9)
            axes[ri, 1].set_ylabel("Euclidean partial-gradient norm")
            axes[ri, 2].set_ylabel("Positive gives inward plain-GD displacement")
            axes[ri, 2].axhline(0, color="#777777", linewidth=.7)
        if any_gradients:
            fig.suptitle(f"{name}: frozen-cache HIER partial derivatives\nProxy tangent and whole ball coordinates; no shared encoder, base gradient, clipping or AdamW update", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .91))
            save(fig, run, "cached_partial_gradient_geometry", "gradient_partials",
                 "Frozen-cache partial gradient geometry",
                 "Euclidean cached-feature/proxy-parameter partial derivatives of production ST HIER. Geometry uses unweighted partials; the middle column includes the recorded lambda. Zero gradient energy is unavailable. Proxy-only whole dependence is not applicable. Positive radial gradient is an inward plain-GD tendency, and does not measure AdamW/model displacement.")
        else:
            plt.close(fig)
            warnings.append(run + ": gradients were not evaluated; partial-gradient figure unavailable.")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
        any_reference = False
        for component, color in (("sample", "#2563eb"), ("proxy", "#d97706")):
            reference = [row.get("reference_counterfactual", {}).get(component, {}) for row in rows]
            median = [item.get("distance_absolute_difference_quantiles", {}).get("p50") for item in reference]
            p95 = [item.get("distance_absolute_difference_quantiles", {}).get("p95") for item in reference]
            if any(_finite(value) for value in median):
                any_reference = True
                _line(axes[0], epochs, median, component + " median", color)
                _line(axes[0], epochs, p95, component + " p95", color, "--", "s")
            for field, title, style in (("pair_proxy_ids_agreement", "pair ID", "-"),
                                        ("triple_proxy_ids_agreement", "triple ID", "--")):
                disagreements = [1. - value if value is not None else None for value in [
                    _average(item.get(field) for item in source.get("same_query_same_noise_operator_agreements", [])
                             if item.get("available")) for source in reference]]
                _line(axes[1], epochs, disagreements, component + " " + title, color, style)
        axes[0].set(title="Production versus reviewed reference distance gap",
                    xlabel="Actual saved checkpoint epoch", ylabel="|d_production - d_reference| in distance units")
        axes[1].set(title="Same-query/same-noise selected ID change",
                    xlabel="Actual saved checkpoint epoch", ylabel="Ancestor ID disagreement fraction", ylim=(0, 1.03))
        for axis in axes:
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=8)
        if any_reference:
            fig.suptitle(f"{name}: numerical operator counterfactual\nIdentical coordinates/query IDs/Gumbel noise; reference operator is not the production training objective", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .9))
            save(fig, run, "numerical_operator_counterfactual", "operator_counterfactual",
                 "Numerical operator counterfactual",
                 "Same-coordinate reviewed pmath versus actual production full-geodesic operator, with fixed query IDs and identical Gumbel seeds. Distance/operator changes may change hard ancestors. This diagnostic neither reruns original HIER nor isolates long-training causal effects.")
        else:
            plt.close(fig)
    return {"figures": figures, "warnings": warnings}
