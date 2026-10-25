"""Static scientific figures from the public Audit85 aggregate tables only.

Usage: python plot.py --analysis <statistics> --out <figures> --scope primary
Use --scope auxiliary to present historical controls separately. No weights,
coordinates, object IDs, model, GPU or training inputs are loaded here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LABELS = {
    "TS13": "TS13 / cycle H0", "TS14": "TS14 / cycle A0",
    "B05_A1": "B05 + A1", "NATIVE_A1": "Native + A1",
    "new_b05": "B05 + A1", "new_native": "Native + A1",
    "TS06_B05_H0": "TS06 / B05 H0", "TS01_HIST": "TS01 / native HIST",
    "TS11": "TS11 / S22 H0", "TS12": "TS12 / S22 A0",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    if not Path(path).is_file() or not Path(path).stat().st_size:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=np.nan):
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else default


def vector(items, key):
    return np.asarray([number(row, key) for row in items], dtype=np.float64)


def percentage(value):
    return np.nan if value is None else 100 * value


def label(spec, epoch=False):
    text = LABELS.get(spec["run_id"], LABELS.get(spec.get("key"), spec["run_id"]))
    return text + (f" (e{spec['epoch']})" if epoch else "")


def validate_distributions(histogram, cdf):
    for key in {(r["observation_id"], r["domain"]) for r in histogram}:
        selected = [r for r in histogram if (r["observation_id"], r["domain"]) == key]
        total = vector(selected, "percent").sum()
        if not np.isclose(total, 100., atol=1e-7):
            raise ValueError(f"Histogram percentages do not sum to100: {key}: {total}")
    for key in {(r["observation_id"], r["domain"]) for r in cdf}:
        selected = sorted((r for r in cdf if (r["observation_id"], r["domain"]) == key),
                          key=lambda row: number(row, "depth"))
        values = vector(selected, "cumulative_percent")
        if np.any(np.diff(values) < -1e-9) or values.min() < 0 or not np.isclose(values[-1], 100.):
            raise ValueError(f"Invalid ECDF aggregate: {key}")


class Figures:
    def __init__(self, analysis, out, scope="primary"):
        self.analysis, self.out, self.scope = Path(analysis), Path(out), scope
        self.summary = json.loads((self.analysis / "summary.json").read_text(encoding="utf-8"))
        if self.summary.get("status") != "completed":
            raise ValueError("Plot only a completed immutable statistics export")
        all_specs = self.summary["observations"]
        has_primary = any(item.get("primary") for item in all_specs)
        self.specs = [s for s in all_specs if scope == "all" or
                      (bool(s.get("primary")) if scope == "primary" else not s.get("primary"))]
        if scope == "primary" and not has_primary:
            self.specs = all_specs
        if not self.specs:
            raise ValueError("No observations in the requested plot scope")
        self.ids = {s["observation_id"] for s in self.specs}
        self.runs = list(dict.fromkeys(s["run_id"] for s in self.specs))
        present_stages = {s["stage"] for s in self.specs}
        self.stages = [s for s in ("best", "e300") if s in present_stages]
        self.stages += sorted(present_stages - set(self.stages))
        self.tables = {}
        for name in ("depth_histograms", "depth_cdf", "class_intra_angles", "class_pair_angles",
                     "class_centroid_angles", "representative_pair_angles", "representative_pair_histograms",
                     "proxy_stats", "proxy_depth_bands"):
            self.tables[name] = [r for r in rows(self.analysis / (name + ".csv")) if r["observation_id"] in self.ids]
        validate_distributions(self.tables["depth_histograms"], self.tables["depth_cdf"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.files = []
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
            "svg.fonttype": "none", "savefig.facecolor": "white"})

    def specs_for(self, stage=None, proxies=False):
        return [s for s in self.specs if (stage is None or s["stage"] == stage)
                and (not proxies or s["proxies"]["available"])]

    def selected(self, name, observation, **filters):
        return [r for r in self.tables[name] if r["observation_id"] == observation
                and all(str(r.get(key)) == str(value) for key, value in filters.items())]

    def color(self, spec):
        return COLORS[self.runs.index(spec["run_id"]) % len(COLORS)]

    def finish(self, fig, name, title, note=""):
        fig.suptitle(title, fontsize=13, y=.995)
        if note:
            fig.text(.01, .008, note, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, .045 if note else .015, 1, .965))
        for extension in ("png", "svg"):
            path = self.out / (name + "." + extension)
            fig.savefig(path, dpi=190, bbox_inches="tight")
            self.files.append({"file": path.name, "sha256": digest(path)})
        plt.close(fig)

    @staticmethod
    def no_data(ax, text="N/A: no proxy population"):
        ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", color=".4")

    def depth_figures(self):
        curvatures = sorted({s["native_c"] for s in self.specs})
        fig, axes = plt.subplots(len(curvatures), len(self.stages), figsize=(6.2 * len(self.stages), 3.4 * len(curvatures)), squeeze=False)
        for i, c in enumerate(curvatures):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]
                candidates = [s for s in self.specs_for(stage) if s["native_c"] == c]
                for spec in candidates:
                    data = self.selected("depth_histograms", spec["observation_id"], domain="native_whole")
                    ax.step(vector(data, "bin_center"), vector(data, "percent"), where="mid", color=self.color(spec), label=label(spec, True))
                ax.set(title=f"{stage}; native c={c:g}", xlabel="Origin hyperbolic distance", ylabel="Whole objects per0.1 bin (%)")
                ax.axvline(4.6, color=".5", lw=.8, ls=":")
                ax.grid(alpha=.16)
                if candidates: ax.legend()
                else: self.no_data(ax, "No observation at this curvature/stage")
        self.finish(fig, "01_native_whole_depth_percentage", "Native whole depth distributions",
                    "Each observation is normalized by its own whole population. c1 and c0.1 are separate metric domains; dotted line:4.6 reference.")
        fig, axes = plt.subplots(1, len(self.stages), figsize=(6.2 * len(self.stages), 4.1), squeeze=False)
        for ax, stage in zip(axes[0], self.stages):
            candidates = self.specs_for(stage, proxies=True)
            for spec in candidates:
                data = self.selected("depth_histograms", spec["observation_id"], domain="proxy")
                ax.step(vector(data, "bin_center"), vector(data, "percent"), where="mid", color=self.color(spec), label=label(spec, True))
            ax.axvline(4.6, color=".25", ls="--", label="Constructed HIER whole shell4.6")
            ax.set(title=stage, xlabel="Origin distance (c0.1)", ylabel="Proxies per0.1 bin (%)")
            ax.grid(alpha=.16); ax.legend()
            if not candidates: self.no_data(ax)
        self.finish(fig, "02_proxy_depth_percentage", "Proxy depth distributions and the constructed whole shell",
                    "Each proxy population sums to100%. No KDE smoothing; H0 has no proxies and is omitted, never assigned depth0.")
        panels = [("native_whole", c) for c in curvatures] + [("proxy", .1)]
        fig, axes = plt.subplots(len(panels), len(self.stages), figsize=(6.2 * len(self.stages), 3.1 * len(panels)), squeeze=False)
        for i, (domain, c) in enumerate(panels):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]; count = 0
                for spec in self.specs_for(stage, proxies=domain == "proxy"):
                    if domain == "native_whole" and spec["native_c"] != c: continue
                    data = self.selected("depth_cdf", spec["observation_id"], domain=domain)
                    ax.step(vector(data, "depth"), vector(data, "cumulative_percent"), where="post", color=self.color(spec), label=label(spec, True))
                    count += 1
                ax.axvline(4.6, color=".5", lw=.8, ls="--")
                ax.set(title=f"{stage}; {domain.replace('_', ' ')}; c={c:g}", xlabel="Origin hyperbolic distance", ylabel="Cumulative population (%)", ylim=(0, 102))
                ax.grid(alpha=.16)
                if count: ax.legend(loc="lower right")
                else: self.no_data(ax)
        self.finish(fig, "03_depth_ecdf", "Depth ECDF: bin-independent population comparison",
                    "CDFs use the saved depth threshold grid (0.025); constructed HIER whole views are at4.6. Native c1 is not compared to proxies as ancestry.")

    def class_figures(self):
        names = self.summary["class_names"]
        fig, axes = plt.subplots(len(self.stages), 1, figsize=(15, 4 * len(self.stages)), squeeze=False)
        for ax, stage in zip(axes[:, 0], self.stages):
            for spec in self.specs_for(stage):
                data = sorted(self.selected("class_intra_angles", spec["observation_id"]), key=lambda row: int(row["class_id"]))
                x = vector(data, "class_id")
                ax.plot(x, vector(data, "median"), color=self.color(spec), marker=".", lw=1.2, label=label(spec, True))
                ax.fill_between(x, vector(data, "p10"), vector(data, "p90"), color=self.color(spec), alpha=.11)
            ax.set(title=f"{stage}: median; shading P10-P90", ylabel="Within-class pair angle (degrees)", xticks=np.arange(len(names)))
            ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=.2); ax.legend(ncol=2)
        self.finish(fig, "04_class_intra_angles", "All40 classes: exact within-class sample-pair angles",
                    "Each class has its own exact pair denominator. Shading is a distribution interval, not a confidence interval or multi-seed variability.")
        fig, axes = plt.subplots(len(self.runs), len(self.stages), figsize=(7.8 * len(self.stages), 6.4 * len(self.runs)), squeeze=False)
        heat = None
        for i, run in enumerate(self.runs):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]
                spec = next((s for s in self.specs if s["run_id"] == run and s["stage"] == stage), None)
                if spec is None:
                    self.no_data(ax, "No checkpoint"); continue
                matrix = np.zeros((len(names), len(names)))
                for row in self.selected("class_centroid_angles", spec["observation_id"]):
                    a, b = int(row["class_a"]), int(row["class_b"])
                    matrix[a, b] = matrix[b, a] = number(row, "angle_degrees")
                heat = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=180, interpolation="nearest")
                ax.set(title=f"{label(spec, True)} / {stage}", xticks=np.arange(len(names)), yticks=np.arange(len(names)))
                ax.set_xticklabels(names, rotation=90, fontsize=5.5)
                ax.set_yticklabels(names, fontsize=5.5)
        if heat is not None:
            fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=.45, pad=.015, label="Center angle (degrees)")
        # Keep a dedicated colorbar space, rather than allowing tight_layout
        # to move the shared colorbar on top of the last matrix.
        fig.suptitle("Direction-center angles: common0-180 degree scale", fontsize=13, y=.995)
        fig.subplots_adjust(left=.09, right=.89, bottom=.06, top=.97, hspace=.38, wspace=.35)
        for ext in ("png", "svg"):
            path = self.out / ("05_class_centroid_angles." + ext)
            fig.savefig(path, dpi=190, bbox_inches="tight")
            self.files.append({"file": path.name, "sha256": digest(path)})
        plt.close(fig)
        fig, axes = plt.subplots(len(self.stages), 1, figsize=(15, 3.8 * len(self.stages)), squeeze=False)
        for ax, stage in zip(axes[:, 0], self.stages):
            for spec in self.specs_for(stage):
                data = sorted(self.selected("class_intra_angles", spec["observation_id"]), key=lambda row: int(row["class_id"]))
                ax.plot(vector(data, "class_id"), vector(data, "cone_radius_p95"), color=self.color(spec), marker=".", label=label(spec, True))
            ax.set(title=f"{stage}: P95 sample-to-own-center cone radius", ylabel="Cone radius (degrees)", xticks=np.arange(len(names)))
            ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=.2); ax.legend(ncol=2)
        self.finish(fig, "04b_class_cone_radii", "Class directional cone radii",
                    "The percentile cone describes observed directions. It is not the classifier decision boundary, and cone overlap does not imply misclassification.")

    def representative_figures(self):
        colors = ["#0072B2", "#009E73", "#D55E00"]
        policy = self.summary["representative_pairs_policy"]
        for pair in policy["pairs"]:
            a, b = pair["class_a"], pair["class_b"]
            fig, axes = plt.subplots(2, len(self.stages), figsize=(7.1 * len(self.stages), 9), squeeze=False)
            for j, stage in enumerate(self.stages):
                ax, interval = axes[:, j]
                stagespecs = self.specs_for(stage)
                ylabels = []
                for i, spec in enumerate(stagespecs):
                    data = self.selected("representative_pair_histograms", spec["observation_id"], class_a=a, class_b=b)
                    ax.step(vector(data, "bin_right"), np.cumsum(vector(data, "percent")), where="post", color=self.color(spec), label=label(spec, True))
                    cross = self.selected("representative_pair_angles", spec["observation_id"], class_a=a, class_b=b)[0]
                    left = self.selected("class_intra_angles", spec["observation_id"], class_id=a)[0]
                    right = self.selected("class_intra_angles", spec["observation_id"], class_id=b)[0]
                    for position, row in enumerate((left, right, cross)):
                        median, p10, p90 = [number(row, k) for k in ("median", "p10", "p90")]
                        interval.errorbar(median, i + (position - 1) * .22,
                            xerr=[[median - p10], [p90 - median]], fmt="o", ms=4, capsize=2,
                            color=colors[position])
                    ylabels.append(label(spec))
                    interval.text(.98, (i + .55) / max(1, len(stagespecs)),
                        f"center {number(cross,'centroid_angle'):.1f}; cone gap95 {number(cross,'cone_gap_p95'):.1f}",
                        transform=interval.transAxes, ha="right", va="center", fontsize=7, color=".3")
                ax.set(title=f"{stage}: all cross-class sample pairs", xlabel="Cross-class angle (degrees)", ylabel="Cumulative pairs (%)", xlim=(0, 180), ylim=(0, 102))
                ax.grid(alpha=.17); ax.legend(loc="lower right")
                interval.set(title=f"{stage}: exact P10 / median / P90 intervals", xlabel="Sample-pair angle (degrees)", yticks=np.arange(len(stagespecs)), ylim=(-.6, len(stagespecs) - .3))
                interval.set_yticklabels(ylabels)
                interval.grid(axis="x", alpha=.17)
                interval.legend(handles=[Line2D([], [], color=c, marker="o", label=t) for c, t in zip(colors, (pair["name_a"] + " within", pair["name_b"] + " within", "between classes"))], loc="upper left", fontsize=7)
            self.finish(fig, f"06_pair_{pair['name_a']}_vs_{pair['name_b']}",
                        f"Fixed comparison: {pair['name_a']} vs {pair['name_b']}",
                        "Same class pair in every model. Bottom intervals are distributions, not error bars across seeds. Cone gap95 = center angle - both P95 radii.")

    def proxy_figures(self):
        proxy_runs = [run for run in self.runs if any(s["run_id"] == run and s["proxies"]["available"] for s in self.specs)]
        columns = max(1, len(proxy_runs))
        fig, axes = plt.subplots(len(self.stages), columns, figsize=(5.2 * columns, 3.7 * len(self.stages)), squeeze=False)
        scatter = None
        for i, stage in enumerate(self.stages):
            for j, run in enumerate(proxy_runs or [None]):
                ax = axes[i, j]
                spec = next((s for s in self.specs_for(stage, proxies=True) if s["run_id"] == run), None)
                if spec is None:
                    self.no_data(ax); continue
                data = self.selected("proxy_stats", spec["observation_id"])
                scatter = ax.scatter(vector(data, "depth"), vector(data, "angular_nn32_entropy_nats"), c=vector(data, "angular_nn32_purity"), cmap="viridis", vmin=0, vmax=1, s=10, alpha=.55, edgecolors="none")
                correlation = spec["proxies"]["depth_vs_class_breadth"]
                rho = f"rho={correlation['spearman_depth_vs_nn32_entropy']:.3f}" if correlation.get("available") else "depth/breadth nearly constant"
                ax.set(title=f"{label(spec, True)} / {stage}\n{rho}", xlabel="Proxy origin distance (c0.1)", ylabel="NN32 label entropy (nats)")
                ax.axvline(4.6, color=".45", lw=.8, ls="--"); ax.grid(alpha=.15)
        if scatter is not None:
            fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=.6, pad=.015, label="NN32 dominant-class fraction")
        fig.suptitle("Proxy depth and neighborhood class breadth", y=.995, fontsize=13)
        fig.subplots_adjust(left=.06, right=.88, bottom=.09, top=.9, wspace=.32, hspace=.5)
        for ext in ("png", "svg"):
            path = self.out / ("07_proxy_depth_class_breadth." + ext)
            fig.savefig(path, dpi=190, bbox_inches="tight")
            self.files.append({"file": path.name, "sha256": digest(path)})
        plt.close(fig)
        fig, axes = plt.subplots(2, len(self.stages), figsize=(6.8 * len(self.stages), 7.8), squeeze=False)
        order_keys = [("pair_shallower_than_both_endpoints", "pair below endpoints"),
                      ("triple_shallower_than_all_endpoints", "triple below endpoints"),
                      ("triple_shallower_than_pair", "triple below pair")]
        for i, domain in enumerate(("sample", "proxy")):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]; specs = self.specs_for(stage, proxies=True)
                for bar, (key, text) in enumerate(order_keys):
                    values = [percentage(s["proxies"]["frozen_orders"][domain]["domains"]["noncollision"]["tolerance"]["0.001"][key]) for s in specs]
                    ax.bar(np.arange(len(specs)) + (bar - 1) * .24, values, .23, color=COLORS[bar], label=text)
                ax.set(title=f"{stage}: {domain} endpoints", ylabel="Noncollision draws with strict order (%)", ylim=(0, 103), xticks=np.arange(len(specs)))
                ax.set_xticklabels([label(s) for s in specs], rotation=12, ha="right")
                if specs: ax.legend(fontsize=7)
                else: self.no_data(ax)
                ax.grid(axis="y", alpha=.15)
        self.finish(fig, "08_frozen_ancestor_orders", "Frozen deterministic minimax ancestry: depth tolerance0.001",
                    "Each model mines its own graph on common fixed object batches. This is not stochastic training Gumbel replay; noncollision does not exclude endpoint self-selection.")
        fig, axes = plt.subplots(1, len(self.stages), figsize=(6.8 * len(self.stages), 4.2), squeeze=False)
        collision_fields = [("sample", "collision_fraction", "sample collision"), ("proxy", "collision_fraction", "proxy collision"),
                            ("proxy", "pair_equals_an_endpoint_fraction", "proxy pair=self endpoint"),
                            ("proxy", "triple_equals_an_endpoint_fraction", "proxy triple=self endpoint")]
        for ax, stage in zip(axes[0], self.stages):
            specs = self.specs_for(stage, proxies=True)
            for i, (domain, key, text) in enumerate(collision_fields):
                values = [percentage(s["proxies"]["frozen_orders"][domain][key]) for s in specs]
                ax.bar(np.arange(len(specs)) + (i - 1.5) * .19, values, .18, color=COLORS[i], label=text)
            ax.set(title=stage, ylabel="Frozen draws (%)", ylim=(0, 103), xticks=np.arange(len(specs)))
            ax.set_xticklabels([label(s) for s in specs], rotation=12, ha="right")
            if specs: ax.legend(fontsize=7)
            else: self.no_data(ax)
            ax.grid(axis="y", alpha=.15)
        self.finish(fig, "08b_frozen_collisions_endpoint_identity", "Frozen ancestor collisions and endpoint self-selection",
                    "All-draw denominators, separate from the noncollision depth-order figure. A selected endpoint is not evidence of a distinct higher-level ancestor.")
        self.usage_figures()

    def usage_figures(self):
        fig, axes = plt.subplots(2, len(self.stages), figsize=(6.5 * len(self.stages), 7.1), squeeze=False)
        for i, field in enumerate(("sample_usage", "proxy_usage")):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]; plotted = 0
                for spec in self.specs_for(stage, proxies=True):
                    if not spec["proxies"]["usage"][field]["available"]: continue
                    data = self.selected("proxy_stats", spec["observation_id"])
                    counts = np.sort(vector(data, field))[::-1]
                    if counts.sum() <= 0: continue
                    ax.plot(np.arange(1, len(counts) + 1), 100 * np.cumsum(counts) / counts.sum(), color=self.color(spec), label=label(spec, True))
                    plotted += 1
                ax.set(title=f"{stage}: saved {field}", xlabel="Proxies ordered by use (descending)", ylabel="Cumulative saved selection count (%)", ylim=(0, 102))
                ax.grid(alpha=.16)
                if plotted: ax.legend()
                else: self.no_data(ax, "N/A: saved training usage unavailable")
        self.finish(fig, "09_saved_training_proxy_usage", "Actual saved training usage concentration",
                    "Only saved training counts are plotted. Missing usage is N/A; static nearest-neighbor binding is not substituted for actual training use.")
        fig, axes = plt.subplots(2, len(self.stages), figsize=(6.5 * len(self.stages), 7.2), squeeze=False)
        for i, domain in enumerate(("angular", "distance")):
            for j, stage in enumerate(self.stages):
                ax = axes[i, j]; specs = self.specs_for(stage, proxies=True)
                for bar, (metric, text) in enumerate((("coverage_percent", "objects covered"), ("top4_object_slot_share", "top4 object slot share"), ("maximum_repeated_neighbor_set", "largest repeated NN4 set"))):
                    values = []
                    for spec in specs:
                        summary = spec["proxies"][domain + "_nn4_hotspots"]
                        value = summary[metric]
                        values.append(value if metric == "coverage_percent" else 100 * value if metric == "top4_object_slot_share" else 100 * value / spec["proxies"]["proxy_count"])
                    ax.bar(np.arange(len(specs)) + (bar - 1) * .24, values, .23, color=COLORS[bar], label=text)
                ax.set(title=f"{stage}: {domain} NN4", ylabel="Percent of declared population / slots", ylim=(0, 103), xticks=np.arange(len(specs)))
                ax.set_xticklabels([label(s) for s in specs], rotation=12, ha="right")
                if specs: ax.legend(fontsize=7)
                else: self.no_data(ax)
                ax.grid(axis="y", alpha=.15)
        self.finish(fig, "09b_static_nn4_hotspots", "Static proxy-to-whole NN4 coverage and hotspots",
                    "Coverage /9840 objects; top4 share /all proxy NN4 slots; largest repeated set /proxy count. Separate denominators are explicit, not interchangeable usage measures.")

    def run(self):
        self.depth_figures(); self.class_figures(); self.representative_figures(); self.proxy_figures()
        sources = [{"file": path.name, "sha256": digest(path)} for path in [self.analysis / "summary.json", *sorted(self.analysis.glob("*.csv"))]]
        manifest = {"schema": "audit85-public-scientific-figures-1", "scope": self.scope,
                    "observations": [s["observation_id"] for s in self.specs],
                    "sources": sources, "files": self.files,
                    "limits": ["Only within-model geometry is compared", "Native curvatures remain separate", "Frozen minimax is not training Gumbel replay", "Class binding is not independent morphology ground truth"]}
        (self.out / "figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"scope": self.scope, "figures": len(self.files) // 2, "png_svg_files": len(self.files)}), flush=True)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=("primary", "auxiliary", "all"), default="primary")
    args = parser.parse_args(argv)
    return Figures(args.analysis, args.out, args.scope).run()


if __name__ == "__main__":
    main()
