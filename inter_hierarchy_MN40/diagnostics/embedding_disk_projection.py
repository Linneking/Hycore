"""Project original HyCoRe test embeddings to comparable 2-D disk views.

Two qualitative projections are exported:
1. direction PCA with each sample's original Poincare radius restored;
2. uncentered tangent-space PCA followed by a 2-D expmap at the origin.

Neither projection is an isometry. The first is intended to inspect radial
placement and dominant angular organization; the second shows variance carried
by the leading two tangent components.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = HERE.parents[1] / "classification_ModelNet40"
sys.path.insert(0, str(CLASSIFICATION_ROOT))
from data import ModelNet40  # noqa: E402
from models.pointmlp import Hype_pointMLP  # noqa: E402


DEFAULT_CLASSES = "chair,lamp,table,sofa,stool,desk,bed,bookshelf"


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base.data)

    def __getitem__(self, index):
        points = self.base.data[index][: self.base.num_points].copy()
        label = int(self.base.label[index].reshape(-1)[0])
        return torch.from_numpy(points), label, index


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", default=DEFAULT_CLASSES)
    parser.add_argument("--partition", default="test", choices=("train", "test"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("model", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def read_class_names(base):
    path = CLASSIFICATION_ROOT / "data" / "modelnet40_ply_hdf5_2048" / "shape_names.txt"
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Standard ModelNet40 order, used only if shape_names.txt is unavailable.
    return """airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox""".split()


def pca_basis(x, dimensions=2):
    """Uncentered PCA through the origin, appropriate for tangent directions."""
    _, singular, vectors = torch.pca_lowrank(x.float(), q=dimensions, center=False)
    vectors = vectors[:, :dimensions]
    # Resolve sign ambiguity deterministically using each component's largest loading.
    for column in range(dimensions):
        pivot = torch.argmax(vectors[:, column].abs())
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    energy = singular[:dimensions].square()
    total = x.float().square().sum().clamp_min(1e-12)
    return vectors, (energy / total).cpu().numpy()


def disk_projections(mu, tangent):
    eps = 1e-12
    directions = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(eps)
    direction_basis, direction_energy = pca_basis(directions)
    direction_2d = directions @ direction_basis
    direction_2d = direction_2d / direction_2d.norm(dim=1, keepdim=True).clamp_min(eps)
    ball_radius = mu.norm(dim=1, keepdim=True)
    radial_preserving = direction_2d * ball_radius

    tangent_basis, tangent_energy = pca_basis(tangent)
    tangent_2d = tangent @ tangent_basis
    tangent_norm = tangent_2d.norm(dim=1, keepdim=True)
    tangent_disk = torch.tanh(tangent_norm) * tangent_2d / tangent_norm.clamp_min(eps)
    return radial_preserving, tangent_disk, direction_energy, tangent_energy


def setup_disk(ax, title):
    circle = plt.Circle((0, 0), 1.0, facecolor="#fafafa", edgecolor="#222222", lw=1.2, zorder=0)
    ax.add_patch(circle)
    ax.axhline(0, color="#d8d8d8", lw=.55, zorder=0)
    ax.axvline(0, color="#d8d8d8", lw=.55, zorder=0)
    ax.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect="equal", title=title)
    ax.set_xticks([])
    ax.set_yticks([])


def class_colors(count):
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(count - 1, 1)) for i in range(count)]


def plot_selected(coords_a, coords_b, labels, class_names, selected_labels, out):
    colors = class_colors(len(selected_labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, coords, title in zip(
        axes, (coords_a, coords_b),
        ("Radial-preserving direction PCA", "Tangent PCA + 2-D expmap"),
    ):
        setup_disk(ax, title)
        for color, label in zip(colors, selected_labels):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, color=color,
                       alpha=.72, linewidths=0, label=class_names[label])
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, markerscale=1.4)
    fig.suptitle("Original HyCoRe: selected ModelNet40 classes in a shared 2-D disk projection", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "selected8_disk_projection_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7.2))
    selected_mask = np.isin(labels, selected_labels)
    for ax, color, label in zip(axes.flat, colors, selected_labels):
        setup_disk(ax, class_names[label])
        ax.scatter(coords_a[selected_mask, 0], coords_a[selected_mask, 1], s=5,
                   color="#d4d4d4", alpha=.25, linewidths=0)
        mask = labels == label
        ax.scatter(coords_a[mask, 0], coords_a[mask, 1], s=17, color=color,
                   alpha=.85, linewidths=0)
        centroid = coords_a[mask].mean(axis=0)
        ax.scatter([centroid[0]], [centroid[1]], marker="x", color="black", s=38, linewidths=1.2)
    fig.suptitle("Radial-preserving projection: one highlighted class per panel", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "selected8_disk_small_multiples.png", dpi=240)
    plt.close(fig)


def plot_all_classes(coords, labels, class_names, out):
    colors = class_colors(len(class_names))
    fig, ax = plt.subplots(figsize=(12, 11))
    setup_disk(ax, "All 40 ModelNet40 classes — radial-preserving direction PCA")
    for label, color in enumerate(colors):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=7, color=color, alpha=.42, linewidths=0)
        centroid = coords[mask].mean(axis=0)
        abbreviation = class_names[label][:4]
        ax.text(centroid[0], centroid[1], abbreviation, fontsize=6.2, color="black",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none", alpha=.62))
    fig.tight_layout()
    fig.savefig(out / "all40_disk_overview.png", dpi=260)
    plt.close(fig)


def normalized_points(points):
    points = points - points.mean(axis=0, keepdims=True)
    return points / max(float(np.linalg.norm(points, axis=1).max()), 1e-12)


def draw_cloud(ax, points, title):
    points = normalized_points(points)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=points[:, 1],
               s=1.8, cmap="viridis", linewidths=0, alpha=.92)
    ax.view_init(elev=18, azim=42)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=7)


def representative_indices(coords, labels, selected_labels):
    representatives = {}
    for label in selected_labels:
        local = np.flatnonzero(labels == label)
        xy = coords[local]
        centroid = xy.mean(axis=0)
        candidates = [
            local[np.argmin(xy[:, 0])], local[np.argmax(xy[:, 0])],
            local[np.argmin(xy[:, 1])], local[np.argmax(xy[:, 1])],
            local[np.argmin(np.linalg.norm(xy - centroid, axis=1))],
        ]
        unique = []
        for index in candidates:
            if int(index) not in unique:
                unique.append(int(index))
        if len(unique) < 5:
            distance = np.linalg.norm(xy - centroid, axis=1)
            for index in local[np.argsort(distance)[::-1]]:
                if int(index) not in unique:
                    unique.append(int(index))
                if len(unique) == 5:
                    break
        representatives[label] = unique[:5]
    return representatives


def plot_representative_shapes(coords, labels, ids, radii, points, class_names, selected_labels, out):
    roles = ("left", "right", "bottom", "top", "center")
    representatives = representative_indices(coords, labels, selected_labels)
    fig = plt.figure(figsize=(14.2, 2.35 * len(selected_labels)))
    for row, label in enumerate(selected_labels):
        for col, index in enumerate(representatives[label]):
            ax = fig.add_subplot(len(selected_labels), 5, row * 5 + col + 1, projection="3d")
            title = (f"{class_names[label]} | {roles[col]} | id={ids[index]}\n"
                     f"xy=({coords[index,0]:.2f},{coords[index,1]:.2f}), r={radii[index]:.3f}")
            draw_cloud(ax, points[index], title)
    fig.suptitle("Representative shapes at directional extremes of the shared disk projection", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .988))
    fig.savefig(out / "selected8_projection_representative_shapes.png", dpi=230)
    plt.close(fig)
    return representatives


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = Hype_pointMLP().to(device)
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.eval()
    base = ModelNet40(num_points=args.num_points, partition=args.partition)
    class_names = read_class_names(base)
    name_to_label = {name: label for label, name in enumerate(class_names)}
    selected_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    missing = [name for name in selected_names if name not in name_to_label]
    if missing:
        raise ValueError(f"unknown ModelNet40 classes: {missing}")
    selected_labels = [name_to_label[name] for name in selected_names]

    loader = DataLoader(IndexedDataset(base), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=False)
    embeddings, tangents, logits_all, labels_all, ids_all, clouds = [], [], [], [], [], []
    with torch.no_grad():
        for points, labels, ids in loader:
            data = points.to(device, dtype=torch.float32).permute(0, 2, 1)
            mu, logits = model(data)
            embeddings.append(mu.cpu())
            tangents.append(model.manifold.logmap0(mu).cpu())
            logits_all.append(logits.cpu())
            labels_all.append(labels)
            ids_all.append(ids)
            clouds.append(points)

    mu = torch.cat(embeddings)
    tangent = torch.cat(tangents)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    ids = torch.cat(ids_all).numpy()
    points = torch.cat(clouds).numpy()
    radial_coords, tangent_coords, direction_energy, tangent_energy = disk_projections(mu, tangent)
    radial_coords = radial_coords.numpy()
    tangent_coords = tangent_coords.numpy()
    radii = mu.norm(dim=1).numpy()
    hyperbolic_radii = model.manifold.dist0(mu.to(device)).cpu().numpy()
    predicted = logits.argmax(dim=1).numpy()

    plot_selected(radial_coords, tangent_coords, labels, class_names, selected_labels, output)
    plot_all_classes(radial_coords, labels, class_names, output)
    representatives = plot_representative_shapes(
        radial_coords, labels, ids, hyperbolic_radii, points, class_names, selected_labels, output
    )

    with (output / "embedding_disk_coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "class_label", "class_name", "predicted_label",
                         "poincare_euclidean_radius", "hyperbolic_radius",
                         "radial_pca_x", "radial_pca_y", "tangent_pca_x", "tangent_pca_y"])
        for i in range(len(labels)):
            writer.writerow([int(ids[i]), int(labels[i]), class_names[labels[i]], int(predicted[i]),
                             float(radii[i]), float(hyperbolic_radii[i]),
                             float(radial_coords[i, 0]), float(radial_coords[i, 1]),
                             float(tangent_coords[i, 0]), float(tangent_coords[i, 1])])

    with (output / "class_centroids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_label", "class_name", "count", "centroid_x", "centroid_y",
                         "mean_hyperbolic_radius", "accuracy"])
        for label, name in enumerate(class_names):
            mask = labels == label
            centroid = radial_coords[mask].mean(axis=0)
            writer.writerow([label, name, int(mask.sum()), float(centroid[0]), float(centroid[1]),
                             float(hyperbolic_radii[mask].mean()), float((predicted[mask] == label).mean())])

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "partition": args.partition,
        "num_samples": int(len(labels)),
        "selected_classes": selected_names,
        "direction_pca_energy_fraction": direction_energy.tolist(),
        "tangent_pca_energy_fraction": tangent_energy.tolist(),
        "warning": "Both 2-D projections are qualitative and are not hyperbolic isometries.",
        "representative_sample_ids": {
            class_names[label]: [int(ids[i]) for i in representatives[label]]
            for label in selected_labels
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"sample_ids": torch.from_numpy(ids), "labels": torch.from_numpy(labels),
                "embedding": mu, "tangent": tangent,
                "radial_preserving_coords": torch.from_numpy(radial_coords),
                "tangent_pca_coords": torch.from_numpy(tangent_coords)}, output / "projection_data.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
