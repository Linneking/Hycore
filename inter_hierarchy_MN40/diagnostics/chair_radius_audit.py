"""Deterministically audit one ModelNet40 class in a HyCoRe checkpoint.

The historical filename is retained for compatibility; ``--class-label`` and
``--class-name`` make the diagnostic reusable for any ModelNet40 category.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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

class IndexedSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        sample_id = int(self.indices[position])
        # Direct array access avoids all random train transforms.
        points = self.base.data[sample_id][: self.base.num_points].copy()
        label = int(self.base.label[sample_id].reshape(-1)[0])
        return torch.from_numpy(points), label, sample_id


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", default="test", choices=("train", "test"))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--class-label", type=int, default=8)
    parser.add_argument("--class-name", default="chair")
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


def select_groups(radius, count):
    order = np.argsort(radius)
    n = len(order)
    centers = {"inner": 0.0, "q25": 0.25, "median": 0.5, "q75": 0.75, "outer": 1.0}
    groups = {}
    for name, fraction in centers.items():
        center = int(round(fraction * (n - 1)))
        if name == "inner":
            positions = np.arange(min(count, n))
        elif name == "outer":
            positions = np.arange(max(0, n - count), n)
        else:
            lo = max(0, center - count // 2)
            hi = min(n, lo + count)
            lo = max(0, hi - count)
            positions = np.arange(lo, hi)
        groups[name] = [int(order[p]) for p in positions]
    return groups


def normalize_points(points):
    points = points - points.mean(axis=0, keepdims=True)
    return points / max(float(np.linalg.norm(points, axis=1).max()), 1e-12)


def draw_cloud(ax, points, title, elev=18, azim=42):
    points = normalize_points(points)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=points[:, 1],
               s=2.2, cmap="viridis", linewidths=0, alpha=0.92)
    ax.view_init(elev=elev, azim=azim)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def save_montages(out, points, ids, radius, confidence, groups, class_name):
    prefix = class_name.lower().replace(" ", "_")
    names = list(groups)
    cols = max(map(len, groups.values()))
    fig = plt.figure(figsize=(3.1 * cols, 2.8 * len(names)))
    for row, name in enumerate(names):
        for col, idx in enumerate(groups[name]):
            ax = fig.add_subplot(len(names), cols, row * cols + col + 1, projection="3d")
            draw_cloud(ax, points[idx], f"{name} | id={ids[idx]}\nr={radius[idx]:.4f}, p={confidence[idx]:.3f}")
    fig.suptitle(f"Original HyCoRe: {class_name} samples ordered by hyperbolic radius", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / f"{prefix}_radius_montage.png", dpi=220)
    plt.close(fig)

    extremes = groups["inner"] + groups["outer"]
    views = [(18, 42), (18, 132), (75, 42)]
    fig = plt.figure(figsize=(3 * len(views), 2.65 * len(extremes)))
    for row, idx in enumerate(extremes):
        side = "inner" if row < len(groups["inner"]) else "outer"
        for col, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(len(extremes), len(views), row * len(views) + col + 1, projection="3d")
            draw_cloud(ax, points[idx], f"{side} id={ids[idx]} r={radius[idx]:.4f}", elev, azim)
    fig.suptitle(f"{class_name.title()} radius extremes: three canonical views", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fig.savefig(out / f"{prefix}_radius_extremes_multiview.png", dpi=220)
    plt.close(fig)


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = Hype_pointMLP().to(device)
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.eval()
    base = ModelNet40(num_points=args.num_points, partition=args.partition)
    target_indices = np.flatnonzero(base.label.reshape(-1) == args.class_label)
    if len(target_indices) == 0:
        raise ValueError(f"no samples found for class label {args.class_label}")
    loader = DataLoader(IndexedSubset(base, target_indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, drop_last=False)

    embeddings, tangents, logits_all, ids_all, clouds = [], [], [], [], []
    with torch.no_grad():
        for points, _, sample_ids in loader:
            points_device = points.to(device, dtype=torch.float32)
            mu, logits = model(points_device.permute(0, 2, 1))
            embeddings.append(mu.cpu())
            tangents.append(model.manifold.logmap0(mu).cpu())
            logits_all.append(logits.cpu())
            ids_all.append(sample_ids)
            clouds.append(points)

    mu = torch.cat(embeddings)
    tangent = torch.cat(tangents)
    logits = torch.cat(logits_all)
    sample_ids = torch.cat(ids_all).numpy()
    point_clouds = torch.cat(clouds).numpy()
    with torch.no_grad():
        mu_device = mu.to(device)
        prototype = model.manifold.expmap0(tangent.mean(0, keepdim=True).to(device))
        radius = model.manifold.dist0(mu_device).cpu().numpy()
        prototype_distance = model.manifold.dist(mu_device, prototype.expand_as(mu_device)).cpu().numpy()

    direction = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype_direction = tangent.mean(0)
    prototype_direction /= prototype_direction.norm().clamp_min(1e-12)
    cosine = (direction * prototype_direction).sum(1).numpy()
    angle_deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities[:, args.class_label].numpy()
    predicted = logits.argmax(1).numpy()

    order = np.argsort(radius)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    percentile = 100 * rank / max(len(order) - 1, 1)
    groups = select_groups(radius, args.per_group)
    selected = {idx: name for name, indices in groups.items() for idx in indices}

    prefix = args.class_name.lower().replace(" ", "_")
    with (output / f"{prefix}_radius_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_index", "sample_id", "partition", "radius", "radius_percentile",
                         "prototype_hyperbolic_distance", "angle_to_prototype_deg",
                         "target_class_score", "predicted_label", "selected_group"])
        for idx in order:
            writer.writerow([int(idx), int(sample_ids[idx]), args.partition, float(radius[idx]),
                             float(percentile[idx]), float(prototype_distance[idx]), float(angle_deg[idx]),
                             float(confidence[idx]), int(predicted[idx]), selected.get(int(idx), "")])

    torch.save({"sample_ids": torch.from_numpy(sample_ids), "embedding": mu, "tangent": tangent,
                "radius": torch.from_numpy(radius), "prototype": prototype.cpu()},
               output / f"{prefix}_embeddings.pt")
    chosen = sorted(selected)
    np.savez_compressed(output / f"selected_{prefix}_pointclouds.npz", points=point_clouds[chosen],
                        sample_ids=sample_ids[chosen], radius=radius[chosen])

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_test_acc": checkpoint.get("best_test_acc"),
        "checkpoint_best_test_acc_avg": checkpoint.get("best_test_acc_avg"),
        "partition": args.partition, "class_label": args.class_label, "class_name": args.class_name,
        "num_samples": int(len(radius)), "num_points": args.num_points,
        "radius": {"min": float(radius.min()), "q25": float(np.quantile(radius, .25)),
                   "median": float(np.median(radius)), "q75": float(np.quantile(radius, .75)),
                   "max": float(radius.max()), "mean": float(radius.mean()), "std": float(radius.std())},
        "prototype_distance": {"mean": float(prototype_distance.mean()), "std": float(prototype_distance.std())},
        "angle_to_prototype_deg": {"mean": float(angle_deg.mean()), "std": float(angle_deg.std())},
        "class_accuracy": float((predicted == args.class_label).mean()),
        "selected": {name: [{"sample_id": int(sample_ids[i]), "radius": float(radius[i])}
                            for i in indices] for name, indices in groups.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(radius, bins=20, color="#4472C4", alpha=.85)
    axes[0].set(xlabel="Hyperbolic radius d(o,z)", ylabel=f"{args.class_name.title()} count", title="Radius distribution")
    axes[1].scatter(radius, prototype_distance, c=angle_deg, s=18, cmap="viridis", alpha=.8)
    axes[1].set(xlabel="Hyperbolic radius", ylabel=f"Distance to {args.class_name} prototype", title="Radius vs prototype distance")
    axes[2].scatter(radius, confidence, c=prototype_distance, s=18, cmap="plasma", alpha=.8)
    axes[2].set(xlabel="Hyperbolic radius", ylabel="Target-class score", title="Radius vs class score")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_radius_statistics.png", dpi=220)
    plt.close(fig)
    save_montages(output, point_clouds, sample_ids, radius, confidence, groups, args.class_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
