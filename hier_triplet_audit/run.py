"""Nine-setting, no-training HIER triplet audit on original HyCoRe features.

Usage (from the repository root)::

    python hier_triplet_audit/run.py --checkpoint /path/to/best_checkpoint.pth \
        --data-dir /path/to/modelnet40_ply_hdf5_2048 --output /new/run/directory

The output directory must not exist.  Only ModelNet40's training split is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core import (DEFAULT_GRID, GridConfig, balanced_batches, draw_triplets,
                  mine_pools, mining_counts, poincare_dist_matrix,
                  proxy_probe, random_proxy_bank)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_ROOT = REPO_ROOT / "classification_ModelNet40"


class FixedClouds(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray):
        self.points = points
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # The original training dataset translates/shuffles point clouds.  This
        # independent reader intentionally performs neither operation.
        return torch.from_numpy(self.points[index]), int(self.labels[index]), index


def read_training_h5(data_dir: Path, num_points: int):
    paths = sorted(data_dir.glob("ply_data_train*.h5"))
    if not paths:
        raise FileNotFoundError(f"no training HDF5 files in {data_dir}")
    data, labels, files = [], [], []
    offset = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            points = np.asarray(handle["data"][:, :num_points, :], dtype=np.float32)
            classes = np.asarray(handle["label"][:], dtype=np.int64).reshape(-1)
        if len(points) != len(classes):
            raise ValueError(f"data/label length mismatch in {path}")
        data.append(points)
        labels.append(classes)
        files.append({"name": path.name, "bytes": path.stat().st_size,
                      "first_sample_id": offset, "count": len(classes)})
        offset += len(classes)
    return np.concatenate(data), np.concatenate(labels), files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def extract_features(args, output: Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray, dict]:
    sys.path.insert(0, str(CLASSIFICATION_ROOT))
    from models.pointmlp import Hype_pointMLP  # noqa: E402

    points, labels, files = read_training_h5(Path(args.data_dir), args.num_points)
    model = Hype_pointMLP().to(device)
    # This is the user's own checkpoint, not an arbitrary downloaded pickle.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("net", checkpoint.get("model", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = DataLoader(FixedClouds(points, labels), batch_size=args.inference_batch_size,
                        shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
    chunks, ids_seen, labels_seen, predictions = [], [], [], []
    with torch.inference_mode():
        for step, (cloud, target, sample_ids) in enumerate(loader, 1):
            cloud = cloud.to(device, dtype=torch.float32, non_blocking=True).permute(0, 2, 1)
            embedding, logits = model(cloud)
            chunks.append(embedding.detach().cpu())
            ids_seen.append(sample_ids)
            labels_seen.append(target)
            predictions.append(logits.argmax(dim=1).cpu())
            if step % 50 == 0:
                print(f"feature extraction: {step}/{len(loader)} batches", flush=True)
    embedding = torch.cat(chunks).contiguous()
    sample_ids = torch.cat(ids_seen)
    cached_labels = torch.cat(labels_seen)
    predicted = torch.cat(predictions)
    if not torch.equal(sample_ids, torch.arange(len(labels))):
        raise RuntimeError("sample ID order changed during extraction")
    if not np.array_equal(cached_labels.numpy(), labels):
        raise RuntimeError("label order changed during extraction")
    if embedding.shape != (len(labels), 256) or not torch.isfinite(embedding).all():
        raise RuntimeError(f"invalid embedding tensor {tuple(embedding.shape)}")
    norms = embedding.norm(dim=1)
    if not torch.all(norms < 1):
        raise RuntimeError("checkpoint produced points outside the c=1 Poincare ball")
    cache = {"embedding": embedding, "labels": cached_labels, "sample_ids": sample_ids,
             "predicted": predicted, "split": "train_unaugmented",
             "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
             "source_files": files, "num_points": args.num_points}
    torch.save(cache, output / "feature_cache.pt")
    metadata = {
        "sample_count": len(labels), "embedding_dim": embedding.shape[1],
        "class_count": int(len(np.unique(labels))), "source_files": files,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
        "train_accuracy_diagnostic_only": float((predicted == cached_labels).float().mean()),
        "embedding_norm_min": float(norms.min()), "embedding_norm_max": float(norms.max()),
        "checkpoint_sha256": cache["checkpoint_sha256"],
    }
    print(f"cached {len(labels)} train embeddings; checkpoint epoch={metadata['checkpoint_epoch']}", flush=True)
    return embedding, labels, metadata


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(rows: list[dict], numerator: str, denominator: str) -> float | None:
    bottom = sum(float(row[denominator]) for row in rows)
    return sum(float(row[numerator]) for row in rows) / bottom if bottom else None


def bootstrap_interval(rows: list[dict], numerator: str, denominator: str,
                       seed: int = 227, repeats: int = 500):
    if not rows:
        return None
    top = np.asarray([row[numerator] for row in rows], dtype=np.float64)
    bottom = np.asarray([row[denominator] for row in rows], dtype=np.float64)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = generator.integers(0, len(rows), len(rows))
        denominator_sum = bottom[sample].sum()
        if denominator_sum:
            values.append(top[sample].sum() / denominator_sum)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None


def summarize(config: GridConfig, mining_rows: list[dict], probe_rows: list[dict]) -> dict:
    result = asdict(config)
    result.update({
        "batches": len(mining_rows),
        "eligible_anchor_fraction": ratio(mining_rows, "eligible_anchors", "anchors"),
        "mean_positive_pool": ratio(mining_rows, "positive_pool_sum", "eligible_anchors"),
        "mean_negative_pool": ratio(mining_rows, "negative_pool_sum", "eligible_anchors"),
        "expected_j_same_fraction": ratio(mining_rows, "expected_j_same_sum", "eligible_anchors"),
        "expected_k_same_other_fraction": ratio(mining_rows, "expected_k_same_other_sum", "eligible_anchors"),
        "expected_k_cross_fraction": ratio(mining_rows, "expected_k_cross_sum", "eligible_anchors"),
        "expected_k_self_fraction": ratio(mining_rows, "expected_k_self_sum", "eligible_anchors"),
        "sampled_j_same_fraction": ratio(mining_rows, "j_same", "triplets"),
        "sampled_j_cross_fraction": ratio(mining_rows, "j_cross", "triplets"),
        "sampled_k_same_other_fraction": ratio(mining_rows, "k_same_other", "triplets"),
        "sampled_k_cross_fraction": ratio(mining_rows, "k_cross", "triplets"),
        "sampled_k_self_fraction": ratio(mining_rows, "k_self", "triplets"),
        "distance_order_fraction": ratio(mining_rows, "ordered", "triplets"),
        "unique_triplet_fraction": ratio(mining_rows, "unique_triplets", "triplets"),
        "total_triplets": int(sum(row["triplets"] for row in mining_rows)),
        "random_proxy_collision_fraction": ratio(probe_rows, "collisions", "triplets"),
        "random_proxy_pre_active_fraction": ratio(probe_rows, "pre_active", "triplets"),
        "random_proxy_effective_fraction": ratio(probe_rows, "effective", "triplets"),
        "random_proxy_masked_loss_mean": ratio(probe_rows, "masked_loss_sum", "triplets"),
        "j_same_bootstrap95": bootstrap_interval(mining_rows, "j_same", "triplets"),
        "k_same_other_bootstrap95": bootstrap_interval(mining_rows, "k_same_other", "triplets"),
        "distance_order_bootstrap95": bootstrap_interval(mining_rows, "ordered", "triplets"),
    })
    return result


def audit_grid(args, output: Path, embedding: torch.Tensor, labels: np.ndarray,
               device: torch.device) -> list[dict]:
    if args.batch_seeds is None:
        batch_seeds = [22, 42, 2026]
    else:
        batch_seeds = [int(value) for value in args.batch_seeds.split(",")]
    proxy_seeds = [int(value) for value in args.proxy_seeds.split(",")]
    if len(set(batch_seeds)) != len(batch_seeds) or len(set(proxy_seeds)) != len(proxy_seeds):
        raise ValueError("seeds must be distinct")
    banks = {seed: random_proxy_bank(args.proxy_count, embedding.shape[1], seed, device,
                                     c=args.curvature) for seed in proxy_seeds}
    proxy_rows = []
    for seed, bank in banks.items():
        proxy_distance = poincare_dist_matrix(bank, c=args.curvature)
        for k in sorted({config.proxy_k for config in DEFAULT_GRID}):
            proxy_config = GridConfig(f"proxy_k{k}", "hier", k, k, 0)
            positive, negative = mine_pools(proxy_distance, np.arange(len(bank)), proxy_config)
            triples = draw_triplets(positive, negative, seed=800000 + seed * 100 + k,
                                    per_anchor=args.triples_per_anchor)
            probe = proxy_probe(proxy_distance, triples, seed=900000 + seed * 100 + k,
                                tau=args.tau, margin=args.margin, proxy_nodes=True)
            proxy_rows.append({"proxy_seed": seed, "proxy_k": k, **probe})
        print(f"random proxy bank {seed}: proxy-proxy branch complete", flush=True)
    write_csv(output / "proxy_branch.csv", proxy_rows)

    mining_rows, sample_probe_rows = [], []
    total_batches = len(batch_seeds) * args.batches_per_seed
    completed = 0
    for batch_seed in batch_seeds:
        for batch_index, ids in enumerate(balanced_batches(labels, batch_seed, args.batches_per_seed)):
            cloud_embedding = embedding[ids].to(device)
            batch_labels = labels[ids]
            distance = poincare_dist_matrix(cloud_embedding, c=args.curvature)
            cp_distances = {seed: poincare_dist_matrix(cloud_embedding, bank,
                                                       c=args.curvature) for seed, bank in banks.items()}
            for config in DEFAULT_GRID:
                positive, negative = mine_pools(distance, batch_labels, config)
                triplets = draw_triplets(positive, negative,
                                         seed=1000000 + batch_seed * 1000 + batch_index,
                                         per_anchor=args.triples_per_anchor)
                counts = mining_counts(distance, batch_labels, positive, negative, triplets)
                mining_rows.append({"config": config.name, "batch_seed": batch_seed,
                                    "batch_index": batch_index, **counts})
                for proxy_seed, cp_distance in cp_distances.items():
                    probe = proxy_probe(cp_distance, triplets,
                                        seed=3000000 + proxy_seed * 100000 +
                                             batch_seed * 1000 + batch_index,
                                        tau=args.tau, margin=args.margin)
                    sample_probe_rows.append({"config": config.name,
                                              "batch_seed": batch_seed,
                                              "batch_index": batch_index,
                                              "proxy_seed": proxy_seed, **probe})
            completed += 1
            if completed % 10 == 0 or completed == total_batches:
                print(f"audited {completed}/{total_batches} fixed 5x8 batches", flush=True)
    write_csv(output / "batch_mining.csv", mining_rows)
    write_csv(output / "sample_proxy_probe.csv", sample_probe_rows)
    summaries = []
    for config in DEFAULT_GRID:
        mine = [row for row in mining_rows if row["config"] == config.name]
        probe = [row for row in sample_probe_rows if row["config"] == config.name]
        summaries.append(summarize(config, mine, probe))
    write_csv(output / "summary.csv", [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in summaries
    ])
    (output / "summary.json").write_text(json.dumps({
        "note": "No weights updated. Proxy metrics use random initialization, not a trained hierarchy.",
        "grid": summaries,
        "proxy_branch": proxy_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-seeds", default=None,
                        help="comma-separated, default: 22,42,2026")
    parser.add_argument("--proxy-seeds", default="11,29")
    parser.add_argument("--batches-per-seed", type=int, default=20)
    parser.add_argument("--triples-per-anchor", type=int, default=50)
    parser.add_argument("--proxy-count", type=int, default=512)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = arguments()
    if args.batches_per_seed <= 0 or args.triples_per_anchor <= 0 or args.proxy_count <= 30:
        raise ValueError("invalid audit size")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "start_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "geoopt": package_version("geoopt"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "grid": [asdict(config) for config in DEFAULT_GRID],
        "split": "ModelNet40 train, no augmentation", "weights_updated": False,
        "proxy_interpretation": "randomly initialized HIER-style proxies only",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.monotonic()
    try:
        torch.manual_seed(22)
        np.random.seed(22)
        torch.backends.cudnn.benchmark = False
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        embeddings, labels, extraction = extract_features(args, output, device)
        summaries = audit_grid(args, output, embeddings, labels, device)
        manifest.update({"status": "complete", "extraction": extraction,
                         "elapsed_seconds": time.monotonic() - start,
                         "end_utc": datetime.now(timezone.utc).isoformat()})
        if device.type == "cuda":
            manifest["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        print(json.dumps({"status": "complete", "output": str(output),
                          "configs": len(summaries),
                          "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc),
                         "traceback": traceback.format_exc(),
                         "end_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
