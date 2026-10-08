"""Read-only CPU verification of original HIER maps and nearest-four numerics.

The official distance is evaluated directly in modest row blocks. Tensor caches
already contain ball coordinates. Only the saved ``cluster_loss.lcas`` is mapped.
Private row identities stay in the output directory, never in this source file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_git_head(reference: Path) -> str:
    """Read the ref without Git filesystem access outside the allowed sandbox."""
    git_dir = reference / ".git"
    if git_dir.is_file():
        content = git_dir.read_text(encoding="utf-8").strip()
        git_dir = (reference / content.removeprefix("gitdir:").strip()).resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    ref = head.removeprefix("ref:").strip()
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise ValueError("HEAD ref is not present")


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
            "median": float(np.median(a)), "std": float(a.std()),
            "max": float(a.max()), "quantiles_0_1_5_25_50_75_95_99_100":
            np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()}


def nearest_exact(mu, proxies, c, block):
    """FP64 acosh identity, mathematically exact ball distance."""
    p = proxies.double()
    p2 = p.square().sum(1)[None, :]
    distances, indices = [], []
    for start in range(0, len(mu), block):
        x = mu[start:start + block].double()
        x2 = x.square().sum(1)[:, None]
        delta2 = (x2 + p2 - 2 * (x @ p.T)).clamp_min(0)
        acosh_argument = 1 + 2 * c * delta2 / ((1 - c*x2) * (1 - c*p2))
        distances.append(torch.acosh(acosh_argument.clamp_min(1)) / math.sqrt(c))
    d = torch.cat(distances, 0).T.contiguous()
    top = torch.topk(-d, 4, dim=1)
    return top.indices.numpy(), (-top.values).numpy(), d


def nearest_official(mu, proxies, c, block, pmath):
    """Unmodified released FP32 dist_matrix; no expanded-norm approximation."""
    matrix = torch.empty((len(mu), len(proxies)), dtype=torch.float32)
    t0 = time.perf_counter()
    for start in range(0, len(mu), block):
        matrix[start:start + block] = pmath.dist_matrix(mu[start:start+block], proxies, c)
        if start % (block * 64) == 0:
            print(json.dumps({"official_rows_done": start, "rows": len(mu),
                              "seconds": round(time.perf_counter()-t0, 1)}), flush=True)
    d = matrix.T.contiguous()
    top = torch.topk(-d, 4, dim=1)
    return top.indices.numpy(), (-top.values).numpy(), d


def compare_ids(a, b):
    ordered = np.all(a == b, axis=1)
    same_set = np.all(np.sort(a, 1) == np.sort(b, 1), axis=1)
    return {"proxies": len(a), "ordered_equal": int(ordered.sum()),
            "set_equal": int(same_set.sum()),
            "set_changed_proxy_ids": np.flatnonzero(~same_set).tolist(),
            "order_changed_proxy_ids": np.flatnonzero(~ordered).tolist()}


def coverage(ids):
    unique, count = np.unique(ids, return_counts=True)
    quartets, quartet_count = np.unique(np.sort(ids, axis=1), axis=0, return_counts=True)
    return {"unique_objects": len(unique), "unique_quartets": len(quartets),
            "maximum_repeated_quartet": int(quartet_count.max()),
            "maximum_object_slots": int(count.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.reference / "hier"))
    from hyptorch import pmath
    from hyptorch.nn import ToPoincare
    source_commit = read_git_head(args.reference)
    report = {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
              "source_commit": source_commit,
              "method": "unmodified official FP32 dist_matrix versus FP64 acosh identity",
              "optimizer_updates": 0, "gpu_used": False, "models": {},
              "export_source_note": "Supplied export_cache.py references out before its definition; existing caches are separately checked against weights, not claimed reproduced by that supplied script."}
    allow = [argparse.Namespace, np.core.multiarray.scalar, np.dtype,
             type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.inference_mode():
        for dataset in ["cub", "cars", "sop"]:
            cache_path = args.handoff / "proxy_eval" / "cache" / f"{dataset}_train.pt"
            ck_path = args.handoff / "logs" / f"run_{dataset}_r50" / "checkpoint_best.pth"
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            with torch.serialization.safe_globals(allow):
                ck = torch.load(ck_path, map_location="cpu", weights_only=True)
            mu = cache["embeddings"].float().contiguous()
            proxies = cache["hierarchy_proxies"].float().contiguous()
            raw = ck["cluster_loss"]["lcas"].float()
            c = float(cache["c"])
            mapped = ToPoincare(c=c, clip_r=ck["args"].clip_r,
                                 ball_dim=raw.shape[1], riemannian=True)(raw)
            map_error = (mapped-proxies).abs()
            old_path = args.handoff / "proxy_eval" / "output" / f"{dataset}_train" / "proxy_stats.csv"
            with old_path.open(newline="", encoding="utf-8") as stream:
                old_rows = sorted(csv.DictReader(stream), key=lambda row:int(row["proxy_id"]))
            old_ids = np.asarray([[int(row[f"top{i}_id"]) for i in range(4)] for row in old_rows])
            old_dist = np.asarray([[float(row[f"top{i}_dist"]) for i in range(4)] for row in old_rows])
            sample_ids = cache["sample_id"].numpy()
            if len(np.unique(sample_ids)) != len(sample_ids):
                raise ValueError("This verification requires a single view with unique sample IDs")
            by_id = {int(sample_id): position for position,sample_id in enumerate(sample_ids)}
            old_indices = torch.tensor([[by_id[int(sample_id)] for sample_id in row] for row in old_ids])
            t0 = time.perf_counter()
            exact_indices, exact_dist, exact_matrix = nearest_exact(mu, proxies, c, max(1024, args.block))
            exact_ids = sample_ids[exact_indices]
            official_indices, official_dist, official_matrix = nearest_official(mu, proxies, c, args.block, pmath)
            official_ids = sample_ids[official_indices]
            matrix_error = official_matrix.double() - exact_matrix
            old_exact_distance = exact_matrix[torch.arange(len(proxies))[:,None], old_indices].numpy()
            old_official_distance = official_matrix[torch.arange(len(proxies))[:,None], old_indices].numpy()
            tangent_norm = raw.norm(dim=1).numpy()
            proxy_depth = (2/math.sqrt(c)*np.arctanh(math.sqrt(c)*proxies.double().norm(dim=1).numpy()))
            whole_depth = 2/math.sqrt(c)*np.arctanh(math.sqrt(c)*mu.double().norm(dim=1).numpy())
            item = {"N": len(mu), "P": len(proxies), "D": mu.shape[1], "c": c,
                    "epoch": int(ck["epoch"]), "bn_freeze_effective": bool(ck["args"].bn_freeze),
                    "checkpoint_sha256": sha256(ck_path), "cache_sha256": sha256(cache_path),
                    "checkpoint_hash_matches_cache": sha256(ck_path) == cache["checkpoint_hash"],
                    "official_map_max_abs_error": float(map_error.max()),
                    "official_map_exact_equal": bool(torch.equal(mapped, proxies)),
                    "official_map_within_1e6": bool(float(map_error.max()) < 1e-6),
                    "cache_code_commit": cache["code_commit"],
                    "raw_proxy_tangent_norm": describe(tangent_norm),
                    "raw_proxy_norm_above_clip_count": int((tangent_norm > ck["args"].clip_r).sum()),
                    "proxy_depth": describe(proxy_depth), "whole_depth": describe(whole_depth),
                    "whole_depth_within_1e4_of_4p6": int((np.abs(whole_depth-4.6)<1e-4).sum()),
                    "official_vs_old": compare_ids(official_ids, old_ids),
                    "exact_vs_old": compare_ids(exact_ids, old_ids),
                    "official_vs_exact": compare_ids(official_ids, exact_ids),
                    "official_coverage": coverage(official_ids), "exact_coverage": coverage(exact_ids),
                    "all_distance_official_minus_exact": describe(matrix_error.numpy()),
                    "official_minus_old_nn4_abs_max": float(np.abs(official_dist-old_dist).max()),
                    "old_recorded_same_pairs_minus_exact": describe(old_dist-old_exact_distance),
                    "old_recorded_same_pairs_minus_official": describe(old_dist-old_official_distance),
                    "old_proxy_radius_abs_max": float(np.max(np.abs(
                        proxies.norm(dim=1).numpy()-np.asarray([float(row["radius"]) for row in old_rows])))),
                    "seconds": time.perf_counter()-t0}
            np.savez_compressed(args.output / f"{dataset}_nn4_private.npz", official_ids=official_ids,
                                exact_ids=exact_ids, old_ids=old_ids, official_distance=official_dist,
                                exact_distance=exact_dist)
            report["models"][dataset] = item
            (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding="utf-8")
            print(json.dumps({"completed": dataset, **item},ensure_ascii=False),flush=True)
    print(str(args.output / "verification.json"), flush=True)


if __name__ == "__main__":
    main()
