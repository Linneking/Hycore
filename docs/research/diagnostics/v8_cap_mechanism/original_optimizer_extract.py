"""Extract original HIER hierarchy-proxy AdamW state into private small caches.

Uses PyTorch's weights-only loader with explicit known metadata classes. No
unrestricted pickle loading, model forward, optimizer update, or GPU use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def safe_checkpoint(path):
    allowed = [argparse.Namespace, np.core.multiarray.scalar, np.dtype,
               type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def extract(source, destination):
    before = digest(source)
    checkpoint = safe_checkpoint(source)
    tangent = checkpoint["cluster_loss"]["lcas"].detach().cpu()
    optimizer = checkpoint["optimizer"]
    groups = [group for group in optimizer["param_groups"]
              if group.get("name") == "lcas"]
    if len(groups) != 1 or len(groups[0]["params"]) != 1:
        raise ValueError("Expected one explicitly named lcas parameter group")
    group = groups[0]
    state = optimizer["state"][group["params"][0]]
    if tangent.ndim != 2 or tangent.shape != state["exp_avg"].shape:
        raise ValueError("lcas and Adam first moment dimensions disagree")
    if tangent.shape != state["exp_avg_sq"].shape:
        raise ValueError("lcas and Adam second moment dimensions disagree")
    arrays = {"proxy_tangent": tangent.numpy(),
              "exp_avg": state["exp_avg"].detach().cpu().numpy(),
              "exp_avg_sq": state["exp_avg_sq"].detach().cpu().numpy(),
              "step": np.asarray(state["step"].item() if torch.is_tensor(state["step"])
                                 else state["step"])}
    for key in ("proxy_tangent", "exp_avg", "exp_avg_sq"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Nonfinite extracted state: " + key)
    metadata = {"format": "original-hier-proxy-adam-state-1",
                "source_checkpoint_sha256": before,
                "source_checkpoint_bytes": source.stat().st_size,
                "epoch": int(checkpoint["epoch"]),
                "args": vars(checkpoint["args"]),
                "proxy_optimizer_group": {key: value for key, value in group.items()
                                           if key != "params"},
                "checkpoint_fields": list(checkpoint),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "loader": "weights_only=True with explicit metadata allowlist",
                "optimizer_updates": 0, "model_forwards": 0}
    destination.mkdir(parents=True, exist_ok=False)
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    array_path = destination / "proxy_adam.npz"
    np.savez_compressed(array_path, **arrays)
    with np.load(array_path, allow_pickle=False) as restored:
        for key, value in arrays.items():
            if not np.array_equal(value, restored[key]):
                raise RuntimeError("Extracted cache failed exact roundtrip")
    if digest(source) != before:
        raise RuntimeError("Input checkpoint changed during extraction")
    norms = np.linalg.norm(arrays["proxy_tangent"].astype(np.float64), axis=1)
    result = {"source_sha256": before, "epoch": metadata["epoch"],
              "step": int(arrays["step"]), "shape": list(tangent.shape),
              "raw_norm_quantiles": np.quantile(norms, [0, .05, .5, .95, 1]).tolist(),
              "over_cap_2_3": int(np.sum(norms > 2.3)),
              "optimizer_group": metadata["proxy_optimizer_group"],
              "npz_bytes": array_path.stat().st_size,
              "exact_roundtrip": True, "source_unchanged": True}
    (destination / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh private extraction directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    for dataset in ("cub", "cars", "sop"):
        source = args.handoff / ("logs/run_" + dataset + "_r50/checkpoint_best.pth")
        result = extract(source, args.output / dataset)
        print(json.dumps({"dataset": dataset, **result}), flush=True)


if __name__ == "__main__":
    main()
