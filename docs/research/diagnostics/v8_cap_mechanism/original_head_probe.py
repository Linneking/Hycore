"""Read-only original HIER head probe on at most 12 known training images.

Reuses the already audited source-compatible R50/official ToPoincare adapter.
This measures pre-cap values that cannot be recovered from capped caches.
No metric-learning classifier, optimizer, backward, or test split is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_original_image_epoch_v1 import (  # noqa: E402
    EXPECTED, ImagePool, OriginalWhole, build_transforms, read_identity_map,
    read_sample_map, resolve_paths, state_fingerprint, unwrapped_state,
)
from hier_original_activation_probe_v1 import official_runtime, sha256  # noqa: E402
from hier_original_handoff_loader_v1 import safe_checkpoint  # noqa: E402


def stats(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty or nonfinite head measurement")
    quantiles = np.quantile(a, [0, .05, .25, .5, .75, .95, 1])
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                    map(float, quantiles)), mean=float(a.mean()), std=float(a.std()),
                count=len(a))


def probe(args):
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.gpu_identity:
        raise ValueError("GPU use requires an externally checked physical GPU identity")
    if not 1 <= args.anchors <= 12 or args.microbatch < 1:
        raise ValueError("Use 1--12 training anchors and a positive microbatch")
    hashes = {key: sha256(getattr(args, key))
              for key in ("checkpoint", "clean_cache", "sample_map")}
    if args.identity_map:
        hashes["identity_map"] = sha256(args.identity_map)
    checkpoint = safe_checkpoint(args.checkpoint)
    margs = checkpoint["args"]
    if (margs.dataset.lower() != args.dataset or margs.emb != 512 or
            not math.isclose(margs.hyp_c, .1) or not math.isclose(margs.clip_r, 2.3)):
        raise ValueError("Not the supplied original HIER training geometry")
    with np.load(args.clean_cache, allow_pickle=False) as cache:
        labels = cache["labels"].copy()
        ids = cache["sample_ids"].copy()
        metadata = json.loads(str(cache["metadata_json"].item()))
        indices = np.unique(np.linspace(0, len(ids) - 1, min(args.anchors, len(ids)), dtype=int))
        target = torch.from_numpy(cache["mu"][indices].copy()).float()
    expected_n, expected_classes, _ = EXPECTED[args.dataset]
    if len(ids) != expected_n or len(np.unique(labels)) != expected_classes:
        raise ValueError("Expected the audited complete training cache, not eval/test")
    source_hash = checkpoint.get("source_checkpoint_sha256", hashes["checkpoint"])
    if metadata["checkpoint_hash"] != source_hash or metadata["checkpoint_epoch"] != int(checkpoint["epoch"]):
        raise ValueError("Checkpoint/cache identity mismatch")
    source_paths = read_sample_map(args.sample_map, ids, labels)
    paths = resolve_paths(source_paths, args.dataset, args.images_root,
                          read_identity_map(args.identity_map))
    _, hypnn, _, _ = official_runtime(args.reference)
    model = OriginalWhole(hypnn, margs)
    state = unwrapped_state(checkpoint["stduent"])
    model.load_state_dict(state, strict=True)
    if len(model.state_dict()) != 632 or not all(torch.equal(value, state[key])
                                                for key, value in model.state_dict().items()):
        raise AssertionError("Strict source state verification failed")
    model.requires_grad_(False)
    del checkpoint, state
    model.to(device).eval()
    before = state_fingerprint(model)
    measurements = {"pre_linear": [], "pre_to_poincare": []}

    def hook(name):
        def record(_module, inputs):
            measurements[name].append(inputs[0].detach().float().cpu())
        return record

    handles = [model.last_layer[1].register_forward_pre_hook(hook("pre_linear")),
               model.last_layer[2][0].register_forward_pre_hook(hook("pre_to_poincare"))]
    try:
        _, clean = build_transforms()
        pool = ImagePool(paths, clean)
        actual = []
        with torch.no_grad():
            for start in range(0, len(indices), args.microbatch):
                batch_ids = indices[start:start + args.microbatch]
                images = torch.stack([pool[int(i)][0] for i in batch_ids]).to(device)
                actual.append(model(images).float().cpu())
        actual = torch.cat(actual)
    finally:
        for handle in handles:
            handle.remove()
    if state_fingerprint(model) != before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Network/BN/gradient state changed")
    cosine = F.cosine_similarity(actual, target, dim=1)
    max_delta = float((actual - target).abs().max())
    if float(cosine.min()) < .9999 or max_delta > .01:
        raise AssertionError("Clean anchor replay does not match audited source cache")
    pre_linear = torch.cat(measurements["pre_linear"])
    pre_cap = torch.cat(measurements["pre_to_poincare"])
    linear_norm = pre_linear.norm(dim=1).numpy()
    cap_norm = pre_cap.norm(dim=1).numpy()
    w = model.last_layer[1].weight.detach().float().cpu()
    singular = torch.linalg.svdvals(w)
    bias = model.last_layer[1].bias.detach().float().cpu()
    radius = actual.norm(dim=1)
    depths = 2 / math.sqrt(.1) * torch.atanh((math.sqrt(.1) * radius).clamp(max=1 - 1e-7))
    for key, value in hashes.items():
        if sha256(getattr(args, key)) != value:
            raise AssertionError("An input file changed during the probe")
    result = {
        "dataset": args.dataset, "source_epoch": int(metadata["checkpoint_epoch"]),
        "training_objects": len(ids), "anchors": len(indices),
        "sample_positions": indices.tolist(), "seed": args.seed,
        "pre_linear_norm": stats(linear_norm),
        "pre_to_poincare_norm": stats(cap_norm),
        "precap_over_2_3_count": int((cap_norm > 2.3).sum()),
        "precap_over_2_3_fraction": float((cap_norm > 2.3).mean()),
        "cap_norm_divided_by_2_3": stats(cap_norm / 2.3),
        "postcap_depth": stats(depths.numpy()),
        "linear_weight_row_norm": stats(w.norm(dim=1).numpy()),
        "linear_weight_singular_values": stats(singular.numpy()),
        "linear_weight_frobenius_norm": float(w.norm()),
        "linear_bias_norm": float(bias.norm()),
        "use_lastnorm": bool(margs.use_lastnorm),
        "architecture": "Resnet50 gap+gmp -> nonaffine LayerNorm -> Linear2048to512 -> official ToPoincare cap2.3",
        "explicit_final_l2_normalization": False,
        "clean_anchor_min_cosine": float(cosine.min()),
        "clean_anchor_max_abs_coordinate_delta": max_delta,
        "model_BN_fingerprint_unchanged": True, "strict_state_load": True,
        "optimizer_updates": 0, "backward_calls": 0, "test_forward_calls": 0,
        "device": str(device), "gpu_identity": args.gpu_identity,
        "input_hashes": hashes, "source_checkpoint_sha256": source_hash,
        "code_commit": args.code_commit, "runner_sha256": sha256(__file__),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.time() - started,
        "limits": "At most12 evenly spaced training anchors; descriptive best-checkpoint head probe, not a training history or whole-pool estimate.",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "private_anchor_measurements.npz", sample_positions=indices,
                        pre_linear_norm=linear_norm, pre_to_poincare_norm=cap_norm,
                        postcap_depth=depths.numpy(), output=actual.numpy(), target=target.numpy())
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXPECTED, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--clean-cache", required=True)
    parser.add_argument("--sample-map", required=True)
    parser.add_argument("--identity-map")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-identity")
    parser.add_argument("--anchors", type=int, default=12)
    parser.add_argument("--microbatch", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--code-commit", required=True)
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
