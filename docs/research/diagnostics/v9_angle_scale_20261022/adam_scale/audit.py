"""CPU-only fixed-state AdamW audit of saved V9 steps and original Cars state.

No encoder forwards, gradient evaluation, or source-parameter updates. All
parameter updates below operate on detached private clones. Outputs contain
only anonymous aggregate statistics, never source paths or parameter arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


CAP = 2.3
SNAPSHOTS = ((1, 1), (1, 5), (1, 20), (1, 100), (1, 200), (5, 200), (10, 200))


def stats(value):
    a = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value,
                   dtype=np.float64).reshape(-1)
    if len(a) == 0:
        return None
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite aggregate")
    q = np.quantile(a, [0, .05, .5, .95, 1])
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, q)),
                mean=float(a.mean()))


@torch.no_grad()
def clone_step(saved, *, lr_scale=1., eps=None, current_scale=1., history_scale=1.):
    p = torch.nn.Parameter(saved["parameter"].detach().cpu().clone())
    group = dict(lr=float(saved["lr"]) * lr_scale,
                 betas=tuple(map(float, saved["betas"])),
                 eps=float(saved["eps"]) if eps is None else eps,
                 weight_decay=float(saved["weight_decay"]), foreach=False)
    optimizer = torch.optim.AdamW([p], **group)
    optimizer.state[p] = {
        "step": torch.tensor(float(saved["step_before"]), dtype=torch.float32),
        "exp_avg": saved["exp_avg"].detach().cpu().clone() * history_scale,
        "exp_avg_sq": saved["exp_avg_sq"].detach().cpu().clone() * history_scale ** 2,
    }
    p.grad = saved["gradient"].detach().cpu().clone() * current_scale
    optimizer.step()
    return p.detach()


def update_summary(before, after, *, native_after=None):
    p, q = before.double(), after.double()
    radius = p.norm(dim=-1)
    delta = q - p
    change = q.norm(dim=-1) - radius
    masks = {"all": torch.ones_like(radius, dtype=torch.bool),
             "before_inside": radius <= CAP, "before_outside": radius > CAP}
    if native_after is not None:
        masks["native_first_crossers"] = (radius <= CAP) & (native_after.double().norm(dim=-1) > CAP)
    result = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": int(mask.sum()), "before_norm": stats(radius[mask]),
            "gap_to_cap": stats((CAP - radius)[mask]), "norm_change": stats(change[mask]),
            "step_norm": stats(delta.norm(dim=-1)[mask]),
            "step_over_before_norm": stats((delta.norm(dim=-1) / radius)[mask]),
            "signed_radial_step": stats(((delta * p / radius[:, None]).sum(-1))[mask]),
            "norm_outward_count": int(((change > 0) & mask).sum()),
            "after_over_cap": int(((q.norm(dim=-1) > CAP) & mask).sum()),
            "newly_over_cap": int(((radius <= CAP) & (q.norm(dim=-1) > CAP) & mask).sum()),
            "returned_inside_cap": int(((radius > CAP) & (q.norm(dim=-1) <= CAP) & mask).sum()),
        }
    return result


def analyze_snapshot(payload):
    s = payload["actual_optimizer_step"]
    p, g, m, v = (s[key].detach().cpu().double() for key in
                  ("parameter", "gradient", "exp_avg", "exp_avg_sq"))
    b1, b2 = s["betas"]
    step = int(s["step_before"]) + 1
    sqrt_vhat = ((b2 * v + (1 - b2) * g.square()) / (1 - b2 ** step)).sqrt()
    mhat = (b1 * m + (1 - b1) * g) / (1 - b1 ** step)
    native = clone_step(s)
    actual = s["parameter_after"].detach().cpu()
    replay_error = float((native.double() - actual.double()).abs().max())
    if replay_error > 2e-6:
        raise ValueError(f"Native CPU AdamW replay error {replay_error}")
    u = p / p.norm(dim=-1, keepdim=True)
    result = {
        "epoch": int(payload["epoch"]), "step_in_epoch": int(payload["step"]),
        "adam_step_before": int(s["step_before"]),
        "tensor_dtypes": {key: str(s[key].dtype) for key in
                          ("parameter", "gradient", "exp_avg", "exp_avg_sq", "parameter_after")},
        "hyperparameters": {"lr": float(s["lr"]), "eps": float(s["eps"]),
                            "betas": list(s["betas"]), "weight_decay": float(s["weight_decay"])},
        "native_replay_max_coordinate_error": replay_error,
        "parameter_norm": stats(p.norm(dim=-1)),
        "inside_gap_to_cap": stats((CAP - p.norm(dim=-1))[p.norm(dim=-1) <= CAP]),
        "current_g_norm": stats(g.norm(dim=-1)),
        "current_signed_radial_g": stats((g * u).sum(-1)),
        "history_signed_radial_m": stats((m * u).sum(-1)),
        "sqrt_vhat": stats(sqrt_vhat), "abs_mhat": stats(mhat.abs()),
        "sqrt_vhat_below_eps_fraction": {
            "1e-8": float((sqrt_vhat < 1e-8).double().mean()),
            "1e-4": float((sqrt_vhat < 1e-4).double().mean())},
        "eps1e4_over_sqrt_vhat_plus_eps1e4": stats(1e-4 / (sqrt_vhat + 1e-4)),
        "variants": {},
    }
    variants = {"native": {}, "eps1e4_only": {"eps": 1e-4},
                "lr_half_only": {"lr_scale": .5}, "lr_tenth_only": {"lr_scale": .1}}
    for new_lambda in (.1, .25, 1.):
        scale = new_lambda / .5
        tag = str(new_lambda).replace(".", "p")
        variants[f"lambda{tag}_current_only"] = {"current_scale": scale}
        variants[f"lambda{tag}_all_gradient_history"] = {"current_scale": scale, "history_scale": scale}
    for name, kwargs in variants.items():
        after = native if name == "native" else clone_step(s, **kwargs)
        result["variants"][name] = update_summary(p, after, native_after=native)
    return result


def original_state_summary(array_path, metadata_path):
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with np.load(array_path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    step = int(values["step"])
    b2 = float(meta["proxy_optimizer_group"]["betas"][1])
    sqrt_vhat = np.sqrt(values["exp_avg_sq"].astype(np.float64) / (1 - b2 ** step))
    wanted = ("model", "emb", "use_fp16", "batch_size", "world_size", "epochs", "lr", "lr_decay",
              "lr_decay_gamma", "min_lr_scale", "weight_decay", "lambda1", "lambda2", "hyp_c", "clip_r")
    return {"label": "original_Cars_best", "epoch_metadata": int(meta["epoch"]), "step": step,
            "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
            "tensor_dtypes": {key: str(value.dtype) for key, value in values.items()},
            "shape": list(values["proxy_tangent"].shape),
            "args": {key: meta["args"][key] for key in wanted},
            "optimizer_group": meta["proxy_optimizer_group"],
            "initial_lcas_lr": float(meta["args"]["lr"]) * float(meta["args"]["batch_size"]) *
                               int(meta["args"]["world_size"]) / 180 * float(meta["proxy_optimizer_group"]["lr_scale"]),
            "stored_sqrt_vhat_before_next_gradient": stats(sqrt_vhat),
            "stored_sqrt_vhat_below_eps_fraction": float((sqrt_vhat < meta["proxy_optimizer_group"]["eps"]).mean()),
            "limit": "No current gradient is present. Stored-state denominators are descriptive, not matched training replay."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--original-state", type=Path)
    parser.add_argument("--original-metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new aggregate output")
    if args.original_state is None and args.snapshot_dir is None:
        parser.error("At least one data source is required")
    if bool(args.original_state) != bool(args.original_metadata):
        parser.error("Original state and metadata must be given together")
    torch.set_num_threads(2)
    result = {"format": "v9-adam-scale-audit-1", "device": "cpu",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_updates": 0, "encoder_forwards": 0, "test_forwards": 0,
              "parameter_arrays_exported": False,
              "variant_definitions": {
                  "current_only": "Change only the present HIER gradient, preserving native old m/v; not a new training trajectory.",
                  "all_gradient_history": "Scale current g and old m by s=lambda_new/.5, old v by s^2. Exact rescaling identity at the same assumed gradient history; not a rerun because parameters and gradient evaluation points are held fixed.",
                  "lr_half_only": "Actual fixed-state one-step LR .005 versus .01 initially; WD scales with LR as AdamW specifies.",
                  "eps1e4_only": "Actual fixed-state one-step eps change; old history preserved."},
              "snapshots": []}
    if args.snapshot_dir is not None:
        for epoch, step in SNAPSHOTS:
            payload = torch.load(args.snapshot_dir / f"proxy_cause_e{epoch:03d}_s{step:03d}.pth",
                                 map_location="cpu", weights_only=False)
            result["snapshots"].append(analyze_snapshot(payload))
    if args.original_state:
        result["original_cars"] = original_state_summary(args.original_state, args.original_metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "completed", "snapshots": len(result["snapshots"]),
                      "original_cars": "original_cars" in result}))


if __name__ == "__main__":
    main()
