"""CPU one-step fixed-state AdamW counterfactuals, with anonymous output only.

These are local interventions on saved actual proxy states. They are not fresh
optimizers, training trajectories, or estimates of classification performance.
The current gradient and existing second moment/step remain matched throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


DEFAULT_SNAPSHOTS = "e001_s001,e001_s005,e001_s020,e001_s100,e001_s200,e005_s200,e010_s200"
VARIANTS = ("native", "eps_1e4_only", "history_m_radial_removed", "history_m_zero_keep_v_and_step")
CAP = 2.3


def scalar(value):
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def saved_tensor(snapshot, name):
    result = snapshot[name].detach().cpu().clone()
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError(f"Invalid saved tensor: {name}")
    return result


@torch.no_grad()
def one_step(snapshot, variant):
    """Return a private real-AdamW parameter clone after exactly one update."""
    if variant not in VARIANTS:
        raise ValueError("Unknown counterfactual variant")
    parameter = torch.nn.Parameter(saved_tensor(snapshot, "parameter"))
    gradient = saved_tensor(snapshot, "gradient")
    m = saved_tensor(snapshot, "exp_avg")
    v = saved_tensor(snapshot, "exp_avg_sq")
    if not (parameter.shape == gradient.shape == m.shape == v.shape):
        raise ValueError("Saved AdamW shapes differ")
    if bool((v < 0).any()):
        raise ValueError("Negative second moment")
    eps = scalar(snapshot["eps"])
    if variant == "eps_1e4_only":
        eps = 1e-4
    elif variant == "history_m_radial_removed":
        p64 = parameter.detach().double()
        direction = p64 / p64.norm(dim=-1, keepdim=True)
        m64 = m.double()
        m = (m64 - (m64 * direction).sum(-1, keepdim=True) * direction).to(m.dtype)
    elif variant == "history_m_zero_keep_v_and_step":
        m.zero_()
    optimizer = torch.optim.AdamW([parameter], lr=scalar(snapshot["lr"]),
                                  betas=tuple(map(float, snapshot["betas"])),
                                  eps=eps, weight_decay=scalar(snapshot["weight_decay"]),
                                  foreach=False)
    # Explicitly restore all accumulated state rather than resetting bias correction.
    optimizer.state[parameter] = {
        "step": torch.tensor(float(snapshot["step_before"]), dtype=torch.float32),
        "exp_avg": m, "exp_avg_sq": v,
    }
    parameter.grad = gradient
    optimizer.step()
    return parameter.detach().clone()


def mean(value, mask):
    return float(value[mask].mean()) if bool(mask.any()) else None


@torch.no_grad()
def summarize_update(before, after):
    p, q = before.detach().double(), after.detach().double()
    radius = p.norm(dim=-1)
    if bool((radius == 0).any()):
        raise ValueError("Zero proxy makes radial accounting undefined")
    direction = p / radius[:, None]
    delta = q - p
    radial = (delta * direction).sum(-1)
    change = q.norm(dim=-1) - radius
    displacement = delta.norm(dim=-1)
    crossing = (radius <= CAP) & (q.norm(dim=-1) > CAP)
    returned = (radius > CAP) & (q.norm(dim=-1) <= CAP)
    masks = {"all": torch.ones_like(radius, dtype=torch.bool),
             "before_inside_cap": radius <= CAP,
             "before_over_cap": radius > CAP}
    output = {}
    for cohort, mask in masks.items():
        output[cohort] = {
            "count": int(mask.sum()),
            "mean_exact_norm_change": mean(change, mask),
            "mean_signed_radial_step": mean(radial, mask),
            "mean_finite_step_excess": mean(change - radial, mask),
            "mean_displacement_norm": mean(displacement, mask),
            "norm_outward_count": int(((change > 0) & mask).sum()),
            "newly_over_cap": int((crossing & mask).sum()),
            "returned_inside_cap": int((returned & mask).sum()),
            "after_over_cap": int(((q.norm(dim=-1) > CAP) & mask).sum()),
        }
    return output


@torch.no_grad()
def analyze_snapshot(payload, *, tolerance=2e-6):
    snapshot = payload["actual_optimizer_step"]
    baseline = one_step(snapshot, "native")
    actual = saved_tensor(snapshot, "parameter_after")
    error = (baseline.double() - actual.double()).abs()
    worst = float(error.max())
    if worst > tolerance:
        raise ValueError(f"Native CPU AdamW replay differs from recorded actual step: {worst:.9g}")
    before = saved_tensor(snapshot, "parameter")
    output = {
        "epoch": int(payload["epoch"]), "step_in_epoch": int(payload["step"]),
        "adam_step_before": int(snapshot["step_before"]),
        "hyperparameters": {"lr": scalar(snapshot["lr"]),
                            "betas": list(map(float, snapshot["betas"])),
                            "eps_native": scalar(snapshot["eps"]),
                            "weight_decay": scalar(snapshot["weight_decay"])},
        "baseline_vs_recorded_actual": {
            "max_coordinate_error": worst,
            "mean_row_error_norm": float(error.norm(dim=-1).mean()),
            "tolerance": tolerance, "passed": True,
        },
        "recorded_actual": summarize_update(before, actual), "variants": {},
    }
    for variant in VARIANTS:
        after = baseline if variant == "native" else one_step(snapshot, variant)
        output["variants"][variant] = summarize_update(before, after)
    return output


def selected_steps(text):
    output = []
    for token in text.split(","):
        matched = re.fullmatch(r"e(\d+)_?s(\d+)", token.strip())
        if matched is None:
            raise ValueError("Snapshots must have form e001_s001, comma-separated")
        epoch, step = map(int, matched.groups())
        if epoch < 1 or step < 1 or (epoch, step) in output:
            raise ValueError("Snapshot selections must be positive and distinct")
        output.append((epoch, step))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--baseline-tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if args.threads < 1 or args.baseline_tolerance <= 0:
        parser.error("Threads and tolerance must be positive")
    if args.out.exists():
        raise FileExistsError("Output already exists; use a new anonymous JSON")
    torch.set_num_threads(args.threads)
    result = {
        "format": "v9-one-step-fixed-state-adamw-counterfactuals-1", "device": "cpu",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_updates": 0, "encoder_forwards": 0, "test_forwards": 0,
        "parameter_arrays_exported": False,
        "scope": "One private CPU AdamW clone step per saved state and variant; no retraining",
        "variant_definitions": {
            "native": "Same saved parameter, current gradient, historical m/v, step and hyperparameters",
            "eps_1e4_only": "Change eps to 1e-4 only; preserve old m/v generated by native training",
            "history_m_radial_removed": "Remove saved m's radial component at current parameter before this step; keep g/v/step/hyperparameters",
            "history_m_zero_keep_v_and_step": "Zero historical m only; keep accumulated v and step/bias correction; not a fresh optimizer",
        },
        "cohort_definition": "Raw norm<=2.3 versus raw norm>2.3 before each saved step",
        "limits": "Counterfactuals hold current gradient fixed and do not predict multi-step training or accuracy",
        "snapshots": [], "unavailable": [],
    }
    for epoch, step in selected_steps(args.snapshots):
        path = args.run_dir / f"proxy_cause_e{epoch:03d}_s{step:03d}.pth"
        if not path.exists():
            result["unavailable"].append({"epoch": epoch, "step_in_epoch": step})
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (int(payload["epoch"]), int(payload["step"])) != (epoch, step):
            raise ValueError("Snapshot contents and requested epoch/step differ")
        result["snapshots"].append(analyze_snapshot(payload, tolerance=args.baseline_tolerance))
    result["available_snapshot_count"] = len(result["snapshots"])
    result["private_clone_optimizer_steps"] = len(VARIANTS) * len(result["snapshots"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "available_snapshot_count": len(result["snapshots"]),
                      "unavailable_count": len(result["unavailable"]),
                      "private_clone_optimizer_steps": result["private_clone_optimizer_steps"]}))


if __name__ == "__main__":
    main()
