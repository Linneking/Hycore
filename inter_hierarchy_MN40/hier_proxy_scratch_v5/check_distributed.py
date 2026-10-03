"""CPU Gloo check: dual-rank global64 gradient equals single global64.

This bounded engineering test uses a small encoder with frozen evaluation
BN statistics. Local training BN uses32 rows and is intentionally NOT
mathematically equivalent to one64-row training BN. No dataset/GPU required.
Run as a module from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from .base_protocol import smoothed_ce
from .distributed import gather_with_grad, gather_without_grad


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4)
        self.bn = nn.BatchNorm1d(4)
        self.classifier = nn.Linear(4, 40)

    def forward(self, points):
        mu = self.bn(self.linear(points)).tanh() * 0.05
        nu = self.bn(self.linear(points * 0.7 + 0.01)).tanh() * 0.04
        return {"mu": mu, "nu": nu, "logits": self.classifier(mu)}


def fixed_inputs():
    generator = torch.Generator().manual_seed(2244)
    points = torch.randn(64, 5, generator=generator, dtype=torch.float64)
    labels = torch.arange(32).repeat_interleave(2)
    return points, labels


def fixed_model():
    torch.manual_seed(3344)
    return TinyEncoder().double().eval()


def joint_objective(mu, nu, logits, labels):
    """Three different gradient paths, with global flip and cross-rank pairs.

    Analytic test losses are deliberately small and do not require Geoopt.
    Source HyCoRe expressions are checked separately by base_protocol.
    """
    ce = smoothed_ce(logits, labels)
    intra = ((mu - nu).square().sum(-1) -
             (mu - nu.flip(0)).square().sum(-1) + 4).relu().mean()
    inter = ((mu - mu.roll(33, 0)).square().sum(-1) +
             (mu[:, None] - mu[None, :]).square().sum(-1).mean(1)).mean()
    return ce + 0.01 * intra + 0.03 * inter


def parameter_gradient_vector(model):
    pieces = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("the gradient check must exercise every parameter")
        pieces.append(parameter.grad.reshape(-1))
    return torch.cat(pieces).detach().cpu()


def _worker(rank, world, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=Path(rendezvous).as_uri())
    try:
        model = DDP(fixed_model())
        points, labels = fixed_inputs()
        start, end = rank * 32, (rank + 1) * 32
        outputs = model(points[start:end])
        outputs["mu"].retain_grad()
        outputs["nu"].retain_grad()
        mu = gather_with_grad(outputs["mu"])
        nu = gather_with_grad(outputs["nu"])
        logits = gather_with_grad(outputs["logits"])
        gold = gather_without_grad(labels[start:end])
        loss = joint_objective(mu, nu, logits, gold)
        loss.backward()
        payload = {"rank": rank, "loss": float(loss),
                   "parameter_grad": parameter_gradient_vector(model).tolist(),
                   # Feature gradients SUM across duplicate global-loss
                   # evaluations; DDP averages only parameter gradients.
                   "local_mu_grad": outputs["mu"].grad.tolist(),
                   "local_nu_grad": outputs["nu"].grad.tolist(),
                   "gathered_labels": gold.tolist()}
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def run_check(tolerance=1e-10):
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("CPU Gloo is required for this distributed check")
    points, labels = fixed_inputs()
    model = fixed_model()
    outputs = model(points)
    outputs["mu"].retain_grad()
    outputs["nu"].retain_grad()
    reference_loss = joint_objective(outputs["mu"], outputs["nu"], outputs["logits"], labels)
    reference_loss.backward()
    parameter_reference = parameter_gradient_vector(model)
    with tempfile.TemporaryDirectory(prefix="hycore_v5_gloo_") as directory:
        rendezvous = str(Path(directory, "rendezvous").resolve())
        mp.spawn(_worker, args=(2, rendezvous, directory), nprocs=2, join=True)
        results = [json.loads(Path(directory, f"rank{r}.json").read_text(encoding="utf-8"))
                   for r in range(2)]
    report = {"backend": "gloo", "world_size": 2, "global_batch": 64,
              "local_batch": 32, "bn": "evaluation statistics fixed",
              "objective": "global CE + analytic intra(global flip) + analytic inter",
              "loss_extra_world_size_factor": 1,
              "feature_backward_reduction": "SUM",
              "parameter_backward_reduction": "DDP mean",
              "tolerance": tolerance, "ranks": []}
    for item in results:
        r = item["rank"]
        start, end = r * 32, (r + 1) * 32
        grad = torch.tensor(item["parameter_grad"], dtype=torch.float64)
        mu_grad = torch.tensor(item["local_mu_grad"], dtype=torch.float64)
        nu_grad = torch.tensor(item["local_nu_grad"], dtype=torch.float64)
        row = {"rank": r, "loss_absolute_error": abs(item["loss"] - float(reference_loss)),
               "parameter_gradient_max_error": float((grad - parameter_reference).abs().max()),
               "local_mu_gradient_div_world_max_error": float(
                   (mu_grad / 2 - outputs["mu"].grad[start:end]).abs().max()),
               "local_nu_gradient_div_world_max_error": float(
                   (nu_grad / 2 - outputs["nu"].grad[start:end]).abs().max()),
               "global_label_order_correct": item["gathered_labels"] == labels.tolist()}
        row["passed"] = row["global_label_order_correct"] and all(
            row[key] <= tolerance for key in
            ("loss_absolute_error", "parameter_gradient_max_error",
             "local_mu_gradient_div_world_max_error", "local_nu_gradient_div_world_max_error"))
        report["ranks"].append(row)
    report["passed"] = all(row["passed"] for row in report["ranks"])
    if not report["passed"]:
        raise AssertionError(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("check output must use a new path")
    result = run_check()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
