"""HyCoRe-seg losses: shared features, Euclidean mean, then expmap0."""

import torch
from torch import nn
from torch.nn import functional as F


def choose_parts(features, generator, n_min=200, n_max=600):
    """One center/object and a shared uniform integer part size per batch.

    Indices are discrete; gradients through gathered feature values are kept.
    Ties are resolved by stable point-index ordering. The center is included.
    """
    b, n, _ = features.shape
    if not 1 <= n_min <= n_max <= n:
        raise ValueError((n_min, n_max, n))
    size = int(torch.randint(n_min, n_max + 1, (), generator=generator))
    centers = torch.randint(n, (b,), generator=generator).to(features.device)
    with torch.no_grad():
        center_values = features[torch.arange(b, device=features.device), centers]
        distances = (features.detach() - center_values[:, None]).square().sum(-1)
        # A center must belong even when all features coincide.
        distances[torch.arange(b, device=features.device), centers] = -1.0
        indices = distances.argsort(dim=-1, stable=True)[:, :size]
    selected = features.gather(1, indices[:, :, None].expand(-1, -1, features.shape[-1]))
    return selected.mean(1), indices, centers, size


def negative_indices(categories, generator):
    labels = categories.detach().cpu().reshape(-1)
    result = torch.zeros(len(labels), dtype=torch.long)
    valid = torch.zeros(len(labels), dtype=torch.bool)
    for i in range(len(labels)):
        candidates = (labels != labels[i]).nonzero().reshape(-1)
        if len(candidates):
            result[i] = candidates[int(torch.randint(len(candidates), (), generator=generator))]
            valid[i] = True
    return result.to(categories.device), valid.to(categories.device)


def relation_losses(ball, whole, part, negative, valid, gamma, delta):
    whole_depth, part_depth = ball.dist0(whole), ball.dist0(part)
    radial_values = F.relu(whole_depth - part_depth + gamma)
    positive_distance = ball.dist(whole, part)
    negative_distance = ball.dist(whole, part[negative])
    contrast_values = F.relu(positive_distance - negative_distance + delta)
    radial = radial_values.mean()
    contrast = contrast_values[valid].mean() if valid.any() else whole.sum() * 0.0
    diagnostics = {
        "hier": float(radial.detach()), "contr": float(contrast.detach()),
        "valid_negative": int(valid.sum()), "objects": len(whole),
        "whole_depth": float(whole_depth.detach().mean()),
        "part_depth": float(part_depth.detach().mean()),
        "depth_gap": float((part_depth - whole_depth).detach().mean()),
        "hier_satisfied": float((radial_values.detach() <= 1e-6).float().mean()),
        "contr_satisfied": float((contrast_values[valid].detach() <= 1e-6).float().mean()) if valid.any() else None,
        "whole_projected": float((whole.detach().norm(dim=-1) >= .996 - 1e-6).float().mean()),
        "part_projected": float((part.detach().norm(dim=-1) >= .996 - 1e-6).float().mean()),
    }
    return radial, contrast, diagnostics


class HyCoReSegLoss(nn.Module):
    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780):
        super().__init__()
        import geoopt

        self.ball = geoopt.PoincareBall(c=1.0)
        self.alpha, self.beta = alpha, beta
        self.gamma_base, self.delta = gamma_base, delta
        self.part_min, self.part_max = part_min, part_max
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 100003)

    def forward(self, features, categories):
        if self.alpha == 0 and self.beta == 0:
            return features.sum() * 0.0, {"disabled": True}
        pooled, indices, centers, size = choose_parts(
            features, self.generator, self.part_min, self.part_max)
        whole = self.ball.expmap0(features.mean(1))
        part = self.ball.expmap0(pooled)
        negative, valid = negative_indices(categories, self.generator)
        radial, contrast, info = relation_losses(
            self.ball, whole, part, negative, valid, self.gamma_base / size, self.delta)
        info.update(part_points=size, gamma=self.gamma_base / size, delta=self.delta)
        return self.alpha * contrast + self.beta * radial, info
