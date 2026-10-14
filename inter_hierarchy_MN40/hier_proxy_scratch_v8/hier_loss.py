"""Official c=.1/cap/hook HIER; V6 self-k correctness fix retained."""
from __future__ import annotations
import math
import torch
from torch import nn
from .geometry import C, CLIP_R, to_ball, official_distance, geometry_stats, assert_output_geometry
from ..hier_proxy_scratch_v5.relations import mine_reciprocal_triplets
from ..hier_proxy_scratch_v6.hier_loss import ghhc_loss
from ..hier_proxy_scratch_v6.structure_monitor import ancestor_summary


@torch.no_grad()
def ancestor_summary_c(points, proxies, triplets, details):
    report = ancestor_summary(points * math.sqrt(C), proxies * math.sqrt(C), triplets, details)
    for moment in report["moments"].values():
        moment["sum"] /= math.sqrt(C)
        moment["sum_squares"] /= C
        for key in ("min", "max"):
            if moment[key] is not None:
                moment[key] /= math.sqrt(C)
    report["depth_units"] = "origin hyperbolic distance at c=.1"
    return report


class HIERLoss(nn.Module):
    def __init__(self, num_proxies=512, dim=256, seed=22):
        super().__init__()
        self.num_proxies, self.dim, self.c = num_proxies, dim, C
        self.margin, self.tau, self.clip_r = .1, .1, CLIP_R
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn(num_proxies, dim, generator=generator) / math.sqrt(dim) * CLIP_R * .9
        self.tangent_proxies = nn.Parameter(values)

    def proxies(self):
        return to_ball(self.tangent_proxies)

    @torch.no_grad()
    def assert_geometry(self):
        if not bool(torch.isfinite(self.tangent_proxies).all()):
            raise RuntimeError("Nonfinite proxy parameters")
        assert_output_geometry(self.proxies())

    @torch.no_grad()
    def project_after_step(self, optimizer):
        # Official HIER caps forward outputs, not saved parameters or Adam state.
        self.assert_geometry()
        norms = self.tangent_proxies.detach().norm(dim=-1)
        return {"enabled": False, "projected_count": 0,
                "parameter_tangent_max": float(norms.max()),
                "forward_clip_count": int((norms > CLIP_R).sum())}

    def forward(self, mu, labels, topk=20, proxy_topk=None, seed=22, data_ids=None,
                exclude_self_negative=True, t_per_anchor=50, generator=None,
                return_mining=False, return_components=False, monitor_structure=True):
        if not exclude_self_negative:
            raise ValueError("V8 retains sample and proxy self-k exclusion")
        proxies = self.proxies()
        assert_output_geometry(mu)
        ds = official_distance(mu)
        dp = official_distance(proxies)
        d_sp = official_distance(mu, proxies)
        similarity = torch.exp(-ds.detach()) + (labels[:, None] == labels[None, :]).to(mu.dtype)
        sample = mine_reciprocal_triplets(similarity, topk, t_per_anchor, seed, True,
                                         labels=labels, data_ids=data_ids)
        mined = mine_reciprocal_triplets(torch.exp(-dp.detach()), proxy_topk or topk,
                                        t_per_anchor, seed + 1, True)
        sample_loss, ss, sd = ghhc_loss(d_sp, sample["triplets"], self.tau, self.margin,
                                       generator, return_details=True)
        proxy_loss, ps, pd = ghhc_loss(dp, mined["triplets"], self.tau, self.margin,
                                      generator, return_details=True)
        stats = {"sample": {**sample["stats"], **ss}, "proxy": {**mined["stats"], **ps},
                 "sample_loss": float(sample_loss.detach()), "proxy_loss": float(proxy_loss.detach()),
                 "geometry": {"whole": geometry_stats(mu.detach()),
                              "proxy": geometry_stats(proxies.detach()),
                              "proxy_numerical_ball_project_count": 0}}
        if monitor_structure:
            stats["structure"] = {"sample": ancestor_summary_c(mu, proxies, sample["triplets"], sd),
                                  "proxy": ancestor_summary_c(proxies, proxies, mined["triplets"], pd)}
        if return_components:
            stats["_components"] = {"sample_loss": sample_loss, "proxy_loss": proxy_loss,
                                    "sample_details": sd, "proxy_details": pd}
        if return_mining:
            stats["_sample_mining"] = sample
        return sample_loss + proxy_loss, stats
