"""V7 retains the V6 HIER kernel and changes proxy parameter geometry only."""
from __future__ import annotations
import torch
from ..hier_proxy_scratch_v6.hier_loss import HIERLoss as V6HIERLoss, expmap0_c1, ghhc_loss
from .proxy_safety import initialize_exact_depth, validate_geometry, project_proxy_parameter, assert_parameter_safe


class HIERLoss(V6HIERLoss):
    def __init__(self, num_proxies=512, dim=256, c=1., margin=.1, tau=.1,
                 seed=22, initial_depth=5., max_depth=6., moment_policy="retain"):
        validate_geometry(initial_depth, max_depth)
        if moment_policy not in ("retain", "remove_outward"):
            raise ValueError("invalid first-moment policy")
        super().__init__(num_proxies=num_proxies, dim=dim, c=c, margin=margin,
                         tau=tau, seed=seed, clip_r=2.3)
        self.max_depth, self.initial_depth = float(max_depth), float(initial_depth)
        self.moment_policy = moment_policy
        initialize_exact_depth(self.tangent_proxies, initial_depth)

    @torch.no_grad()
    def set_initial_depth(self, depth):
        validate_geometry(depth, self.max_depth)
        initialize_exact_depth(self.tangent_proxies, depth)
        self.initial_depth = float(depth)

    @torch.no_grad()
    def project_after_step(self, optimizer):
        stats = project_proxy_parameter(self.tangent_proxies, optimizer,
                                        self.max_depth, self.moment_policy)
        assert_parameter_safe(self.tangent_proxies, self.max_depth)
        return stats
