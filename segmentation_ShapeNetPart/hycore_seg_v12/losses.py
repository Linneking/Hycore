"""Matched single-component objectives over the unchanged v3 kernel.

The positive kernel still computes both raw diagnostics and consumes identical
part/negative RNG streams. Only the selected component enters the training graph.
"""
import torch
from torch import nn

from ..hycore_seg_v3.losses import HyCoReSegLoss as _V3Loss


class HyCoReSegLoss(nn.Module):
    def __init__(self, alpha=.01, beta=0., gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=256):
        super().__init__()
        if (alpha, beta) not in {(.01, 0.), (0., .01)}:
            raise ValueError("Single-component coefficients must be (.01,0) or (0,.01)")
        self.alpha, self.beta = float(alpha), float(beta)
        self.regularizers_active = True
        self.kernel = _V3Loss(.01, .01, gamma_base, delta, part_min, part_max,
                              seed, feature_dim)
        self.monitor = None
        self.last_components = {}

    @property
    def generator(self):
        return self.kernel.generator

    @property
    def negative_generator(self):
        return self.kernel.negative_generator

    def forward(self, features, categories):
        self.kernel.monitor = self.monitor
        _, diagnostics = self.kernel(features, categories)
        raw = self.kernel.last_components
        # Never use zero * inactive_loss: it would retain an inactive autograd
        # path and make observational gradient flags misrepresent the objective.
        zero = features.new_zeros(())
        self.last_components = {
            "contr": raw["contr"] if self.alpha else zero,
            "hier": raw["hier"] if self.beta else zero,
        }
        active = "contr" if self.alpha else "hier"
        value = self.last_components[active]
        diagnostics.update(
            alpha=self.alpha, beta=self.beta, regularizers_active=True,
            regularizer_diagnostics_only=False,
            active_regularizer_component=active,
            contr_training_active=bool(self.alpha), hier_training_active=bool(self.beta),
            active_weighted_contr=float(self.last_components["contr"].detach()),
            active_weighted_hier=float(self.last_components["hier"].detach()),
            inactive_component_role="raw_observation_only_no_training_graph",
        )
        self.kernel.last_components = {}
        return value, diagnostics
