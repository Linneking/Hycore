"""Paired active/observation-only use of the frozen v3 regularizer kernel.

E still observes whole/part geometry and consumes the same independent sampling
RNG streams. Its diagnostic computation uses detached H under no_grad and never
adds a regularizer graph to CE. No model forward or BN update is repeated.
"""

import torch
from torch import nn

from segmentation_ShapeNetPart.hycore_seg_v3.losses import HyCoReSegLoss as _V3Loss


class HyCoReSegLoss(nn.Module):
    """Only the approved (.01,.01) and (0,0) regularizer settings are accepted."""

    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=256):
        super().__init__()
        if (alpha, beta) not in {(.01, .01), (0., 0.)}:
            raise ValueError("Paired D/E coefficients must be (.01,.01) or (0,0)")
        self.alpha, self.beta = float(alpha), float(beta)
        self.regularizers_active = bool(alpha or beta)
        # The same positive-coefficient kernel keeps the diagnostic calculations
        # and part/negative RNG consumption identical in both arms.
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
        if self.regularizers_active:
            value, diagnostics = self.kernel(features, categories)
            self.last_components = dict(self.kernel.last_components)
        else:
            with torch.no_grad():
                _, diagnostics = self.kernel(features.detach(), categories)
            value = features.new_zeros(())
            self.last_components = {"contr": value, "hier": value}
        diagnostics.update(
            alpha=self.alpha, beta=self.beta,
            regularizers_active=self.regularizers_active,
            regularizer_diagnostics_only=not self.regularizers_active,
            active_weighted_contr=float(self.last_components["contr"].detach()),
            active_weighted_hier=float(self.last_components["hier"].detach()),
        )
        # The public components describe active training losses only. In E the
        # raw `contr` and `hier` diagnostics are observations, not active losses.
        self.kernel.last_components = {}
        return value, diagnostics
