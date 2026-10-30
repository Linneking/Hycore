"""Active F/G/H use of the frozen v3 whole/part regularizer kernel.

Every arm uses coefficients (.01,.01). The wrapper preserves the existing
independent part/negative RNG streams, diagnostics and active loss components.
Only the model-selected regularizer source changes across the approved arms.
"""

from segmentation_ShapeNetPart.hycore_seg_v5.losses import HyCoReSegLoss as _V5Loss


class HyCoReSegLoss(_V5Loss):
    """F/G/H all enable both original regularizers; no observation-only arm."""

    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=256):
        if (alpha, beta) != (.01, .01):
            raise ValueError("Approved F/G/H require alpha=beta=.01")
        super().__init__(alpha, beta, gamma_base, delta, part_min, part_max,
                         seed, feature_dim)
