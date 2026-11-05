"""Replace the regularizer whole prefix with a selected feature part."""

import torch

from ..hycore_seg_v3.losses import HyCoReSegLoss as _Kernel
from ..hycore_seg_v5.losses import HyCoReSegLoss as _PairedLoss


def overwrite_whole_prefix(features, part_indices):
    """Return [selected part, original suffix] without changing input storage."""
    selected = features.gather(
        1, part_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return torch.cat((selected, features[:, part_indices.shape[1]:]), dim=1)


class PrefixKernel(_Kernel):
    def pool_whole(self, features, part_indices):
        original_mean = features.mean(dim=1)
        overwritten = overwrite_whole_prefix(features, part_indices)
        self._capture("whole_original_tangent", original_mean)
        return overwritten.mean(dim=1)


class HyCoReSegLoss(_PairedLoss):
    manifest_metadata = {
        "display_name": "TAN-R-PREFIX",
        "matched_reference": "S5-TAN-R; only regularizer whole prefix feature overwrite",
        "whole_regularizer_pooling": "mean(cat(gather(original_H,part_indices),original_H[:,K:]))",
        "part_selection": "one selection on original H before prefix overwrite",
        "overwrite_location": "regularizer whole feature rows only",
        "main_branch_overwrite": False,
        "raw_xyz_or_label_overwrite": False,
    }

    def __init__(self, alpha=.01, beta=.01, gamma_base=1000., delta=4.,
                 part_min=200, part_max=600, seed=4780, feature_dim=256):
        super().__init__(alpha, beta, gamma_base, delta, part_min, part_max,
                         seed, feature_dim, kernel_type=PrefixKernel)
