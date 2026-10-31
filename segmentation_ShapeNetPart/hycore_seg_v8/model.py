"""Experiment I model: H's main tail and a shared penultimate regularizer path.

The whole cloud is encoded once for segmentation CE.  A feature-neighborhood
part is selected from that whole pass, gathered as raw XYZ, and encoded by the
same STN, EdgeConv, Conv9/BN9/Leaky, EXP and first Mobius layer.  Both forward
passes run in train mode; the trainer must expect two BN updates per batch.
The final Mobius 128->50 classifier is used only by per-point CE.
"""

import torch

from segmentation_ShapeNetPart.hycore_seg_v2.model import _CommunityNativePartSeg
from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2 as _HTail


class NativeI(_HTail):
    """H's parameterization with separate whole and part regularizer encodings."""

    def __init__(self, k=40, dropout=.5):
        # Keep the exact H constructor and all named trainable parameters.
        super().__init__(arm="h", k=k, dropout=dropout)
        self.arm = "i"
        self.architecture = "pseg3_i_v8_reencoded_part"
        self.euclidean_feature_dim = 256
        self.feature_dim = 128  # Width of the two regularized ball embeddings.
        self.regularizer_feature_stage = "after_emb2_whole_and_reencoded_part"
        self.part_selection_stage = "after_conv9_bn9_leaky_whole"
        self.head_feature_stage = "after_conv9_bn9_leaky"

    def _encode_euclidean(self, points, category, branch):
        if branch not in {"whole", "part"}:
            raise ValueError("The branch must be whole or part")
        before = _CommunityNativePartSeg.forward_features(self, points, category)
        self._capture(branch + "_pre_conv9", before)
        after = self.conv9(before.transpose(1, 2).contiguous())
        after = after.transpose(1, 2).contiguous()
        self._capture(branch + "_post_conv9", after)
        if branch == "whole":
            # Retain H's existing monitor keys for whole-path comparisons.
            self._capture("pre_conv9", before)
            self._capture("post_conv9", after)
        return after

    def forward_features(self, points, category):
        """Return the full cloud's post-Conv9 per-point Euclidean features."""
        return self._encode_euclidean(points, category, "whole")

    def encode_part(self, points, category):
        """Encode selected raw XYZ with shared parameters and current BN mode.

        The trainer calls this in train mode, with autograd enabled, after the
        whole pass.  No evaluation-mode switch or no_grad belongs here.
        """
        return self._encode_euclidean(points, category, "part")

    def regularizer_embedding(self, features, branch):
        """Euclidean mean -> EXP -> shared M(256->128), omitting final MLR."""
        if branch not in {"whole", "part"}:
            raise ValueError("The branch must be whole or part")
        if features.ndim != 3 or features.shape[-1] != self.euclidean_feature_dim:
            raise ValueError("Expected per-point post-Conv9 features [B,N,256]")
        tangent = features.mean(dim=1)
        self._capture(branch + "_tangent", tangent)
        ball256 = self.manifold.expmap0(tangent)
        self._capture(branch + "_exp", ball256, tangent_input=tangent)
        mu128 = self.emb2(ball256)
        self._capture(branch + "_mu", mu128)
        return mu128

    def forward(self, points, category, return_features=False):
        features = self.forward_features(points, category)
        logits = self.forward_head(features).transpose(1, 2).contiguous()
        return (logits, features) if return_features else logits
