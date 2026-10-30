"""F/G/H: matched two-map HyCoRe heads and explicit regularizer taps.

The original Euclidean Conv9/BN9/Leaky block is evaluated once per batch.
F/G regularize its input; H regularizes its output. G alone retains dp1.
"""

import torch

from segmentation_ShapeNetPart.hycore_seg_v2.model import _CommunityNativePartSeg
from segmentation_ShapeNetPart.hycore_seg_v5.model import NativeTail2 as _V5Tail2


class NativeTail2(_V5Tail2):
    """Keep original Conv9 and the official 256->128->50 Mobius maps.

    ``forward_features`` returns the main head's post-Conv9 features.
    ``forward(..., return_features=True)`` returns logits [B,50,N] and the
    arm-specific *regularizer source* [B,N,256]. Those features differ for
    F/G, so the head must always consume the separately computed post-Conv9 H.
    """

    def __init__(self, arm="f", k=40, dropout=.5):
        arm = str(arm).lower()
        if arm not in {"f", "g", "h"}:
            raise ValueError("The F/G/H v6 package only defines arms f, g and h")
        # The complete v5 constructor preserves original Conv9 and surviving
        # official Mobius states, including the global constructor RNG state.
        super().__init__(arm="d", k=k, dropout=dropout)
        if arm != "g":
            self.dp1 = torch.nn.Identity()
        self.arm = arm
        self.architecture = "pseg3_fgh_v6_e9"
        self.regularizer_feature_stage = (
            "after_conv9_bn9_leaky" if arm == "h" else "before_conv9")
        self.head_feature_stage = "after_conv9_bn9_leaky"
        self.dropout_active = isinstance(self.dp1, torch.nn.Dropout) and self.dp1.p > 0

    def _feature_pair(self, points, category):
        # Call the frozen community frontend directly: it already applies dp1
        # (Identity in F/H). Do not call v5.forward_features, which applies L9.
        before = _CommunityNativePartSeg.forward_features(self, points, category)
        self._capture("pre_conv9", before)
        after = self.conv9(before.transpose(1, 2).contiguous())
        after = after.transpose(1, 2).contiguous()
        self._capture("post_conv9", after)
        source = after if self.arm == "h" else before
        self._capture("regularizer_source", source)
        return after, source

    def forward_features(self, points, category):
        after, _ = self._feature_pair(points, category)
        return after

    def forward(self, points, category, return_features=False):
        after, source = self._feature_pair(points, category)
        # Inherited forward_head captures h=post-Conv9 and the EXP/M outputs.
        logits = self.forward_head(after).transpose(1, 2).contiguous()
        return (logits, source) if return_features else logits
