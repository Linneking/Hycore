"""Versioned two-map HyCoRe tail with the original Euclidean Conv9 block.

Both user-authorized arms preserve the original dp1/Conv9/BN9/LeakyReLU,
replacing only Conv10/Conv11 with the official 256->128->50 Mobius maps.
The regularizer receives H after the preserved Conv9 block, before EXP.
"""

from types import SimpleNamespace

import torch

from segmentation_ShapeNetPart.hycore_seg_v1.vendor.dgcnn_model import DGCNN_partseg
from segmentation_ShapeNetPart.hycore_seg_v2.model import _CommunityNativePartSeg


class NativeTail2(_CommunityNativePartSeg):
    """D/E: keep dp1 and Euclidean Conv9; use two official Mobius maps.

    XYZ [B,3,N] and category one-hot [B,16] produce logits [B,50,N].
    ``return_features=True`` additionally returns post-Conv9 H [B,N,256].
    """

    def __init__(self, arm="d", k=40, dropout=.5):
        arm = str(arm).lower()
        if arm not in {"d", "e"}:
            raise ValueError("The tail2 v5 package only defines arms d and e")
        original_rng = torch.get_rng_state()
        # Reproduce v3's full original constructor and surviving M2/M3 states.
        super().__init__(k=k, dropout=dropout)
        # Recover the original Conv9+BN9 from the same constructor start without
        # consuming extra global RNG or changing any already-created parameter.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(original_rng)
            reference = DGCNN_partseg(
                SimpleNamespace(k=k, emb_dims=1024, dropout=dropout), 50)
        self.bn9 = reference.bn9
        self.conv9 = reference.conv9
        del self.emb
        self.arm = arm
        self.architecture = "pseg3_tail2_v5_e9"
        self.has_m1 = False  # No same-width M; emb2/mlr are the two maps.
        self.has_euclidean_conv9 = True
        self.regularizer_feature_stage = "after_conv9_bn9_leaky"
        self.dropout_active = isinstance(self.dp1, torch.nn.Dropout) and self.dp1.p > 0
        self.monitor = None

    def _capture(self, name, tensor, tangent_input=None):
        if self.monitor is not None:
            self.monitor.capture(name, tensor, tangent_input=tangent_input)

    def forward_features(self, points, category):
        features = super().forward_features(points, category)
        self._capture("pre_conv9", features)
        features = self.conv9(features.transpose(1, 2).contiguous())
        return features.transpose(1, 2).contiguous()

    def forward_head(self, features):
        self._capture("h", features)
        ball_features = self.manifold.expmap0(features)
        self._capture("point_exp", ball_features, tangent_input=features)
        ball_features = self.emb2(ball_features)
        self._capture("m2", ball_features)
        logits = self.mlr(ball_features)
        self._capture("m3", logits)
        return logits

    def head_parameters(self):
        for layer in (self.emb2, self.mlr):
            yield from (parameter for parameter in layer.parameters() if parameter.requires_grad)
