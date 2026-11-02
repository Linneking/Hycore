"""S5-TAN-R0: native H frontend with a real-valued final tangent readout.

Only the last forward operation changes.  The original last-layer parameters,
weight transport, constructor order and random-number consumption are retained.
"""

from torch.nn import functional as F

from ..hycore_seg_v9.model import NativeTail2 as _NativeTail2


class NativeTail2(_NativeTail2):
    def __init__(self, arm="h", k=40, dropout=.5):
        if str(arm).lower() != "h":
            raise ValueError("S5-TAN-R0 only defines the no-dp1 H main branch")
        super().__init__(arm="h", k=k, dropout=dropout)
        self.architecture = "pseg3_h_v10_final_tangent_readout"
        self.readout_coordinate_space = "origin_tangent_real_scores"

    def forward_head(self, features):
        self._capture("h", features)
        ball_features = self.manifold.expmap0(features)
        self._capture("point_exp", ball_features, tangent_input=features)
        ball_features = self.emb2(ball_features)
        self._capture("m2", ball_features)
        # Retain mlr._weight/_bias and its effective transported weight.  Unlike
        # mlr(ball_features), q has no final output EXP or ball projection.
        tangent_features = self.manifold.logmap0(ball_features)
        scores = F.linear(tangent_features, self.mlr.weight)
        self._capture("final_tangent", scores)
        return scores
