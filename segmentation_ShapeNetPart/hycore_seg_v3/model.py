"""The three fresh Pseg3 arms from proposal 105.

The complete v2 constructor always runs first.  Structural ablations therefore
do not shift the initialization of any surviving frontend or head parameter.
"""

from torch import nn

from segmentation_ShapeNetPart.hycore_seg_v2.model import _CommunityNativePartSeg


class NativePseg3(_CommunityNativePartSeg):
    """A: no dp1; B: no dp1/M1; C: the original Pseg3 structure.

    Inputs are XYZ [B,3,N] and object-category one-hot [B,16].  Outputs follow
    v2: logits [B,50,N], optionally accompanied by H [B,N,256].
    """

    def __init__(self, arm, k=40, dropout=.5):
        arm = str(arm).lower()
        if arm not in {"a", "b", "c"}:
            raise ValueError("Unknown Pseg3 v3 arm: " + arm)
        super().__init__(k=k, dropout=dropout)
        self.arm = arm
        self.architecture = "pseg3_v3_" + arm
        self.has_m1 = arm != "b"
        if arm in {"a", "b"}:
            self.dp1 = nn.Identity()
        if not self.has_m1:
            del self.emb
        # A passive observer: capture must never replace a training tensor.
        self.monitor = None

    def _capture(self, name, tensor, tangent_input=None):
        if self.monitor is not None:
            self.monitor.capture(name, tensor, tangent_input=tangent_input)

    def forward_head(self, features):
        self._capture("h", features)
        ball_features = self.manifold.expmap0(features)
        self._capture("point_exp", ball_features, tangent_input=features)
        if self.has_m1:
            ball_features = self.emb(ball_features)
            self._capture("m1", ball_features)
        ball_features = self.emb2(ball_features)
        self._capture("m2", ball_features)
        logits = self.mlr(ball_features)
        self._capture("m3", logits)
        return logits

    def head_parameters(self):
        layers = (self.emb, self.emb2, self.mlr) if self.has_m1 else (self.emb2, self.mlr)
        for layer in layers:
            yield from (parameter for parameter in layer.parameters() if parameter.requires_grad)
