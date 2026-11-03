"""Two no-R tail interventions chosen after matched absence-error diagnosis."""
import torch
from torch.nn import functional as F

from ..hycore_seg_v10.model import NativeTail2 as _Reference


class NativeTail2(_Reference):
    def __init__(self, arm="h", k=40, dropout=.5, variant="bnleaky"):
        if variant not in {"bnleaky", "bias", "bnleaky_bias"}:
            raise ValueError("Unknown declared tail candidate")
        super().__init__(arm=arm, k=k, dropout=dropout)
        self.variant = variant
        if variant in {"bnleaky", "bnleaky_bias"}:
            self.tail_bn = torch.nn.BatchNorm1d(128)
            self.tail_activation = torch.nn.LeakyReLU(.2)
            self.readout_description = "q=linear(leaky(BN(logmap0(mu128))),mlr.weight)"
        if variant in {"bias", "bnleaky_bias"}:
            self.score_bias = torch.nn.Parameter(torch.zeros(50))
            self.readout_description = "q=linear(logmap0(mu128),mlr.weight)+score_bias50; zero_initial_bias"
        if variant == "bnleaky_bias":
            self.readout_description = "q=linear(leaky(BN(logmap0(mu128))),mlr.weight)+score_bias50"
        self.architecture = "pseg3_h_v13_tangent_" + variant

    def head_parameters(self):
        yield from super().head_parameters()
        if hasattr(self, "tail_bn"):
            yield from self.tail_bn.parameters()
        if hasattr(self, "score_bias"):
            yield self.score_bias

    def forward_head(self, features):
        self._capture("h", features)
        ball_features = self.manifold.expmap0(features)
        self._capture("point_exp", ball_features, tangent_input=features)
        ball_features = self.emb2(ball_features)
        self._capture("m2", ball_features)
        tangent = self.manifold.logmap0(ball_features)
        self._capture("tail_tangent", tangent)
        if hasattr(self, "tail_bn"):
            normalized = self.tail_bn(tangent.transpose(1, 2)).transpose(1, 2).contiguous()
            self._capture("tail_bn", normalized)
            tangent = self.tail_activation(normalized)
            self._capture("tail_activation", tangent)
        scores = F.linear(tangent, self.mlr.weight)
        if hasattr(self, "score_bias"):
            scores = scores + self.score_bias
        self._capture("final_tangent", scores)
        return scores
