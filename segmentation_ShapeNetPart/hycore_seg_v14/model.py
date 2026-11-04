"""Shape-conditioned tangent scores and a single late Mobius layer."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ..hycore_seg_v1.vendor.dgcnn_model import DGCNN_partseg
from ..hycore_seg_v10.model import NativeTail2 as _Reference


class NativeTail2(_Reference):
    def __init__(self, arm="h", k=40, dropout=.5, variant="shape"):
        if variant not in {"shape", "late_m"}:
            raise ValueError("Unknown structure variant")
        baseline_rng = torch.get_rng_state()
        super().__init__(arm=arm, k=k, dropout=dropout)
        self.variant = variant
        self.architecture = "pseg3_h_v14_" + variant
        if variant == "shape":
            # Preserve the matched training RNG stream and start at TAN logits.
            with torch.random.fork_rng(devices=[]):
                self.shape_bias = nn.Linear(256, 50)
            nn.init.zeros_(self.shape_bias.weight)
            nn.init.zeros_(self.shape_bias.bias)
            self.new_parameter_prefixes = ("shape_bias.",)
            self.readout_description = "TAN scores plus shape_bias(cat(mean(T),max(T)))"
        else:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(baseline_rng)
                original = DGCNN_partseg(SimpleNamespace(k=k, emb_dims=1024, dropout=dropout), 50)
            self.conv10 = original.conv10
            del self.emb2
            self.new_parameter_prefixes = ("conv10.",)
            self.head_feature_stage = "after_conv10_bn10_leaky"
            self.readout_coordinate_space = "poincare_ball_coordinates"
            self.readout_description = "conv10_bn_leaky -> expmap0 -> mobius_128_to_50"

    def head_parameters(self):
        if self.variant == "shape":
            yield from super().head_parameters()
            yield from self.shape_bias.parameters()
        else:
            yield from self.conv10.parameters()
            yield from self.mlr.parameters()

    def additional_parameter_groups(self):
        if self.variant == "shape":
            return {"shape_bias": dict(self.shape_bias.named_parameters())}
        return {"euclidean10": dict(self.conv10.named_parameters())}

    def run_metadata(self):
        if self.variant == "shape":
            return {"display_name": "TAN-SHAPE-R0", "final_output_space": "real_tangent_scores",
                    "matched_reference": "S5-TAN-R0"}
        return {"display_name": "LATE-M-R0", "final_output_space": "poincare_ball_coordinates",
                "matched_reference": "S5-R0"}

    def forward_head(self, features):
        self._capture("h", features)
        if self.variant == "late_m":
            embedding = self.conv10(features.transpose(1, 2).contiguous())
            embedding = embedding.transpose(1, 2).contiguous()
            self._capture("pre_exp_128", embedding)
            ball = self.manifold.expmap0(embedding)
            self._capture("point_exp", ball, tangent_input=embedding)
            scores = self.mlr(ball)
            self._capture("m3", scores)
            return scores

        ball = self.manifold.expmap0(features)
        self._capture("point_exp", ball, tangent_input=features)
        hidden = self.emb2(ball)
        self._capture("m2", hidden)
        tangent = self.manifold.logmap0(hidden)
        pooled = torch.cat((tangent.mean(dim=1), tangent.amax(dim=1)), dim=-1)
        offset = self.shape_bias(pooled)
        self._capture("shape_offset", offset)
        scores = F.linear(tangent, self.mlr.weight) + offset.unsqueeze(1)
        self._capture("final_tangent", scores)
        return scores
