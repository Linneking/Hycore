"""Observational monitoring with a real-valued final readout node.

The incoming EXP and first M retain the original ball diagnostics.  The final
q node is never interpreted as a ball point or assigned a boundary/depth.
"""

import math

import torch

from ..hycore_seg_v3.monitoring import GeometryMonitor as _GeometryMonitor
from ..hycore_seg_v3.monitoring import _scalar, _stats


class GeometryMonitor(_GeometryMonitor):
    def begin(self, enabled, epoch=0, step=0):
        super().begin(enabled, epoch, step)
        self._targets = None
        self._score_scale = 5.
        return self

    def capture(self, name, tensor, tangent_input=None):
        if name == "final_tangent" and tangent_input is not None:
            raise ValueError("Real-valued final scores cannot be labelled as EXP output")
        return super().capture(name, tensor, tangent_input=tangent_input)

    def capture_targets(self, targets, scale=5.):
        if not self.enabled:
            return
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("Score scale must be finite and positive")
        scores = self.nodes.get("final_tangent")
        if scores is None or targets.shape != scores.shape[:-1]:
            raise ValueError("Targets must match the final score batch and point dimensions")
        self._targets = targets.detach()
        self._score_scale = float(scale)

    @torch.no_grad()
    def summarize(self):
        result = super().summarize()
        scores = self.nodes.get("final_tangent")
        if scores is None:
            return result
        detached = scores.detach()
        stats = result["geometry"]["final_tangent"]
        stats.update(coordinate_space="origin_tangent_real_scores",
                     logit_min=_scalar(detached.min()),
                     logit_max=_scalar(detached.max()),
                     logit_mean=_scalar(detached.double().mean()),
                     logit_std=_scalar(detached.double().std(unbiased=False)),
                     supervision_scale=self._score_scale)
        probabilities = (self._score_scale * detached).softmax(dim=-1)
        stats["scaled_max_class_probability"] = _stats(probabilities.amax(dim=-1))
        if self._targets is not None:
            correct = probabilities.gather(-1, self._targets.long().unsqueeze(-1)).squeeze(-1)
            stats["scaled_correct_class_probability"] = _stats(correct)
        return result

    def clear(self):
        super().clear()
        self._targets = None
        self._score_scale = 5.
