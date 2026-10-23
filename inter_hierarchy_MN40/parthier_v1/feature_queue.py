"""Detached FIFO part candidates, ordered by global row then crop scale.

Callers supply the same gathered global batch on every rank. This class
does not gather, update a model, or perform another encoder forward.
"""
from __future__ import annotations

import torch


class PartFeatureQueue:
    VERSION = 1

    def __init__(self, capacity: int = 2048):
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError("queue capacity must be a nonnegative integer")
        self.capacity = capacity
        self.features = torch.empty((0, 0))
        self.object_ids = torch.empty(0, dtype=torch.long)
        self.scales = torch.empty(0, dtype=torch.long)
        self.steps = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.object_ids)

    @staticmethod
    def _validate(parts, object_ids, step):
        if parts.ndim != 3 or parts.shape[1] != 3 or parts.shape[2] < 2:
            raise ValueError("parts must be [B,3,D>=2]")
        if parts.dtype not in (torch.float32, torch.float64):
            raise ValueError("part features must be FP32 or FP64")
        if not bool(torch.isfinite(parts).all()):
            raise ValueError("part features must be finite")
        if not bool((parts.square().sum(-1) < 1).all()):
            raise ValueError("part features must be inside the c=1 ball")
        ids = torch.as_tensor(object_ids, device=parts.device)
        if ids.ndim != 1 or len(ids) != len(parts) or ids.dtype != torch.long:
            raise ValueError("one int64 object ID is required per global row")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return ids

    @torch.no_grad()
    def enqueue(self, parts, object_ids, step: int):
        ids = self._validate(parts, object_ids, step)
        if self.capacity == 0 or not len(parts):
            return
        if len(self) and self.features.shape[1] != parts.shape[2]:
            raise ValueError("queue embedding dimension changed")
        if len(self) and step < int(self.steps[-1]):
            raise ValueError("queue steps must not move backwards")
        device = parts.device
        new = {
            "features": parts.detach().reshape(-1, parts.shape[2]).clone(),
            "object_ids": ids.detach().repeat_interleave(3).clone(),
            "scales": torch.arange(3, device=device).repeat(len(parts)),
            "steps": torch.full((3 * len(parts),), step, dtype=torch.long, device=device),
        }
        for name, value in new.items():
            old = getattr(self, name)
            if len(self):
                old = old.to(device=device, dtype=value.dtype)
                value = torch.cat((old, value), dim=0)
            setattr(self, name, value[-self.capacity:].detach().clone())

    def entries(self, *, device=None, dtype=None):
        """Read detached candidates; returned tensors must not be mutated."""
        return {
            "features": self.features.to(device=device, dtype=dtype),
            "object_ids": self.object_ids.to(device=device),
            "scales": self.scales.to(device=device),
            "steps": self.steps.to(device=device),
        }

    def state_dict(self):
        """Portable snapshot, independent of live storage and autograd."""
        return {"version": self.VERSION, "capacity": self.capacity, **{
            name: getattr(self, name).detach().cpu().clone()
            for name in ("features", "object_ids", "scales", "steps")}}

    def load_state_dict(self, state):
        if state.get("version") != self.VERSION or state.get("capacity") != self.capacity:
            raise ValueError("queue state version/capacity differs from this run")
        names = ("features", "object_ids", "scales", "steps")
        if any(not isinstance(state.get(name), torch.Tensor) for name in names):
            raise ValueError("queue state must contain feature and metadata tensors")
        tensors = {name: state[name].detach().cpu().clone() for name in names}
        features = tensors["features"]
        if features.ndim != 2:
            raise ValueError("invalid queue feature dimensions")
        count = len(features)
        if count > self.capacity:
            raise ValueError("invalid queue feature dimensions/capacity")
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError("queue features must be FP32 or FP64")
        if count and features.shape[1] < 2:
            raise ValueError("invalid queue embedding dimension")
        if not bool(torch.isfinite(features).all()) or not bool((features.square().sum(-1) < 1).all()):
            raise ValueError("invalid c=1 queue features")
        for name in names[1:]:
            if tensors[name].shape != (count,) or tensors[name].dtype != torch.long:
                raise ValueError("queue metadata must be aligned int64 vectors")
        if bool(((tensors["scales"] < 0) | (tensors["scales"] > 2)).any()):
            raise ValueError("queue scales must be 0,1,2")
        if bool((tensors["steps"] < 0).any()) or bool((tensors["steps"][1:] < tensors["steps"][:-1]).any()):
            raise ValueError("queue step metadata must be nonnegative and ordered")
        for name, value in tensors.items():
            setattr(self, name, value)


FeatureQueue = PartFeatureQueue
