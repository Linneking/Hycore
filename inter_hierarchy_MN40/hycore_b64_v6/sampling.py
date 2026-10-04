"""One global no-replacement permutation, split into consecutive rank halves.

The original HyCoRe shuffle/drop-last rule is adapted to global64 over two
local32 ranks. A private Python RNG performs exactly one shuffle per epoch;
its stream is independent of augmentation, workers and torch RNG state. This
is a versioned sampling adaptation, not the original torch RandomSampler
random sequence. There is no padding, replacement or per-epoch step cap.
"""

from __future__ import annotations

import hashlib
import json
import operator
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterator, Optional


SAMPLER_VERSION = "hycore-global64-python-shuffle-seed-plus-epoch-v6-1"


def _integer(value, name: str, *, minimum: int = 0) -> int:
    """Accept Python/NumPy integer scalars without silently truncating floats."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _digest(values) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class GlobalShuffleBatchPlan:
    """Dataset positions and corresponding source IDs in global batch order."""

    epoch: int
    step: int
    indices: tuple[int, ...]
    ids: tuple[int, ...]
    world_size: int = 2

    @property
    def global_batch_size(self) -> int:
        return len(self.indices)

    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.world_size

    def _rank_slice(self, rank: int) -> slice:
        use_rank = _integer(rank, "rank")
        if use_rank >= self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        start = use_rank * self.local_batch_size
        return slice(start, start + self.local_batch_size)

    def rank_indices(self, rank: int) -> tuple[int, ...]:
        return self.indices[self._rank_slice(rank)]

    def rank_ids(self, rank: int) -> tuple[int, ...]:
        return self.ids[self._rank_slice(rank)]


class SourceShuffleBatchSampler:
    """DataLoader batch sampler for fixed two-rank global64/local32 training.

    ``ids[position]`` identifies the instance returned by that position in the
    training dataset. IDs may be in any order, but must be unique nonnegative
    integers. Construct the same sampler on both ranks, with only ``rank``
    different, and gather rank0 followed by rank1 to recover the global plan.

    Each epoch uses ``random.Random(seed + epoch).shuffle`` once on dataset
    positions. Only complete global batches are retained, so N=9840 gives
    153 steps, 9792 distinct draws and 48 dropped IDs. Even N<64 produces no
    padded batch. Iteration is replayable and never advances ``epoch``;
    trainers must call ``set_epoch``. Smoke limits belong in the trainer, so
    the recorded sampler plan always describes the complete nominal epoch.
    """

    def __init__(self, ids, rank: int = 0, seed: int = 22,
                 global_batch: int = 64, world_size: int = 2):
        self.rank = _integer(rank, "rank")
        self.seed = _integer(seed, "seed")
        self.global_batch = _integer(global_batch, "global_batch", minimum=1)
        self.world_size = _integer(world_size, "world_size", minimum=1)
        if self.global_batch != 64 or self.world_size != 2:
            raise ValueError("this protocol requires global_batch=64 and world_size=2")
        if self.rank >= self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.ids = tuple(_integer(value, "ID") for value in ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("training IDs must be unique; duplicate IDs are not allowed")
        self.ids_sha256 = _digest(self.ids)
        self.epoch = 0
        self._cached_epoch: Optional[int] = None
        self._cached_permutation: tuple[int, ...] = ()

    @property
    def global_batch_size(self) -> int:
        return self.global_batch

    @property
    def local_batch_size(self) -> int:
        return self.global_batch // self.world_size

    @property
    def dataset_size(self) -> int:
        return len(self.ids)

    def __len__(self) -> int:
        return self.dataset_size // self.global_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _integer(epoch, "epoch")

    def epoch_permutation(self, epoch: Optional[int] = None) -> tuple[int, ...]:
        """Return all dataset positions, including the tail, after one shuffle."""
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch")
        if self._cached_epoch != use_epoch:
            permutation = list(range(self.dataset_size))
            random.Random(self.seed + use_epoch).shuffle(permutation)
            self._cached_permutation = tuple(permutation)
            self._cached_epoch = use_epoch
        return self._cached_permutation

    def global_plan(self, step: int,
                    epoch: Optional[int] = None) -> GlobalShuffleBatchPlan:
        use_step = _integer(step, "step")
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch")
        if use_step >= len(self):
            raise ValueError("step must be in [0, len(sampler))")
        start = use_step * self.global_batch
        indices = self.epoch_permutation(use_epoch)[start:start + self.global_batch]
        return GlobalShuffleBatchPlan(
            epoch=use_epoch, step=use_step, indices=indices,
            ids=tuple(self.ids[index] for index in indices),
            world_size=self.world_size)

    def iter_global_plans(self, epoch: Optional[int] = None
                          ) -> Iterator[GlobalShuffleBatchPlan]:
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch")
        for step in range(len(self)):
            yield self.global_plan(step, use_epoch)

    def __iter__(self) -> Iterator[list[int]]:
        for plan in self.iter_global_plans():
            yield list(plan.rank_indices(self.rank))

    def epoch_summary(self, epoch: Optional[int] = None,
                      include_ids: bool = True) -> dict:
        """Return JSON-ready full-epoch coverage, order and tail accounting.

        ``plans``, ``used_ids`` and ``dropped_ids`` are run artifacts and must
        remain server-local. Set ``include_ids=False`` for a shareable scalar
        summary; identity and complete-plan digests are always retained.
        The returned plan is not limited by how many trainer steps execute.
        """
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch")
        permutation = self.epoch_permutation(use_epoch)
        draws = len(self) * self.global_batch
        used_indices = permutation[:draws]
        dropped_indices = permutation[draws:]
        used_ids = tuple(self.ids[index] for index in used_indices)
        dropped_ids = tuple(self.ids[index] for index in dropped_indices)
        summary = {
            "version": SAMPLER_VERSION, "epoch": use_epoch, "seed": self.seed,
            "ids_sha256": self.ids_sha256, "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size, "steps": len(self),
            "draws": draws, "unique": len(set(used_ids)),
            "coverage": draws / self.dataset_size if self.dataset_size else 0.0,
            "dropped_count": len(dropped_ids), "padding_count": 0,
            "repeated_draws": 0, "plan_sha256": _digest(used_ids),
            "permutation_sha256": _digest(permutation),
            "dropped_ids_sha256": _digest(dropped_ids),
        }
        if include_ids:
            summary["used_ids"] = list(used_ids)
            summary["dropped_ids"] = list(dropped_ids)
            summary["plans"] = [
                {"step": plan.step, "indices": list(plan.indices), "ids": list(plan.ids)}
                for plan in self.iter_global_plans(use_epoch)]
        return summary

    def state_dict(self) -> dict:
        """Save epoch and immutable rank/configuration/ordered-ID identity."""
        return {
            "version": SAMPLER_VERSION, "epoch": self.epoch,
            "seed": self.seed, "rank": self.rank,
            "global_batch": self.global_batch, "world_size": self.world_size,
            "dataset_size": self.dataset_size, "steps": len(self),
            "ids_sha256": self.ids_sha256,
            "drop_last": True, "padding_count": 0,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Reject changed IDs/order, rank or config before restoring the epoch."""
        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("sampler state keys mismatch")
        for key, value in expected.items():
            if key != "epoch" and (type(state[key]) is not type(value) or state[key] != value):
                raise ValueError(f"sampler state mismatch: {key}")
        self.set_epoch(state["epoch"])
