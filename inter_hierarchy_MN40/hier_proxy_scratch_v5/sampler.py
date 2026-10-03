"""Released-HIER class sampling adapted to a global 64-instance batch.

Each step uniformly samples distinct classes, splits class blocks between
ranks, and samples instances within each class with replacement. The exact
random sequence is versioned here; the rules, rather than NumPy's upstream
random stream, are reproduced. No torch import is needed for CPU audits.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional


SAMPLER_VERSION = "hier-source-class-batch-v5-1"


@dataclass(frozen=True)
class GlobalBatchPlan:
    """One global plan; class blocks are contiguous and rank-disjoint."""

    epoch: int
    step: int
    classes: tuple[int, ...]
    indices: tuple[int, ...]
    labels: tuple[int, ...]
    world_size: int
    classes_per_rank: int
    instances_per_class: int

    @property
    def local_batch_size(self) -> int:
        return self.classes_per_rank * self.instances_per_class

    def _rank_slice(self, rank: int) -> slice:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        start = rank * self.local_batch_size
        return slice(start, start + self.local_batch_size)

    def rank_indices(self, rank: int) -> tuple[int, ...]:
        return self.indices[self._rank_slice(rank)]

    def rank_labels(self, rank: int) -> tuple[int, ...]:
        return self.labels[self._rank_slice(rank)]

    def rank_classes(self, rank: int) -> tuple[int, ...]:
        self._rank_slice(rank)
        start = rank * self.classes_per_rank
        return self.classes[start:start + self.classes_per_rank]


class SourceClassBatchSampler:
    """DataLoader-compatible local batch sampler sharing a global plan.

    Construct the same sampler on each rank with only ``rank`` differing.
    ``labels`` are in the local training-subset index order. Each yielded
    index addresses that subset, not an unsplit parent dataset. A repeated
    index within a two-instance class block is intentional source behavior.

    A step has its own seeded RNG, so querying plans out of order or replaying
    an interrupted epoch cannot change subsequent plans. Iteration itself
    does not advance the epoch; call ``set_epoch`` explicitly.
    """

    def __init__(self, labels, world_size: int = 2,
                 classes_per_rank: int = 16, instances_per_class: int = 2,
                 steps: int = 200, seed: int = 22, rank: int = 0):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if classes_per_rank < 2 or classes_per_rank % 2:
            raise ValueError("classes_per_rank must be even and >=2 for flip-safe blocks")
        if instances_per_class < 1 or steps < 1:
            raise ValueError("instances_per_class and steps must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.labels = tuple(int(label) for label in labels)
        if not self.labels:
            raise ValueError("labels cannot be empty")
        groups = defaultdict(list)
        for index, label in enumerate(self.labels):
            groups[label].append(index)
        self.by_class = {label: tuple(indices) for label, indices in groups.items()}
        self.classes = tuple(sorted(self.by_class))
        self.world_size = int(world_size)
        self.classes_per_rank = int(classes_per_rank)
        self.instances_per_class = int(instances_per_class)
        self.steps = int(steps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.epoch = 0
        self.global_classes_per_batch = self.world_size * self.classes_per_rank
        if self.global_classes_per_batch > len(self.classes):
            raise ValueError("global classes per batch exceed available training classes")

    @property
    def global_batch_size(self) -> int:
        return self.global_classes_per_batch * self.instances_per_class

    @property
    def local_batch_size(self) -> int:
        return self.classes_per_rank * self.instances_per_class

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def _step_rng(self, epoch: int, step: int) -> random.Random:
        key = f"{SAMPLER_VERSION}:{self.seed}:{epoch}:{step}".encode("ascii")
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))

    def global_plan(self, step: int, epoch: Optional[int] = None) -> GlobalBatchPlan:
        use_epoch = self.epoch if epoch is None else int(epoch)
        if use_epoch < 0 or not 0 <= step < self.steps:
            raise ValueError("nonnegative epoch and step in [0, steps) required")
        rng = self._step_rng(use_epoch, int(step))
        selected = tuple(rng.sample(self.classes, self.global_classes_per_batch))
        indices = []
        labels = []
        for label in selected:
            pool = self.by_class[label]
            for _ in range(self.instances_per_class):
                indices.append(rng.choice(pool))
                labels.append(label)
        return GlobalBatchPlan(
            epoch=use_epoch, step=int(step), classes=selected,
            indices=tuple(indices), labels=tuple(labels),
            world_size=self.world_size, classes_per_rank=self.classes_per_rank,
            instances_per_class=self.instances_per_class)

    def iter_global_plans(self, epoch: Optional[int] = None) -> Iterator[GlobalBatchPlan]:
        for step in range(self.steps):
            yield self.global_plan(step, epoch)

    def __iter__(self) -> Iterator[list[int]]:
        for plan in self.iter_global_plans():
            yield list(plan.rank_indices(self.rank))

    def state_dict(self) -> dict:
        return {
            "version": SAMPLER_VERSION, "epoch": self.epoch, "seed": self.seed,
            "steps": self.steps, "world_size": self.world_size,
            "classes_per_rank": self.classes_per_rank,
            "instances_per_class": self.instances_per_class,
            "labels_sha256": hashlib.sha256(
                ",".join(map(str, self.labels)).encode("ascii")).hexdigest(),
        }

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        for key in expected:
            if key != "epoch" and state.get(key) != expected[key]:
                raise ValueError(f"sampler state mismatch: {key}")
        self.set_epoch(int(state["epoch"]))
