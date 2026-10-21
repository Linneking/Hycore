"""Balanced global64 plans with persistent, shuffled class queues.

This sampler is deliberately stateful between epochs and read-only within an
already prepared epoch. All ranks prepare the same complete global plan; a
DataLoader worker's prefetch or a diagnostic query cannot consume queue items.
Checkpoint state restores the current epoch, allowing the next set_epoch call
to continue the exact class queues. Resuming model updates mid-epoch is outside
this sampler's contract; re-querying any current-epoch step is supported.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterator, Optional

from ..hier_proxy_scratch_v5.sampler import GlobalBatchPlan


SAMPLER_VERSION = "hier-class-cycle-v9c-1"
SCHEDULE_BLOCK_STEPS = 5


def _integer(value, name: str, minimum: int = 0) -> int:
    """Accept integer scalar types, without silently truncating a float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return converted


class ClassCycleBatchSampler:
    """Two ranks of 16 distinct classes x two different objects per class.

    Forty classes are shuffled into five groups of eight at each five-step
    block. Each group is omitted once, so every class appears four times.
    Selected class blocks are independently shuffled before their rank split.
    Every class has an independently seeded shuffled object queue. Its cursor
    and cycle number persist across epoch boundaries, and a permutation is
    exhausted before its next permutation is used. If a two-object pair
    crosses a cycle boundary, the new first two positions are swapped only
    when necessary to avoid repeating the same object inside that pair.

    Production uses 300 steps; short smoke epochs may finish an incomplete
    scheduling block. The exact partial-block size is included in the state.
    Epochs must be prepared in order: set_epoch(0), set_epoch(1), and so on.
    Repeating set_epoch(current_epoch) and querying plans are both idempotent.
    """

    def __init__(self, labels, world_size: int = 2,
                 classes_per_rank: int = 16, instances_per_class: int = 2,
                 steps: int = 300, seed: int = 4780, rank: int = 0):
        self.world_size = _integer(world_size, "world_size", 1)
        self.classes_per_rank = _integer(classes_per_rank, "classes_per_rank", 1)
        self.instances_per_class = _integer(instances_per_class, "instances_per_class", 1)
        self.steps = _integer(steps, "steps", 1)
        self.seed = _integer(seed, "seed", 0)
        self.rank = _integer(rank, "rank", 0)
        if (self.world_size, self.classes_per_rank, self.instances_per_class) != (2, 16, 2):
            raise ValueError("class-cycle protocol requires world_size=2, classes_per_rank=16, instances_per_class=2")
        if self.rank >= self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.labels = tuple(_integer(label, "label", 0) for label in labels)
        groups = defaultdict(list)
        for index, label in enumerate(self.labels):
            groups[label].append(index)
        self.by_class = {label: tuple(indices) for label, indices in groups.items()}
        self.classes = tuple(sorted(self.by_class))
        if len(self.classes) != 40:
            raise ValueError("class-cycle protocol requires exactly 40 training classes")
        if any(len(indices) < 2 for indices in self.by_class.values()):
            raise ValueError("each class needs at least two distinct objects")
        self.global_classes_per_batch = 32
        self.epoch = 0
        self._labels_sha256 = hashlib.sha256(
            ",".join(map(str, self.labels)).encode("ascii")).hexdigest()
        self._queues = {label: self._permutation(label, 0) for label in self.classes}
        self._cursors = {label: 0 for label in self.classes}
        self._cycles = {label: 0 for label in self.classes}
        self._next_epoch = 0
        self._plans: tuple[GlobalBatchPlan, ...] = ()
        self._epoch_start = None

    @property
    def global_batch_size(self) -> int:
        return 64

    @property
    def local_batch_size(self) -> int:
        return 32

    def __len__(self) -> int:
        return self.steps

    def _rng(self, *parts) -> random.Random:
        key = ":".join(map(str, (SAMPLER_VERSION, self.seed, *parts))).encode("ascii")
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))

    def _permutation(self, label: int, cycle: int) -> list[int]:
        values = list(self.by_class[label])
        self._rng("class-queue", label, cycle).shuffle(values)
        return values

    def _draw_one(self, label: int, forbidden: Optional[int] = None) -> int:
        if self._cursors[label] == len(self._queues[label]):
            self._cycles[label] += 1
            self._queues[label] = self._permutation(label, self._cycles[label])
            self._cursors[label] = 0
            if self._queues[label][0] == forbidden:
                self._queues[label][0], self._queues[label][1] = self._queues[label][1], self._queues[label][0]
        cursor = self._cursors[label]
        index = self._queues[label][cursor]
        self._cursors[label] += 1
        if index == forbidden:
            raise RuntimeError("class queue produced a duplicate within one pair")
        return index

    def _queue_state(self) -> dict:
        return {str(label): {"permutation": list(self._queues[label]),
                             "cursor": self._cursors[label], "cycle": self._cycles[label]}
                for label in self.classes}

    def _restore_queues(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {str(label) for label in self.classes}:
            raise ValueError("sampler queue class keys mismatch")
        queues, cursors, cycles = {}, {}, {}
        for label in self.classes:
            row = state[str(label)]
            if not isinstance(row, dict) or set(row) != {"permutation", "cursor", "cycle"}:
                raise ValueError("sampler queue state fields mismatch")
            values = [_integer(value, "queue index", 0) for value in row["permutation"]]
            if len(values) != len(self.by_class[label]) or set(values) != set(self.by_class[label]):
                raise ValueError("sampler queue is not a permutation of its training class")
            cursor = _integer(row["cursor"], "queue cursor", 0)
            if cursor > len(values):
                raise ValueError("sampler queue cursor exceeds permutation")
            queues[label], cursors[label] = values, cursor
            cycles[label] = _integer(row["cycle"], "queue cycle", 0)
        self._queues, self._cursors, self._cycles = queues, cursors, cycles

    def _prepare_epoch(self, epoch: int) -> None:
        if epoch != self._next_epoch:
            raise ValueError("epochs must be prepared sequentially; restore a checkpoint before resuming")
        self._epoch_start = self._queue_state()
        plans = []
        for block in range((self.steps + SCHEDULE_BLOCK_STEPS - 1) // SCHEDULE_BLOCK_STEPS):
            class_order = list(self.classes)
            self._rng("class-block", epoch, block).shuffle(class_order)
            for offset in range(SCHEDULE_BLOCK_STEPS):
                step = block * SCHEDULE_BLOCK_STEPS + offset
                if step >= self.steps:
                    break
                omitted_start = offset * 8
                selected = class_order[:omitted_start] + class_order[omitted_start + 8:]
                self._rng("rank-class-order", epoch, step).shuffle(selected)
                indices, labels = [], []
                for label in selected:
                    first = self._draw_one(label)
                    second = self._draw_one(label, forbidden=first)
                    indices.extend((first, second))
                    labels.extend((label, label))
                plans.append(GlobalBatchPlan(
                    epoch=epoch, step=step, classes=tuple(selected),
                    indices=tuple(indices), labels=tuple(labels),
                    world_size=self.world_size, classes_per_rank=self.classes_per_rank,
                    instances_per_class=self.instances_per_class))
        self._plans = tuple(plans)
        self.epoch, self._next_epoch = epoch, epoch + 1

    def set_epoch(self, epoch: int) -> None:
        use_epoch = _integer(epoch, "epoch", 0)
        if self._plans and use_epoch == self.epoch:
            return
        self._prepare_epoch(use_epoch)

    def _ensure_epoch(self) -> None:
        if not self._plans:
            self._prepare_epoch(self.epoch)

    def global_plan(self, step: int, epoch: Optional[int] = None) -> GlobalBatchPlan:
        use_step = _integer(step, "step", 0)
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch", 0)
        if use_step >= self.steps:
            raise ValueError("step must be in [0, steps)")
        if use_epoch != self.epoch:
            raise ValueError("global_plan only queries the prepared current epoch; call set_epoch first")
        self._ensure_epoch()
        return self._plans[use_step]

    def iter_global_plans(self, epoch: Optional[int] = None) -> Iterator[GlobalBatchPlan]:
        use_epoch = self.epoch if epoch is None else _integer(epoch, "epoch", 0)
        if use_epoch != self.epoch:
            raise ValueError("iter_global_plans only queries the current epoch; call set_epoch first")
        self._ensure_epoch()
        yield from self._plans

    def __iter__(self) -> Iterator[list[int]]:
        for plan in self.iter_global_plans():
            yield list(plan.rank_indices(self.rank))

    def _identity(self) -> dict:
        return {"version": SAMPLER_VERSION, "seed": self.seed, "steps": self.steps,
                "world_size": self.world_size, "classes_per_rank": self.classes_per_rank,
                "instances_per_class": self.instances_per_class,
                "labels_sha256": self._labels_sha256,
                "schedule_block_steps": SCHEDULE_BLOCK_STEPS,
                "complete_schedule_blocks_per_epoch": self.steps // SCHEDULE_BLOCK_STEPS,
                "partial_schedule_block_steps": self.steps % SCHEDULE_BLOCK_STEPS}

    def state_dict(self) -> dict:
        self._ensure_epoch()
        return {**self._identity(), "epoch": self.epoch, "next_epoch": self._next_epoch,
                "epoch_start_queues": {label: {**row, "permutation": list(row["permutation"])}
                                       for label, row in self._epoch_start.items()},
                "next_epoch_queues": self._queue_state()}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("sampler state must be a dictionary")
        for key, value in self._identity().items():
            if state.get(key) != value:
                raise ValueError(f"sampler state mismatch: {key}")
        epoch = _integer(state.get("epoch"), "saved epoch", 0)
        if state.get("next_epoch") != epoch + 1:
            raise ValueError("sampler next_epoch mismatch")
        original_queues, original_epoch, original_next = self._queue_state(), self.epoch, self._next_epoch
        original_plans, original_start = self._plans, self._epoch_start
        try:
            self._restore_queues(state.get("epoch_start_queues"))
            self._plans = ()
            self._next_epoch = epoch
            self._prepare_epoch(epoch)
            if self._queue_state() != state.get("next_epoch_queues"):
                raise ValueError("sampler next-epoch queue state does not match replayed current epoch")
        except (ValueError, TypeError, KeyError):
            self._restore_queues(original_queues)
            self.epoch, self._next_epoch = original_epoch, original_next
            self._plans, self._epoch_start = original_plans, original_start
            raise


# Explicit aliases ease migration of versioned runners without changing legacy code.
CycleClassBatchSampler = ClassCycleBatchSampler
SourceClassBatchSampler = ClassCycleBatchSampler
