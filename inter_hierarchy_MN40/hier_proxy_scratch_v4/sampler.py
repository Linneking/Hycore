"""Epoch-covering class-block batches, including minimal fixed-shape padding.

Every dataset instance appears at least once in a complete epoch. Instances
are shuffled once in each class, divided into size-S chunks, then classes
are scheduled at most once per batch. The number of batches is the smallest
possible with these chunks and C distinct classes in every batch. Padding
is explicit and reported, and class blocks make HyCoRe flip negatives safe
when C is even. No old sampler or original reproduction path is changed.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class EpochCoveringClassBatchSampler(Sampler[list[int]]):
    def __init__(self, labels, classes_per_batch: int = 4,
                 samples_per_class: int = 8, seed: int = 22):
        targets = np.asarray(labels).reshape(-1)
        if classes_per_batch < 2 or classes_per_batch % 2:
            raise ValueError("classes_per_batch must be positive and even for flip-safe blocks")
        if samples_per_class < 1 or not len(targets):
            raise ValueError("positive samples_per_class and nonempty labels required")
        self.by_class = defaultdict(list)
        for index, target in enumerate(targets.tolist()):
            self.by_class[int(target)].append(index)
        self.classes = sorted(self.by_class)
        if len(self.classes) < classes_per_batch:
            raise ValueError("classes_per_batch exceeds the number of classes")
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.dataset_size = len(targets)
        self.epoch = 0
        self.chunk_counts = {
            cls: math.ceil(len(indices) / self.samples_per_class)
            for cls, indices in self.by_class.items()}
        self.batches_per_epoch = max(max(self.chunk_counts.values()), math.ceil(
            sum(self.chunk_counts.values()) / self.classes_per_batch))
        self._batches = None
        self.stats = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None
        self.stats = {}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _plan(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        counts = dict(self.chunk_counts)
        # Complete the class token total to C*M. A class can occur at most
        # once in each of M batches, so token counts never exceed M.
        missing = self.classes_per_batch * len(self) - sum(counts.values())
        for _ in range(missing):
            candidates = [cls for cls in self.classes if counts[cls] < len(self)]
            if not candidates:
                raise RuntimeError("unable to pad a distinct-class epoch")
            cls = int(rng.choice(candidates))
            counts[cls] += 1
        chunks = {}
        for cls in self.classes:
            pool = np.asarray(self.by_class[cls], dtype=np.int64)
            shuffled = rng.permutation(pool).tolist()
            chunks[cls] = []
            for start in range(0, len(shuffled), self.samples_per_class):
                chunk = shuffled[start:start + self.samples_per_class]
                if len(chunk) < self.samples_per_class:
                    # Do not duplicate a point inside a class block when the
                    # class itself has enough unique instances to avoid it.
                    needed = self.samples_per_class - len(chunk)
                    candidates = np.asarray([index for index in pool if index not in set(chunk)])
                    if len(candidates) >= needed:
                        chunk.extend(rng.choice(candidates, needed, replace=False).tolist())
                    else:
                        chunk.extend(rng.choice(pool, needed, replace=True).tolist())
                chunks[cls].append(chunk)
            while len(chunks[cls]) < counts[cls]:
                chunks[cls].append(rng.choice(
                    pool, self.samples_per_class,
                    replace=len(pool) < self.samples_per_class).tolist())
        remaining = dict(counts)
        next_chunk = {cls: 0 for cls in self.classes}
        batches = []
        for _ in range(len(self)):
            # Largest residual degree first is a valid bipartite schedule:
            # before each step sum= C*m and maximum residual <= m. Removing
            # one token from each of the C largest preserves this invariant.
            tie_order = rng.permutation(self.classes).tolist()
            chosen = sorted(tie_order, key=lambda cls: -remaining[cls])[:self.classes_per_batch]
            if any(remaining[cls] <= 0 for cls in chosen):
                raise RuntimeError("class schedule exhausted before epoch completion")
            rng.shuffle(chosen)
            batch = []
            for cls in chosen:
                batch.extend(chunks[cls][next_chunk[cls]])
                next_chunk[cls] += 1
                remaining[cls] -= 1
            batches.append(batch)
        draws = len(batches) * self.classes_per_batch * self.samples_per_class
        unique = len({index for batch in batches for index in batch})
        if unique != self.dataset_size or any(remaining.values()):
            raise RuntimeError("epoch coverage invariant violated")
        self._batches = batches
        self.stats = {
            "epoch": self.epoch,
            "batches": len(batches),
            "batch_size": self.classes_per_batch * self.samples_per_class,
            "dataset_size": self.dataset_size,
            "planned_draws": draws,
            "planned_unique_instances": unique,
            "planned_instance_coverage": unique / self.dataset_size,
            "planned_repeated_draws": draws - unique,
            "planned_repeat_fraction": (draws - unique) / draws,
            "class_chunks_before_batch_padding": sum(self.chunk_counts.values()),
            "extra_class_chunks": missing,
            "flip_safe_class_blocks": True,
        }

    def __iter__(self):
        if self._batches is None:
            self._plan()
        for batch in self._batches:
            yield list(batch)
