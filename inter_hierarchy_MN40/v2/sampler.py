"""Class-balanced batch sampling for reliable within-class triplets."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels,
        classes_per_batch: int = 5,
        samples_per_class: int = 8,
        seed: int = 0,
        batches_per_epoch: int | None = None,
    ):
        labels = np.asarray(labels).reshape(-1)
        self.by_class = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            self.by_class[int(label)].append(index)
        self.classes = np.asarray(sorted(self.by_class))
        if classes_per_batch > len(self.classes):
            raise ValueError("classes_per_batch exceeds number of classes")
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        batch_size = classes_per_batch * samples_per_class
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(labels) / batch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            chosen = rng.choice(self.classes, self.classes_per_batch, replace=False)
            batch = []
            for cls in chosen:
                pool = np.asarray(self.by_class[int(cls)])
                replace = len(pool) < self.samples_per_class
                batch.extend(rng.choice(pool, self.samples_per_class, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch
