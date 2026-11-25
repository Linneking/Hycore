"""Class-balanced sampling for the 15-class ScanObjectNN experiments."""

from collections.abc import Iterator
from operator import index

import numpy as np
from torch.utils.data import Sampler


class BalancedBatches(Sampler[list[int]]):
    """Draw 32 rows per step and give two classes a rotating third slot."""

    def __init__(self, labels, seed: int, steps: int = 356):
        self.seed = index(seed)
        self.steps = index(steps)
        if self.seed < 0 or self.steps < 1:
            raise ValueError("Expected a nonnegative seed and a positive step count.")

        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Expected one integer class label per training row.")

        classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(classes) != 15 or np.any(counts < 3):
            raise ValueError("Expected 15 classes with at least three rows each.")
        self._rows_by_class = tuple(np.flatnonzero(inverse == i) for i in range(len(classes)))
        rotation_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0]))
        self._extra_class_order = rotation_rng.permutation(len(classes)).tolist()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = index(epoch)
        if epoch < 0:
            raise ValueError("Expected a nonnegative epoch number.")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1, epoch]))
        class_count = len(self._rows_by_class)
        # Continue the rotation across epochs so a class is not favored at each boundary.
        offset = 2 * self.steps * epoch
        for step in range(self.steps):
            first_slot = (offset + 2 * step) % class_count
            extra_classes = (
                self._extra_class_order[first_slot],
                self._extra_class_order[(first_slot + 1) % class_count],
            )
            batch = []
            for class_index, rows in enumerate(self._rows_by_class):
                count = 3 if class_index in extra_classes else 2
                batch.extend(rng.choice(rows, count, replace=False).tolist())
            rng.shuffle(batch)
            yield batch
