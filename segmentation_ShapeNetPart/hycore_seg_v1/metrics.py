"""ShapeNetPart object-wise IoU; no global 50-class IoU substitution."""

import numpy as np

NAMES = ("airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife",
         "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table")
PART_COUNTS = (4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3)
PART_STARTS = (0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47)


def shape_ious(predictions, targets, categories):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    result = []
    for pred, target, cat in zip(predictions, targets, np.asarray(categories).reshape(-1)):
        scores = []
        start = PART_STARTS[int(cat)]
        for part in range(start, start + PART_COUNTS[int(cat)]):
            intersection = np.count_nonzero((pred == part) & (target == part))
            union = np.count_nonzero((pred == part) | (target == part))
            scores.append(intersection / union if union else 1.)
        result.append(float(np.mean(scores)))
    return np.asarray(result, dtype=np.float64)


class SegMetrics:
    def __init__(self):
        self.counts = np.zeros(16, dtype=np.int64)
        self.iou_sums = np.zeros(16, dtype=np.float64)
        self.correct = self.points = 0

    def update(self, predictions, targets, categories):
        scores = shape_ious(predictions, targets, categories)
        categories = np.asarray(categories).reshape(-1)
        np.add.at(self.counts, categories, 1)
        np.add.at(self.iou_sums, categories, scores)
        self.correct += int(np.count_nonzero(np.asarray(predictions) == targets))
        self.points += np.asarray(targets).size
        return scores

    def result(self):
        per_class = np.divide(self.iou_sums, self.counts,
                              out=np.zeros(16, dtype=np.float64), where=self.counts > 0)
        present = self.counts > 0
        return {
            "instance_miou": float(self.iou_sums.sum() / max(1, self.counts.sum())),
            "class_miou": float(per_class[present].mean()) if present.any() else None,
            "point_accuracy": self.correct / max(1, self.points),
            "objects": int(self.counts.sum()), "classes_present": int(present.sum()),
            "per_class": {name: {"count": int(n), "miou": float(score) if n else None}
                          for name, n, score in zip(NAMES, self.counts, per_class)},
        }
