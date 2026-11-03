"""Unchanged IoU/evaluation plus observational GT-absence error counts."""
import numpy as np
import torch
import torch.nn.functional as F
from ..hycore_seg_v9.metrics import SegMetrics as _ReferenceMetrics
from ..hycore_seg_v1.metrics import PART_STARTS, PART_COUNTS
from ..hycore_seg_v1.train import bn_counts
from ..hycore_seg_v1.utils import write_json


class SegMetrics(_ReferenceMetrics):
    def __init__(self):
        super().__init__()
        self.absent = np.zeros(50, dtype=np.int64)
        self.absent_false_positive = np.zeros(50, dtype=np.int64)
        self.present_iou_sum = np.zeros(50, dtype=np.float64)

    def update(self, predictions, targets, categories):
        result = super().update(predictions, targets, categories)
        for pred, target, category in zip(predictions, targets, categories):
            gt = np.bincount(target, minlength=50)
            pr = np.bincount(pred, minlength=50)
            tp = np.bincount(target[pred == target], minlength=50)
            start = PART_STARTS[int(category)]
            for k in range(start, start + PART_COUNTS[int(category)]):
                if gt[k] == 0:
                    self.absent[k] += 1
                    self.absent_false_positive[k] += int(pr[k] > 0)
                else:
                    self.present_iou_sum[k] += tp[k] / (gt[k] + pr[k] - tp[k])
        return result

    def result(self):
        result = super().result()
        for key, part in result['parts'].items():
            k = int(key)
            present = part['present_objects']
            objects = present + int(self.absent[k])
            part.update(absent_objects=int(self.absent[k]),
                        false_positive_absent_objects=int(self.absent_false_positive[k]),
                        mean_iou_gt_present=float(self.present_iou_sum[k] / present) if present else None,
                        mean_iou_all_objects_absent_both_is_one=float(
                            (self.present_iou_sum[k] + self.absent[k] - self.absent_false_positive[k]) / objects)
                        if objects else None)
        return result


@torch.no_grad()
def evaluate(model, data_loader, device, per_object_path=None):
    model.eval()
    before = bn_counts(model)
    raw, masked = SegMetrics(), SegMetrics()
    ce_sum, objects = 0., 0
    records = []
    for points, categories, targets, ids in data_loader:
        points, categories, targets = points.to(device), categories.to(device), targets.to(device)
        one_hot = F.one_hot(categories, 16).float()
        logits = model(points.transpose(1, 2).contiguous(), one_hot).transpose(1, 2).contiguous()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        ce = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
        predictions = logits.argmax(-1)
        masked_logits = logits.clone()
        for row, category in enumerate(categories.tolist()):
            start, end = PART_STARTS[category], PART_STARTS[category] + PART_COUNTS[category]
            masked_logits[row, :, :start] = -torch.inf
            masked_logits[row, :, end:] = -torch.inf
        pred_np, target_np, cat_np = predictions.cpu().numpy(), targets.cpu().numpy(), categories.cpu().numpy()
        ious = raw.update(pred_np, target_np, cat_np)
        masked.update(masked_logits.argmax(-1).cpu().numpy(), target_np, cat_np)
        ce_sum += float(ce) * len(categories)
        objects += len(categories)
        if per_object_path:
            records.extend({"id": sample_id, "category": int(cat), "shape_iou": float(iou)}
                           for sample_id, cat, iou in zip(ids, cat_np, ious))
    if bn_counts(model) != before:
        raise AssertionError("Evaluation changed BN state")
    result = raw.result()
    result.update(ce=ce_sum / max(1, objects), masked_secondary=masked.result())
    if per_object_path:
        write_json(per_object_path, records)
    return result
