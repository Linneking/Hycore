import numpy as np
from ..hycore_seg_v1.metrics import SegMetrics as BaseMetrics, PART_STARTS, PART_COUNTS
from ..hycore_seg_v1.train import bn_counts
from ..hycore_seg_v1.utils import append_json, write_json
import torch
import torch.nn.functional as F

class SegMetrics(BaseMetrics):
    def __init__(self):
        super().__init__();self.confusion=np.zeros((50,50),dtype=np.int64);self.missing=np.zeros(50,dtype=np.int64);self.present=np.zeros(50,dtype=np.int64)
    def update(self,predictions,targets,categories):
        p,y=np.asarray(predictions),np.asarray(targets)
        self.confusion+=np.bincount((y*50+p).reshape(-1),minlength=2500).reshape(50,50)
        for a,b in zip(p,y):
            for k in np.unique(b):
                self.present[k]+=1;self.missing[k]+=int(not np.any(a==k))
        return super().update(p,y,categories)
    def result(self):
        out=super().result();g=self.confusion.sum(1);p=self.confusion.sum(0);t=self.confusion.diagonal()
        out['parts']={str(k):dict(gt=int(g[k]),pred=int(p[k]),tp=int(t[k]),recall=float(t[k]/g[k]) if g[k] else None,precision=float(t[k]/p[k]) if p[k] else None,missing_objects=int(self.missing[k]),present_objects=int(self.present[k])) for k in range(50)}
        return out

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
