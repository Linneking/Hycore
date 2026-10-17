"""Explicit sample domains and checkpoint selection for full9840/testsel."""
from __future__ import annotations
import math
import numpy as np


def full_training_ids(labels):
    labels = np.asarray(labels).reshape(-1)
    if labels.size != 9840 or not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("Expected official ModelNet40 train9840 covering labels0..39")
    return np.arange(labels.size, dtype=np.int64)


def select_best(previous, metrics, prefix, epoch):
    """Test is scoring only. Ties keep the earlier model; no CE tiebreak."""
    oa, aa, ce = (float(metrics[prefix + "_" + k]) for k in ("oa", "aa", "ce"))
    if not all(math.isfinite(v) for v in (oa, aa, ce)):
        raise ValueError("Nonfinite model-selection metrics")
    count = int(metrics[prefix + "_count"])
    if prefix == "test":
        if count != 2468 or metrics.get("test_smoke_partial", False):
            raise ValueError("Test selection requires all2468 official objects")
    elif prefix != "smoke_train":
        raise ValueError("Selection must explicitly name official test or smoke training")
    if previous is not None and oa <= previous[prefix + "_oa"]:
        return previous
    return {"epoch": int(epoch), "selected_by": prefix + " OA strictly higher; first tie retained",
            prefix + "_oa": oa, prefix + "_aa": aa, prefix + "_ce": ce}
