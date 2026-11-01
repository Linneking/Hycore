"""Read-only checkpoint diagnosis for DGCNN / H0 / H part segmentation.

The checkpoint and HDF5 inputs are never modified. No optimizer, backward,
training forward, BN recalibration, or checkpoint selection occurs here.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NAMES = ("airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife",
         "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table")
STARTS = (0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47)
COUNTS = (4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3)
OWNER = np.repeat(np.arange(16), COUNTS)
FOCUS = ("bag", "rocket", "motorbike", "skateboard", "earphone", "cap", "mug")
SCORE_NAMES = ("correct_probability", "global_margin", "legal_part_margin",
               "correct_rank", "legal_correct_rank")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class PartStatistics:
    """Pooled counts and object-conditioned measures, with explicit denominators."""

    def __init__(self):
        self.confusion = np.zeros((50, 50), dtype=np.int64)
        self.class_confusion = np.zeros((16, 50, 50), dtype=np.int64)
        self.objects = np.zeros(16, dtype=np.int64)
        self.single_prediction = np.zeros(16, dtype=np.int64)
        self.constant_labels = np.zeros((16, 50), dtype=np.int64)
        self.illegal_points = np.zeros(16, dtype=np.int64)
        self.total_points = np.zeros(16, dtype=np.int64)
        self.gt_present = np.zeros(50, dtype=np.int64)
        self.pred_present = np.zeros(50, dtype=np.int64)
        self.missing = np.zeros(50, dtype=np.int64)
        self.zero_recall = np.zeros(50, dtype=np.int64)
        self.object_iou_sum = np.zeros(50, dtype=np.float64)
        self.present_iou_sum = np.zeros(50, dtype=np.float64)
        self.score_sums = {key: np.zeros(50, dtype=np.float64) for key in SCORE_NAMES}
        self.score_mins = {key: np.full(50, np.inf) for key in SCORE_NAMES}
        self.score_maxs = {key: np.full(50, -np.inf) for key in SCORE_NAMES}
        self.rank_hist = np.zeros((50, 50), dtype=np.int64)

    def update(self, pred, target, categories, indices, ids, scores):
        records = []
        if pred.shape != target.shape or pred.ndim != 2:
            raise ValueError("Expected matching [B,N] labels")
        if np.any((target < 0) | (target >= 50)) or np.any((pred < 0) | (pred >= 50)):
            raise ValueError("Part labels out of range")
        for b, cat in enumerate(categories):
            cat = int(cat)
            p, y = pred[b], target[b]
            if np.any(OWNER[y] != cat):
                raise ValueError("Ground-truth part does not belong to object category")
            conf = np.bincount(y * 50 + p, minlength=2500).reshape(50, 50)
            self.confusion += conf
            self.class_confusion[cat] += conf
            gt = np.bincount(y, minlength=50)
            pr = np.bincount(p, minlength=50)
            tp = np.diag(conf)
            self.objects[cat] += 1
            self.total_points[cat] += y.size
            self.illegal_points[cat] += int(np.count_nonzero(OWNER[p] != cat))
            unique = np.flatnonzero(pr)
            if len(unique) == 1:
                self.single_prediction[cat] += 1
                self.constant_labels[cat, unique[0]] += 1
            self.pred_present += (pr > 0)
            part_rows = []
            ious = []
            for k in range(STARTS[cat], STARTS[cat] + COUNTS[cat]):
                union = gt[k] + pr[k] - tp[k]
                iou = float(tp[k] / union) if union else 1.0
                ious.append(iou)
                self.object_iou_sum[k] += iou
                if gt[k]:
                    self.gt_present[k] += 1
                    self.present_iou_sum[k] += iou
                    self.missing[k] += int(pr[k] == 0)
                    self.zero_recall[k] += int(tp[k] == 0)
                part_rows.append({"part": k, "gt": int(gt[k]), "pred": int(pr[k]),
                                  "tp": int(tp[k]), "iou": iou,
                                  "recall": ratio(tp[k], gt[k]),
                                  "precision": ratio(tp[k], pr[k])})
            for k in np.flatnonzero(gt):
                mask = y == k
                for key in SCORE_NAMES:
                    values = np.asarray(scores[key][b][mask], dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite classification diagnostics")
                    self.score_sums[key][k] += values.sum()
                    self.score_mins[key][k] = min(self.score_mins[key][k], float(values.min()))
                    self.score_maxs[key][k] = max(self.score_maxs[key][k], float(values.max()))
                self.rank_hist[k] += np.bincount(
                    scores["correct_rank"][b][mask].astype(np.int64) - 1, minlength=50)
            records.append({"index": int(indices[b]), "id": str(ids[b]), "category": NAMES[cat],
                            "category_index": cat, "shape_miou": float(np.mean(ious)),
                            "prediction_unique_count": len(unique),
                            "prediction_labels": unique.tolist(),
                            "largest_prediction_fraction": float(pr.max() / y.size),
                            "illegal_prediction_points": int(np.count_nonzero(OWNER[p] != cat)),
                            "gt_counts": {str(k): int(gt[k]) for k in np.flatnonzero(gt)},
                            "pred_counts": {str(k): int(pr[k]) for k in unique},
                            "parts": part_rows})
        return records

    def result(self):
        gt = self.confusion.sum(axis=1)
        pr = self.confusion.sum(axis=0)
        tp = np.diag(self.confusion)
        parts = {}
        for k in range(50):
            cat = int(OWNER[k])
            item = {"part": k, "category": NAMES[cat], "gt_points": int(gt[k]),
                    "pred_points_all_categories": int(pr[k]), "true_positive_points": int(tp[k]),
                    "pooled_iou": ratio(tp[k], gt[k] + pr[k] - tp[k]),
                    "recall": ratio(tp[k], gt[k]), "precision": ratio(tp[k], pr[k]),
                    "owner_category_objects": int(self.objects[cat]),
                    "gt_present_objects": int(self.gt_present[k]),
                    "pred_present_objects_all_categories": int(self.pred_present[k]),
                    "gt_present_but_no_prediction_objects": int(self.missing[k]),
                    "missing_fraction_given_gt_present": ratio(self.missing[k], self.gt_present[k]),
                    "gt_present_but_zero_recall_objects": int(self.zero_recall[k]),
                    "object_mean_iou_owner_category": ratio(self.object_iou_sum[k], self.objects[cat]),
                    "object_mean_iou_gt_present_only": ratio(self.present_iou_sum[k], self.gt_present[k]),
                    "correct_rank_histogram_1_to_50": self.rank_hist[k].tolist()}
            for key in SCORE_NAMES:
                item[key] = {"mean": ratio(self.score_sums[key][k], gt[k]),
                             "min": float(self.score_mins[key][k]) if gt[k] else None,
                             "max": float(self.score_maxs[key][k]) if gt[k] else None}
            parts[str(k)] = item
        categories = {}
        for cat, name in enumerate(NAMES):
            categories[name] = {"objects": int(self.objects[cat]),
                                "single_predicted_label_objects": int(self.single_prediction[cat]),
                                "single_predicted_label_fraction": ratio(self.single_prediction[cat], self.objects[cat]),
                                "constant_prediction_label_counts": {
                                    str(k): int(n) for k, n in enumerate(self.constant_labels[cat]) if n},
                                "illegal_prediction_points": int(self.illegal_points[cat]),
                                "illegal_prediction_fraction": ratio(self.illegal_points[cat], self.total_points[cat]),
                                "confusion_50x50": self.class_confusion[cat].tolist()}
        return {"parts": parts, "categories": categories,
                "confusion_50x50": self.confusion.tolist(),
                "pooled_precision_counts_predictions_in_wrong_object_categories": True,
                "object_iou_absent_both_rule": 1.0,
                "undefined_pooled_ratios": None,
                "rank_definition": "1 + number of logits strictly greater than correct; ties can share rank1"}


def metric_differences(actual, expected):
    if not expected:
        return None
    result = {key: actual[key] - expected[key]
              for key in ("instance_miou", "class_miou", "point_accuracy")
              if expected.get(key) is not None and actual.get(key) is not None}
    result["per_class"] = {key: actual["per_class"][key]["miou"] - value["miou"]
                           for key, value in expected.get("per_class", {}).items()
                           if value.get("miou") is not None and actual["per_class"][key]["miou"] is not None}
    values = [v for k, v in result.items() if k != "per_class"] + list(result["per_class"].values())
    result["max_absolute_difference"] = max(map(abs, values), default=0.)
    result["within_1e_minus5"] = result["max_absolute_difference"] <= 1e-5
    result["objects_match"] = expected.get("objects", actual["objects"]) == actual["objects"]
    return result


def evaluate_split(model, data_path, split, out, batch_size, runtime, expected=None):
    torch, h5py, SegMetrics = runtime["torch"], runtime["h5py"], runtime["SegMetrics"]
    before = runtime["state_hash"](model.state_dict())
    started = time.monotonic()
    meter, stats = SegMetrics(), PartStatistics()
    prediction_batches = []
    with h5py.File(data_path, "r") as data, (out / "objects.jsonl").open("x", encoding="utf-8") as objects:
        categories = data["label"][:].reshape(-1).astype(np.int64)
        selected = (np.arange(len(categories)) if split == "test" else
                    np.flatnonzero(np.isin(categories, [NAMES.index(name) for name in FOCUS])))
        if data["data"].shape[1] < 2048 or data["pid"].shape[1] < 2048:
            raise ValueError("Fewer than 2048 source points")
        ids = data["ids"].asstr()[:]
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(selected), batch_size):
                index = selected[offset:offset + batch_size]
                points = torch.from_numpy(data["data"][index, :2048].astype(np.float32)).cuda()
                target = data["pid"][index, :2048].astype(np.int64)
                cats = categories[index]
                category_tensor = torch.from_numpy(cats).cuda()
                onehot = torch.nn.functional.one_hot(category_tensor, 16).float()
                logits = model(points.transpose(1, 2).contiguous(), onehot).transpose(1, 2).contiguous()
                if logits.shape != (len(index), 2048, 50) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Invalid classification output")
                y = torch.from_numpy(target).cuda()
                true = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                prob = logits.softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
                rank = (logits > true.unsqueeze(-1)).sum(-1) + 1
                wrong = logits.clone().scatter_(-1, y.unsqueeze(-1), float("-inf"))
                margin = true - wrong.max(-1).values
                allowed = torch.as_tensor(OWNER, device=logits.device)[None, :] == category_tensor[:, None]
                legal_wrong = wrong.masked_fill(~allowed[:, None, :], float("-inf"))
                legal_margin = true - legal_wrong.max(-1).values
                legal_rank = ((logits > true.unsqueeze(-1)) & allowed[:, None, :]).sum(-1) + 1
                pred = logits.argmax(-1).cpu().numpy()
                score = dict(zip(SCORE_NAMES, [x.cpu().numpy() for x in
                                 (prob, margin, legal_margin, rank, legal_rank)]))
                meter.update(pred, target, cats)
                records = stats.update(pred, target, cats, index, ids[index], score)
                for record in records:
                    objects.write(json.dumps(record, allow_nan=False) + "\n")
                prediction_batches.append(pred.astype(np.uint8))
                if offset == 0 or offset // batch_size % 25 == 0:
                    write_json(out / "status.json", {"state": "inference", "split": split,
                               "objects_done": min(offset + batch_size, len(selected)),
                               "objects_total": len(selected), "at": now()})
        actual = meter.result()
        if actual["objects"] != len(selected):
            raise AssertionError("Evaluation count mismatch")
        np.savez_compressed(out / "predictions.npz", indices=selected, categories=categories[selected],
                            predictions=np.concatenate(prediction_batches), point_count=np.asarray(2048))
    after = runtime["state_hash"](model.state_dict())
    if before != after:
        raise AssertionError("Model parameters or BN buffers changed during diagnosis")
    result = {"split": split, "metrics": actual, **stats.result(),
              "expected_metrics_difference": metric_differences(actual, expected),
              "state_before_sha256": before, "state_after_sha256": after,
              "state_unchanged": True, "elapsed_seconds": time.monotonic() - started,
              "point_order": "first2048_unpermuted", "model_mode": "eval",
              "grad_mode": "inference_mode", "prediction_rule": "raw_50_class_argmax",
              "train_focus_categories": list(FOCUS) if split != "test" else None}
    write_json(out / "summary.json", result)
    write_json(out / "status.json", {"state": "complete", "objects": actual["objects"], "at": now()})
    return {"metrics": actual, "state_unchanged": True,
            "expected_metrics_difference": result["expected_metrics_difference"],
            "elapsed_seconds": result["elapsed_seconds"]}


def run(args):
    source = Path(args.source).resolve(strict=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"state": "preflight", "at": now()})
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        sys.path.insert(0, str(source))
        import torch
        import h5py
        import geoopt
        from segmentation_ShapeNetPart.hycore_seg_v1.train import ensure_idle
        from segmentation_ShapeNetPart.hycore_seg_v1.metrics import SegMetrics, NAMES as source_names
        from segmentation_ShapeNetPart.hycore_seg_v1.model import DGCNNPartSeg
        from segmentation_ShapeNetPart.hycore_seg_v1.utils import seed_all, state_hash
        from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2
        if tuple(source_names) != NAMES:
            raise ValueError("Category mapping differs from source")
        source_identity = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        source_dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                                 "--", "segmentation_ShapeNetPart"], text=True).strip()
        if source_dirty:
            raise ValueError("Frozen model source is dirty")
        gpu = ensure_idle(args.gpu)
        torch.set_num_threads(4)
        seed_all(4780)
        entries = json.loads(Path(args.runs_json).read_text(encoding="utf-8-sig"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("runs-json must contain a nonempty list")
        names = [item["name"] for item in entries]
        if len(set(names)) != len(names) or any(Path(name).name != name or name in {".", ".."} for name in names):
            raise ValueError("Run names must be unique simple directory names")
        data = Path(args.data).resolve(strict=True)
        data_manifest = json.loads((data / "manifest.json").read_text())
        data_hashes = {}
        for split in ("test", "trainval"):
            actual = file_hash(data / (split + ".h5"))
            expected = data_manifest["splits"][split]["file_sha256"]
            if actual != expected:
                raise ValueError("Data identity mismatch: " + split)
            data_hashes[split] = actual
        manifest = {"started": now(), "pid": os.getpid(), "args": vars(args),
                    "script_sha256": file_hash(__file__), "source_commit": source_identity,
                    "source_status": source_dirty, "gpu": gpu, "python": platform.python_version(),
                    "torch": torch.__version__, "geoopt": geoopt.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                    "data_manifest_sha256": file_hash(data / "manifest.json"),
                    "data_file_sha256": data_hashes, "runs": entries,
                    "state_update_policy": "none", "teacher": None, "smoke": False}
        write_json(out / "manifest.json", manifest)
        runtime = {"torch": torch, "h5py": h5py, "SegMetrics": SegMetrics, "state_hash": state_hash}
        summaries = {}
        for entry in entries:
            destination = out / entry["name"]
            destination.mkdir()
            checkpoint_path = Path(entry["checkpoint"]).resolve(strict=True)
            checkpoint_hash = file_hash(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            if config.get("num_points", 2048) != 2048:
                raise ValueError("Unexpected checkpoint point count")
            seed_all(int(config.get("seed", 4780)))
            kwargs = {"k": int(config.get("k", 40)), "dropout": float(config.get("dropout", .5))}
            if entry["kind"] == "dgcnn":
                model = DGCNNPartSeg(head="euclidean", **kwargs)
            elif entry["kind"] == "h":
                model = NativeTail2(arm="h", **kwargs)
            else:
                raise ValueError("Unknown model kind")
            model.load_state_dict(checkpoint["model"], strict=True)
            loaded_hash = state_hash(model.state_dict())
            if loaded_hash != state_hash(checkpoint["model"]):
                raise AssertionError("Loaded model differs from checkpoint")
            write_json(destination / "checkpoint_identity.json", {
                "path": str(checkpoint_path), "file_sha256": checkpoint_hash,
                "model_sha256": loaded_hash, "epoch": checkpoint.get("epoch"),
                "updates": checkpoint.get("updates"), "config": config,
                "checkpoint_source": checkpoint.get("source"), "evaluation_source": source_identity})
            del checkpoint
            gc.collect()
            model = model.cuda().eval()
            summaries[entry["name"]] = {}
            for split, data_split in (("test", "test"), ("trainval_clean_focus", "trainval")):
                target = destination / split
                target.mkdir()
                write_json(out / "status.json", {"state": "running", "run": entry["name"],
                                                  "split": split, "at": now()})
                result = evaluate_split(model, data / (data_split + ".h5"), split, target,
                                        args.batch_size, runtime,
                                        entry.get("expected_metrics") if split == "test" else None)
                summaries[entry["name"]][split] = result
                write_json(out / "summary.json", summaries)
                print(json.dumps({"run": entry["name"], "split": split, **result}), flush=True)
            if file_hash(checkpoint_path) != checkpoint_hash:
                raise AssertionError("Input checkpoint changed during diagnosis")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(out / "status.json", {"state": "complete", "runs": names, "at": now()})
    except Exception as error:
        write_json(out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "at": now()})
        write_json(out / "status.json", {"state": "failed", "error": repr(error), "at": now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    options = parser.parse_args()
    if options.batch_size < 1:
        parser.error("batch-size must be positive")
    run(options)
