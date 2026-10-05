"""Fixed-input, no-update HyCoRe local32/global64 diagnostic.

Three questions are kept separate: global versus local flip on fixed eval
embeddings; changing local32 training-BN membership with fixed negative IDs;
and inference with each local group's one-step BN buffers. The same checkpoint
parameters are used throughout. No backward, optimizer or parameter update is
performed; all model buffers and probe-start RNG states are restored.

Inputs are fixed clean first1024 points, with one ID-stable400-point kNN patch
copied into the whole prefix. This keeps child/whole view aliasing at forward,
but does not replay augmentation, the source's first whole crop/reordering,
random patch sizes, DDP communication or the complete training trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from .compare_hycore_class_geometry import (
        SOURCE, array_sha256, checkpoint_metadata, describe,
        deterministic_part_indices, load_split, matched_distances, sha256_file,
        utc_now, vector_geometry, write_json,
    )
else:
    from compare_hycore_class_geometry import (
        SOURCE, array_sha256, checkpoint_metadata, describe,
        deterministic_part_indices, load_split, matched_distances, sha256_file,
        utc_now, vector_geometry, write_json,
    )


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args(argv)
    if args.seed < 0 or not 1 <= args.batches <= 16:
        parser.error("Need nonnegative seed and1..16 fixed global64 batches")
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("output-dir must be NEW")
    return args


def loss_summary(output, labels, negatives, part_size=400):
    """Source c1 equations in FP64 from unchanged FP32 model outputs."""
    mu, nu, logits = (np.asarray(output[key], dtype=np.float64)
                      for key in ("mu", "nu", "logits"))
    wr, wd = vector_geometry(mu)
    pr, pd = vector_geometry(nu)
    positive = matched_distances(mu, nu)
    negative = matched_distances(mu, nu[negatives])
    contrastive = np.maximum(positive - negative + 4., 0.)
    radial = np.maximum(-wd + pd + 1000. / part_size, 0.)
    shifted = logits - logits.max(1, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    gold_log = log_probability[np.arange(len(labels)), labels]
    ce = -(.8 * gold_log + .2 / 39 * (log_probability.sum(1) - gold_log))
    return {
        "ce": float(ce.mean()), "rcontr": float(contrastive.mean()),
        "rhier": float(radial.mean()),
        "base": float(ce.mean() + .01 * contrastive.mean() + .01 * radial.mean()),
        "rcontr_active_percent": float(100 * np.mean(contrastive > 0)),
        "rhier_active_percent": float(100 * np.mean(radial > 0)),
        "same_class_negative_percent": float(100 * np.mean(labels == labels[negatives])),
        "positive_distance": describe(positive), "negative_distance": describe(negative),
        "whole_radius": describe(wr), "part_radius": describe(pr),
        "whole_depth": describe(wd), "part_depth": describe(pd),
        "oa_percent": float(100 * np.mean(logits.argmax(1) == labels)),
    }


def output_changes(first, second, ids, labels):
    per_id = []
    arrays = {}
    for name in ("mu", "nu"):
        r0, d0 = vector_geometry(first[name])
        r1, d1 = vector_geometry(second[name])
        prefix = "whole" if name == "mu" else "part"
        arrays[prefix + "_radius_first"] = r0
        arrays[prefix + "_radius_second"] = r1
        arrays[prefix + "_depth_first"] = d0
        arrays[prefix + "_depth_second"] = d1
        arrays[prefix + "_radius_delta"] = r1 - r0
        arrays[prefix + "_depth_delta"] = d1 - d0
        arrays[prefix + "_hyperbolic_displacement"] = matched_distances(first[name], second[name])
    prediction0 = first["logits"].argmax(1)
    prediction1 = second["logits"].argmax(1)
    for index, sample_id in enumerate(ids):
        per_id.append({"id": int(sample_id), "class_id": int(labels[index]),
                       "prediction_first": int(prediction0[index]),
                       "prediction_second": int(prediction1[index]),
                       **{key: float(value[index]) for key, value in arrays.items()}})
    return {"summary": {key: describe(value) for key, value in arrays.items()},
            "prediction_disagreement_percent": float(100 * np.mean(prediction0 != prediction1)),
            "per_id": per_id}


def main(argv=None):
    args = arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"format": "hycore-fixed-batch-split-probe-1", "status": "starting",
                "started_utc": utc_now(), "run_label": args.run_label,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                "command": sys.argv, "seed": args.seed, "batches": args.batches,
                "whole_points": 1024, "part_points": 400,
                "c": 1., "D": 256, "parameter_updates": 0,
                "input": "clean first1024; fixed ID-stable patch400 overwrites whole prefix; no first whole reordering",
                "geometry": "FP64 source c1 equations from unchanged FP32 embeddings; rates in percent",
                "limits": "Conditional checkpoint sensitivity only; not full training causal isolation or DDP replay"}
    manifest["audit_script_sha256"] = sha256_file(__file__)
    manifest["code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        import torch
        sys.path.insert(0, str(SOURCE))
        from models.pointmlp import Hype_pointMLP

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Source PointMLP native FPS requires CUDA")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not isinstance(saved.get("net"), dict) or not saved["net"]:
            raise ValueError("Require original-compatible checkpoint['net']")
        model = Hype_pointMLP().to(device)
        model.load_state_dict({key[7:] if key.startswith("module.") else key: value
                               for key, value in saved["net"].items()}, strict=True)
        model.requires_grad_(False)
        model.eval()
        for name in ("manifold", "manifold2"):
            if not math.isclose(float(getattr(model, name).c.detach()), 1., abs_tol=1e-5):
                raise ValueError("Probe requires loaded c1 checkpoint")
        manifest["checkpoint_fields_unmodified"] = checkpoint_metadata(saved)
        import os
        manifest["runtime"] = {"torch": torch.__version__, "numpy": np.__version__,
                               "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        original_modes = {module: module.training for module in model.modules()}
        original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                        torch.cuda.get_rng_state_all())

        def restore_rng():
            random.setstate(original_rng[0])
            np.random.set_state(original_rng[1])
            torch.set_rng_state(original_rng[2])
            torch.cuda.set_rng_state_all(original_rng[3])

        def set_buffers(values):
            current = dict(model.named_buffers())
            if current.keys() != values.keys():
                raise RuntimeError("Model buffer keys changed")
            for name, value in values.items():
                current[name].copy_(value.to(current[name].device))

        def snapshot_buffers():
            return {name: value.detach().clone() for name, value in model.named_buffers()}

        def parameter_hash():
            digest = hashlib.sha256()
            for name, value in model.named_parameters():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        parameters_before = parameter_hash()
        points, labels, ids, identity = load_split(args.data_dir, "train", 0)
        if len(ids) < args.batches * 64:
            raise ValueError("Not enough train IDs for distinct probe batches")
        selected = np.random.default_rng(args.seed).permutation(len(ids))[:args.batches * 64]
        clean = points[selected].copy()
        gold = labels[selected]
        sample_ids = ids[selected]
        centers, indices = deterministic_part_indices(clean, sample_ids, "train", args.seed, [400])
        fixed_whole = clean.copy()
        patches = clean[np.arange(len(clean))[:, None], indices[400]].copy()
        fixed_whole[:, :400] = patches
        source_before = array_sha256(points)
        prepared_before = array_sha256(fixed_whole)
        manifest["data_identity"] = identity
        manifest["probe_ids_sha256"] = array_sha256(sample_ids)
        manifest["patch_centers_sha256"] = array_sha256(centers)
        manifest["part_indices_sha256"] = array_sha256(indices[400])
        manifest["prepared_whole_sha256"] = prepared_before
        clean_probe = clean[:64]
        clean_probe_gold = gold[:64]
        clean_probe_ids = sample_ids[:64]

        def clean_output(buffers):
            set_buffers(buffers)
            model.eval()
            features, logits = [], []
            for start in range(0, len(clean_probe), 16):
                cloud = torch.from_numpy(np.ascontiguousarray(clean_probe[start:start + 16])).to(device)
                mu, score = model(cloud.transpose(1, 2).contiguous())
                if not torch.isfinite(mu).all() or not torch.isfinite(score).all():
                    raise RuntimeError("Nonfinite clean buffer-probe output")
                features.append(mu.cpu().numpy())
                logits.append(score.cpu().numpy())
            return {"mu": np.concatenate(features), "logits": np.concatenate(logits)}

        def grouped_output(whole, groups, training_bn):
            output = {"mu": np.empty((64, 256), np.float32),
                      "nu": np.empty((64, 256), np.float32),
                      "logits": np.empty((64, 40), np.float32)}
            buffers = []
            for group in groups:
                # Each local rank begins at the same broadcast checkpoint buffers.
                set_buffers(original_buffers)
                restore_rng()
                model.train(training_bn)
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                        module.eval()
                cloud = torch.from_numpy(np.ascontiguousarray(whole[group])).to(device).transpose(1, 2).contiguous()
                child = cloud[:, :, :400]  # Same storage; whole prefix already overwritten.
                nu, _unused = model(child, emb=True)
                mu, score = model(cloud)
                for key, value in (("mu", mu), ("nu", nu), ("logits", score)):
                    if not torch.isfinite(value).all():
                        raise RuntimeError(f"Nonfinite grouped {key}")
                    output[key][group] = value.cpu().numpy()
                if training_bn:
                    for name, value in model.named_buffers():
                        if name.endswith("num_batches_tracked") and int(value - original_buffers[name]) != 2:
                            raise RuntimeError("Expected exactly two local BN updates")
                buffers.append(snapshot_buffers())
            return output, buffers

        reports = []
        with torch.no_grad():
            try:
                clean_reference = clean_output(original_buffers)
                global_negative = np.arange(63, -1, -1)
                local_negative = np.concatenate((np.arange(31, -1, -1), np.arange(63, 31, -1)))
                contiguous = (np.arange(32), np.arange(32, 64))
                for batch in range(args.batches):
                    start = batch * 64
                    whole = fixed_whole[start:start + 64]
                    batch_gold = gold[start:start + 64]
                    batch_ids = sample_ids[start:start + 64]
                    permutation = np.random.default_rng(np.random.SeedSequence([args.seed, 731, batch])).permutation(64)
                    alternate = (permutation[:32], permutation[32:])
                    eval_output, _ = grouped_output(whole, contiguous, False)
                    eval_global = loss_summary(eval_output, batch_gold, global_negative)
                    eval_local = loss_summary(eval_output, batch_gold, local_negative)
                    if eval_global["ce"] != eval_local["ce"] or eval_global["rhier"] != eval_local["rhier"]:
                        raise RuntimeError("Pure pairing probe changed CE/radial loss")
                    normal, normal_buffers = grouped_output(whole, contiguous, True)
                    regrouped, regrouped_buffers = grouped_output(whole, alternate, True)
                    bn_report = {"contiguous": loss_summary(normal, batch_gold, global_negative),
                                 "regrouped": loss_summary(regrouped, batch_gold, global_negative),
                                 "changes": output_changes(normal, regrouped, batch_ids, batch_gold),
                                 "negative_id_mapping_fixed": True}
                    buffer_report = []
                    for grouping_name, local_buffers in (("contiguous", normal_buffers), ("regrouped", regrouped_buffers)):
                        for local_rank, local_state in enumerate(local_buffers):
                            output = clean_output(local_state)
                            radius, depth = vector_geometry(output["mu"])
                            reference_r, reference_d = vector_geometry(clean_reference["mu"])
                            buffer_report.append({"grouping": grouping_name, "local_rank": local_rank,
                                "clean_probe_count": len(clean_probe_ids),
                                "oa_percent": float(100 * np.mean(output["logits"].argmax(1) == clean_probe_gold)),
                                "prediction_disagreement_vs_checkpoint_percent": float(100 * np.mean(
                                    output["logits"].argmax(1) != clean_reference["logits"].argmax(1))),
                                "whole_radius": describe(radius), "whole_depth": describe(depth),
                                "radius_delta_vs_checkpoint": describe(radius - reference_r),
                                "depth_delta_vs_checkpoint": describe(depth - reference_d),
                                "hyperbolic_displacement_vs_checkpoint": describe(matched_distances(
                                    clean_reference["mu"], output["mu"]))})
                    reports.append({"batch": batch + 1, "ids": batch_ids.tolist(), "labels": batch_gold.tolist(),
                        "contiguous_groups_ids": [batch_ids[group].tolist() for group in contiguous],
                        "regrouped_groups_ids": [batch_ids[group].tolist() for group in alternate],
                        "pure_negative_relation": {"eval_global_flip": eval_global, "eval_two_local_flips": eval_local,
                            "ce_and_rhier_identical": True, "rcontr_delta_global_minus_local": eval_global["rcontr"] - eval_local["rcontr"]},
                        "local_bn_membership": bn_report, "post_step_buffers_clean_eval": buffer_report})
                    print(f"{args.run_label}: fixed-batch {batch + 1}/{args.batches}", flush=True)
            finally:
                set_buffers(original_buffers)
                restore_rng()
                for module, training in original_modes.items():
                    module.training = training
        buffers_restored = all(torch.equal(value, original_buffers[name]) for name, value in model.named_buffers())
        np_rng = np.random.get_state()
        rng_restored = (random.getstate() == original_rng[0] and np_rng[0] == original_rng[1][0]
                        and np.array_equal(np_rng[1], original_rng[1][1]) and np_rng[2:] == original_rng[1][2:]
                        and torch.equal(torch.get_rng_state(), original_rng[2])
                        and all(torch.equal(left, right) for left, right in zip(torch.cuda.get_rng_state_all(), original_rng[3])))
        parameters_unchanged = parameter_hash() == parameters_before
        input_unchanged = array_sha256(points) == source_before and array_sha256(fixed_whole) == prepared_before
        checkpoint_unchanged = sha256_file(args.checkpoint) == manifest["checkpoint_sha256"]
        if not all((buffers_restored, rng_restored, parameters_unchanged, input_unchanged, checkpoint_unchanged)):
            raise RuntimeError("Probe restoration/immutability check failed")
        write_json(args.output_dir / "probe.json", {"run_label": args.run_label,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "batches": reports,
            "clean_probe_ids": clean_probe_ids.tolist(),
            "clean_checkpoint_reference_oa_percent": float(100 * np.mean(clean_reference["logits"].argmax(1) == clean_probe_gold)),
            "interpretation": "Effects conditioned on this fixed checkpoint and inputs; no attribution of full-run differences"})
        manifest.update({"status": "completed", "finished_utc": utc_now(), "wall_seconds": time.monotonic() - started,
            "buffers_restored": buffers_restored, "rng_restored_to_probe_start": rng_restored,
            "parameters_unchanged": parameters_unchanged, "source_and_prepared_inputs_unchanged": input_unchanged,
            "checkpoint_unchanged": checkpoint_unchanged, "all_finite": True})
        write_json(args.output_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update({"status": "failed", "finished_utc": utc_now(),
                         "wall_seconds": time.monotonic() - started, "error": f"{type(error).__name__}: {error}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
