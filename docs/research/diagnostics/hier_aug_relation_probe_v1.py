"""Two-view whole probes with explicit physical crop-point controls.

clean_crop_held: one shared permutation and clean-coordinate crop membership;
only source-range anisotropic scale/shift changes. source_natural: independent
scale/shift/shuffle and new augmented-coordinate kNN crops. These latter view
changes intentionally combine augmentation, physical anchors and membership.
Only sequential local32 no-grad forwards and global64 leaf derivatives are
computed. No backbone VJP, optimizer update or source-file modification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

import hier_whole_mechanism_v1 as mech
from hier_gumbel_relation_stability_v1 import checkpoint_identity


FAMILIES = ("clean_crop_held", "source_natural")
VIEW_OFFSET = {"A": 0, "B": 1000000007}


def view_clouds(clean, seed, epoch, step, family, view):
    coordinates, original_ids, geometry = [], [], []
    for position, points in enumerate(clean):
        key = seed + epoch * 31000003 + step * 10007 + position * 1009
        transform = np.random.RandomState((key + VIEW_OFFSET[view]) % (2 ** 32))
        scale, shift = transform.uniform(2 / 3, 3 / 2, size=3), transform.uniform(-.2, .2, size=3)
        permutation = np.arange(len(points))
        if family == "clean_crop_held":
            np.random.RandomState((key + 73000009) % (2 ** 32)).shuffle(permutation)
        elif family == "source_natural":
            transform.shuffle(permutation)
        else:
            raise ValueError("Unknown family")
        augmented = (points * scale + shift).astype(np.float32)[permutation]
        coordinates.append(augmented)
        original_ids.append(permutation)
        geometry.append(points[permutation] if family == "clean_crop_held" else augmented)
    return (torch.from_numpy(np.stack(coordinates)), torch.from_numpy(np.stack(original_ids)),
            torch.from_numpy(np.stack(geometry)))


def nearest_row(points, center, count):
    # Same squared-Euclidean ranking as source kNN, computing only the used
    # center row rather than all N*N rows. Floating tie order is not claimed
    # bitwise identical to historical batched CUDA kNN.
    anchor = points[int(center)]
    score = 2 * (points * anchor).sum(-1) - points.square().sum(-1) - anchor.square().sum()
    return score.topk(int(count)).indices


@torch.no_grad()
def crop_selection(geometry, original_ids, centers, counts):
    whole, child, anchor_whole, anchor_child = [], [], [], []
    pre_ids, post_ids, child_ids = [], [], []
    for position in range(len(geometry)):
        wi = nearest_row(geometry[position], centers[0][position], counts[0])
        ci = nearest_row(geometry[position, wi], centers[1][position], counts[1])
        physical_whole = original_ids[position, wi].clone()
        physical_child = physical_whole[ci].clone()
        overwritten = physical_whole.clone()
        overwritten[:counts[1]] = physical_child
        whole.append(wi)
        child.append(ci)
        anchor_whole.append(int(original_ids[position, int(centers[0][position])]))
        anchor_child.append(int(physical_whole[int(centers[1][position])]))
        pre_ids.append(physical_whole)
        post_ids.append(overwritten)
        child_ids.append(physical_child)
    return {"whole_indices": torch.stack(whole), "child_indices": torch.stack(child),
            "physical_whole_anchor": np.asarray(anchor_whole), "physical_child_anchor": np.asarray(anchor_child),
            "whole_preoverwrite_original_ids": torch.stack(pre_ids).cpu().numpy(),
            "whole_postoverwrite_original_ids": torch.stack(post_ids).cpu().numpy(),
            "child_original_ids": torch.stack(child_ids).cpu().numpy()}


def overwritten_views(cloud, selection, positions, counts):
    source = cloud.transpose(1, 2).contiguous()
    whole = source[:, :, :counts[0]]
    for local, position in enumerate(positions.tolist()):
        whole[local] = source[local, :, selection["whole_indices"][position]]
    before = whole.detach().clone()
    child = whole[:, :, :counts[1]]
    for local, position in enumerate(positions.tolist()):
        child[local] = whole[local, :, selection["child_indices"][position]]
    if child.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr():
        raise RuntimeError("Child/whole alias lost")
    if source.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr():
        raise RuntimeError("Whole/source alias lost")
    if torch.equal(before, whole):
        raise RuntimeError("Child did not overwrite whole")
    return child, whole


@torch.no_grad()
def collect(backbone, cloud, selection, counts, condition, buffers, seed, device):
    mech.set_condition(backbone, condition)
    bn = [module for module in backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    outputs, changes = {}, []
    try:
        for group_number, positions in enumerate(mech.condition_groups(condition)):
            mech.restore_buffers(backbone, buffers)
            mech.seed_selected(seed + group_number * 1000003, device)
            prior = [int(module.num_batches_tracked) for module in bn]
            child, whole = overwritten_views(cloud[positions.to(device)].clone(), selection, positions, counts)
            nu, _ = backbone(child, emb=True)
            mu, logits = backbone(whole)
            for name, value in {"mu": mu, "nu": nu, "logits": logits}.items():
                mech.finite_tensor(value, condition + "/" + name)
                if name not in outputs:
                    outputs[name] = torch.empty((64, *value.shape[1:]), dtype=value.dtype, device=device)
                outputs[name][positions.to(device)] = value
            delta = [int(module.num_batches_tracked) - count for module, count in zip(bn, prior)]
            expected = 0 if condition == "eval" else 2
            if any(count != expected for count in delta):
                raise RuntimeError("Unexpected local32 BN update count")
            changes.append({"min": min(delta), "max": max(delta), "expected": expected})
    finally:
        mech.restore_buffers(backbone, buffers)
    outputs["child_count"] = counts[1]
    return outputs, changes


def membership_comparison(first, second):
    result = {}
    for key in ("whole_preoverwrite_original_ids", "whole_postoverwrite_original_ids", "child_original_ids"):
        values = []
        for left, right in zip(first[key], second[key]):
            old, new = set(left.tolist()), set(right.tolist())
            values.append(len(old & new) / len(old | new))
        result[key + "_jaccard_per_position"] = values
        result[key + "_jaccard_mean"] = float(np.mean(values))
    for key in ("physical_whole_anchor", "physical_child_anchor"):
        result[key + "_agreement"] = float((first[key] == second[key]).mean())
    return result


def self_test():
    torch.set_num_threads(2)
    clean = np.random.RandomState(9).normal(size=(4, 20, 3)).astype(np.float32)
    centers = (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]))
    counts = (16, 8)
    held = {}
    for view in ("A", "B"):
        cloud, original_ids, geometry = view_clouds(clean, 22, 0, 0, "clean_crop_held", view)
        selected = crop_selection(geometry, original_ids, centers, counts)
        child, whole = overwritten_views(cloud.clone(), selected, torch.arange(4), counts)
        assert child.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
        for position in range(4):
            torch.testing.assert_close(child[position], cloud[position, selected["whole_indices"][position][selected["child_indices"][position]]].T)
        held[view] = (cloud, selected, original_ids)
    assert not torch.equal(held["A"][0], held["B"][0])
    assert torch.equal(held["A"][2], held["B"][2])
    compared = membership_comparison(held["A"][1], held["B"][1])
    assert all(value == 1 for key, value in compared.items() if key.endswith("mean") or key.endswith("agreement"))
    natural = []
    for view in ("A", "B"):
        cloud, original_ids, geometry = view_clouds(clean, 22, 0, 0, "source_natural", view)
        np.testing.assert_array_equal(np.sort(original_ids.numpy(), axis=-1), np.tile(np.arange(20), (4, 1)))
        natural.append(crop_selection(geometry, original_ids, centers, counts))
    membership_comparison(*natural)
    print(json.dumps({"self_test": "passed", "checks": ["held shared permutation", "held original-point membership/anchors identical", "view coordinates differ", "source-natural permutation identity tracking", "part overwrite alias", "selected child coordinate identity"]}))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--plan-epoch", type=int, default=0)
    parser.add_argument("--families", default="clean_crop_held,source_natural")
    parser.add_argument("--conditions", default="eval,train_bn")
    parser.add_argument("--hotspot-ids", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        if not all((args.checkpoint, args.data_dir, args.run_dir)):
            parser.error("checkpoint/data-dir/new run-dir required")
        args.families = args.families.split(",")
        args.conditions = args.conditions.split(",")
        args.hotspot_ids = [int(value) for value in args.hotspot_ids.split(",") if value]
        if not 1 <= args.batches <= 16 or args.seed != 22 or args.plan_epoch < 0:
            parser.error("Require batches1..16, seed22 and nonnegative epoch")
        if any(value not in FAMILIES for value in args.families) or len(set(args.families)) != len(args.families):
            parser.error("Invalid families")
        if any(value not in ("eval", "train_bn", "train_bn_regrouped") for value in args.conditions) or len(set(args.conditions)) != len(args.conditions):
            parser.error("Invalid conditions")
    return args


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.distributed.is_initialized():
        raise RuntimeError("Single process only")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    root_manifest = {"status": "starting", "diagnostic_only": True, "optimizer_updates": 0,
        "checkpoint_sha256": mech.digest(args.checkpoint), "script_sha256": mech.digest(__file__),
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv]}
    try:
        root_manifest["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mech.ROOT, text=True).strip()
        root_manifest["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=mech.ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        root_manifest["commit"], root_manifest["branch"] = None, None
    mech.save_json(args.run_dir / "manifest.json", root_manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Production pointMLP needs CUDA")
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
        from models.pointmlp import Hype_pointMLP
        from inter_hierarchy_MN40.hier_proxy_v2_online.train import load_shards
        from inter_hierarchy_MN40.hier_proxy_scratch_v3.train import freeze_curvature
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.distributed import shared_crop_counts
        from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = checkpoint_identity(saved, root_manifest["checkpoint_sha256"])
        backbone = Hype_pointMLP().to(device)
        backbone.load_state_dict(saved["net"], strict=True)
        freeze_curvature(backbone)
        proxy = HIERLoss().to(device)
        proxy.load_state_dict(saved["proxy"], strict=True)
        with torch.no_grad():
            proxy_points = proxy.proxies().detach()
        buffers = mech.snapshot_buffers(backbone)
        model_hash, proxy_hash = mech.module_hash(backbone), mech.module_hash(proxy)
        if model_hash != identity["backbone_initial_sha256"] or proxy_hash != identity["proxy_initial_sha256"]:
            raise RuntimeError("Loaded model/proxy semantic identity mismatch")
        train_ids = np.asarray(saved["train_ids"], dtype=np.int64)
        del saved
        points, labels, shards = load_shards(args.data_dir, "train")
        shard_hashes = {name: mech.digest(args.data_dir / name) for name in shards}
        sampler = SourceClassBatchSampler(labels[train_ids], seed=args.seed, steps=200)
        dropouts = [name for name, module in backbone.named_modules() if isinstance(module, nn.modules.dropout._DropoutNd)]
        original_rng = mech.capture_selected_rng(device)
        directories, manifests = {}, {}
        for family in args.families:
            for view in ("A", "B"):
                key = family + "_" + view
                directory = args.run_dir / key
                directory.mkdir(exist_ok=False)
                directories[key] = directory
                manifests[key] = {**identity, "status": "running", "diagnostic_only": True, "optimizer_updates": 0,
                    "commit": root_manifest["commit"], "branch": root_manifest["branch"],
                    "script_sha256": root_manifest["script_sha256"], "global_batch": 64, "local_forward_batch": 32,
                    "seed": args.seed, "plan_epoch": args.plan_epoch, "hotspot_ids": args.hotspot_ids,
                    "input_mode": "augmented", "input_view": view, "view_augmentation_seed_offset": VIEW_OFFSET[view],
                    "augmentation_family": family, "conditions": args.conditions, "sampler": sampler.state_dict(),
                    "crop_policy": "clean fixed membership and physical anchors" if family == "clean_crop_held" else "augmented coordinates, independent shuffle, numeric centers held",
                    "input_shard_sha256": shard_hashes, "labels_sha256": mech.array_hash(labels),
                    "dropout_modules": dropouts, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "logical_gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(device),
                    "torch_version": torch.__version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "semantics": ["No optimizer update, no shared-backbone VJP, only whole leaf partials.",
                        "Every local32 restores old checkpoint rank0 buffers, then part and overwritten whole; every panel independent.",
                        "Anisotropic scale/shift stay within source augmentation ranges; RNG streams are fixed diagnostic streams.",
                        "Numeric centers/counts/global IDs/negative mapping/Gumbel seed stay fixed across views.",
                        "clean_crop_held selects members on shared-permutation clean coordinates; this is a diagnostic crop intervention.",
                        "source_natural combines coordinate changes, point shuffle, physical anchors and kNN membership; physical IDs are recorded.",
                        "Only selected center rows of squared-Euclidean kNN are computed; floating tie order is not historical CUDA replay."]}
                mech.save_json(directory / "manifest.json", manifests[key])
        for step in range(args.batches):
            subset, ids, gold, modified = mech.make_plan(sampler, step, args.plan_epoch, train_ids, labels, args.hotspot_ids)
            counts = shared_crop_counts(args.seed, args.plan_epoch, step)
            clean = points[ids, :1024].copy()
            _, centers = mech.prepare_inputs(points, ids, args.seed, args.plan_epoch, step, "clean", counts)
            relation_seed = args.seed + args.plan_epoch * 10000019 + step * 10007
            selection_by_family = {}
            for family in args.families:
                selection_by_family[family] = {}
                for view in ("A", "B"):
                    key = family + "_" + view
                    directory = directories[key]
                    cloud_cpu, original_ids, crop_geometry = view_clouds(clean, args.seed, args.plan_epoch, step, family, view)
                    cloud = cloud_cpu.to(device)
                    selection = crop_selection(crop_geometry.to(device), original_ids.to(device), centers, counts)
                    selection_by_family[family][view] = selection
                    report = {"batch": step, "ids": ids.tolist(), "labels": gold.tolist(), "subset_indices": subset.tolist(),
                        "global_flip_negative_ids": ids[::-1].tolist(), "whole_points": counts[0], "part_points": counts[1],
                        "whole_centers": centers[0].tolist(), "part_centers": centers[1].tolist(), "relation_seed": relation_seed,
                        "input_cloud_sha256": mech.array_hash(cloud_cpu), "original_point_permutation_sha256": mech.array_hash(original_ids),
                        "physical_whole_anchor": selection["physical_whole_anchor"].tolist(),
                        "physical_child_anchor": selection["physical_child_anchor"].tolist(),
                        "forced_panel_changes": modified, "conditions": {}}
                    for condition in args.conditions:
                        torch.cuda.reset_peak_memory_stats(device)
                        outputs, bn_changes = collect(backbone, cloud, selection, counts, condition, buffers,
                            args.seed + args.plan_epoch * 10000019 + step * 1009, device)
                        leaf = mech.leaf_components(outputs, torch.from_numpy(gold).to(device), torch.from_numpy(ids).to(device),
                                                    proxy_points, relation_seed, classifier=backbone.classifier)
                        report["conditions"][condition] = {"geometry": mech.geometry(outputs), "mining": leaf["mining"]["stats"],
                            "sample": leaf["sample_stats"], "losses": leaf["losses"], "BN_updates": bn_changes,
                            "whole_leaf_partials": mech.whole_partial_summary(outputs, leaf, gold, ids, args.hotspot_ids),
                            "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                            "non_BN_stochastic_modules_enabled": False}
                        arrays = {name: outputs[name].cpu().numpy() for name in ("mu", "nu", "logits")}
                        arrays.update(ids=ids, labels=gold, triplets=leaf["mining"]["triplets"].cpu().numpy(),
                            mutual=leaf["mining"]["mutual"].cpu().numpy(),
                            pair_proxy_idx=leaf["details"]["pair_proxy_idx"].cpu().numpy(),
                            triple_proxy_idx=leaf["details"]["triple_proxy_idx"].cpu().numpy(),
                            original_point_permutation=original_ids.numpy())
                        for field in ("physical_whole_anchor", "physical_child_anchor", "whole_preoverwrite_original_ids", "whole_postoverwrite_original_ids", "child_original_ids"):
                            arrays[field] = selection[field]
                        np.savez_compressed(directory / f"batch_{step:03d}_{condition}.npz", **arrays)
                        print(json.dumps({"batch": step, "family": family, "view": view, "condition": condition, "status": "passed"}), flush=True)
                    mech.save_json(directory / f"batch_{step:03d}.json", report)
                    manifests[key]["completed_batches"] = step + 1
                    mech.save_json(directory / "manifest.json", manifests[key])
                    del cloud, crop_geometry, outputs, leaf
            physical = {family: membership_comparison(views["A"], views["B"]) for family, views in selection_by_family.items()}
            if "clean_crop_held" in physical:
                if any(value != 1 for name, value in physical["clean_crop_held"].items() if name.endswith("mean") or name.endswith("agreement")):
                    raise RuntimeError("Held-crop view point-membership controls failed")
            mech.save_json(args.run_dir / f"physical_controls_batch_{step:03d}.json", physical)
        mech.restore_buffers(backbone, buffers)
        mech.restore_selected_rng(original_rng, device)
        if mech.module_hash(backbone) != model_hash or mech.module_hash(proxy) != proxy_hash:
            raise RuntimeError("Model or proxy state changed")
        if mech.digest(args.checkpoint) != root_manifest["checkpoint_sha256"] or {name: mech.digest(args.data_dir / name) for name in shards} != shard_hashes:
            raise RuntimeError("Source checkpoint or shards changed")
        if any(parameter.grad is not None for parameter in list(backbone.parameters()) + list(proxy.parameters())):
            raise RuntimeError("Parameter gradient buffers accumulated")
        for key, manifest in manifests.items():
            manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                state_checks={"model_unchanged": True, "proxy_unchanged": True, "BN_restored": True,
                              "source_checkpoint_unchanged": True, "input_shards_unchanged": True})
            mech.save_json(directories[key] / "manifest.json", manifest)
        root_manifest.update(status="completed", wall_seconds=time.perf_counter() - started,
                             finished_utc=datetime.now(timezone.utc).isoformat(), probe_directories=[str(path) for path in directories.values()])
        mech.save_json(args.run_dir / "manifest.json", root_manifest)
    except BaseException as error:
        root_manifest.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        mech.save_json(args.run_dir / "manifest.json", root_manifest)
        raise


if __name__ == "__main__":
    main()
