"""CPU frozen-cache replay of V6 sample/proxy objectives and gradients.

Uses the production FP32 c1 distance, mining and hard-Gumbel functions. This
is a new replay on clean eval embeddings, not historical CUDA training draws,
not a backbone/BN diagnostic and not independent morphology validation.
The equal-radius intervention changes sample mining only; every loss still
uses the original whole and proxy coordinates. Inputs are never overwritten.

Example (paths supplied by the operator, private output outside Git):
  python docs/research/diagnostics/hier_proxy_objective_probe_v1.py \
    --cache <feature_cache.npz> --checkpoint <matching_full_checkpoint.pth> \
    --output-dir <new_directory> --batches 16

Optional --proxy-steps 100 runs three frozen-whole AdamW arms per mining mode
and two inexpensive zero-gradient controls (800 optimizer updates in total).
No backbone is instantiated; no GPU is used. These are optimizer updates and
are explicitly distinguished from the default no-update probe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import (
    mine_sample_triplets, mine_reciprocal_triplets, poincare_distance,
)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss, expmap0_c1, ghhc_loss
from inter_hierarchy_MN40.hier_proxy_scratch_v6.structure_monitor import (
    StructureAccumulator, ancestor_summary, depth, proxy_update_stats,
)

NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()
MODES = ("original", "sample_mining_equal_radius")
COMPONENTS = ("sample", "proxy", "combined")
CHECKPOINT_FORMAT = "hycore-hier-v6-h20-selfk300-1"
SOURCE_PROTOCOL = {"c": 1, "D": 256, "P": 512, "sample_K": 20, "proxy_K": 20,
                   "T": 50, "margin": .1, "tau": .1, "global_batch": 64,
                   "world_size": 2, "classes_per_rank": 16, "instances_per_class": 2,
                   "precision": "FP32", "epochs": 300}


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha_tensor(value):
    return sha_array(value.detach().cpu().numpy())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def moments(values):
    values = torch.as_tensor(values).detach().double().reshape(-1)
    if not values.numel():
        return {"n": 0, "sum": 0., "sum_squares": 0., "positive": 0, "negative": 0, "zero": 0}
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Nonfinite diagnostic quantity")
    return {"n": values.numel(), "sum": float(values.sum()), "sum_squares": float(values.square().sum()),
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum())}


def add_moments(target, current):
    for key, value in current.items():
        target[key] = target.get(key, 0) + value


def finish_moments(value):
    n = value.get("n", 0)
    mean = value.get("sum", 0.) / n if n else None
    return {**value, "mean": mean,
            "std": max(0., value["sum_squares"] / n - mean ** 2) ** .5 if n else None,
            "positive_fraction": value.get("positive", 0) / n if n else None,
            "negative_fraction": value.get("negative", 0) / n if n else None,
            "zero_fraction": value.get("zero", 0) / n if n else None}


def gradient_values(points, gradient):
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("Nonfinite partial gradient")
    direction = points.detach() / points.detach().norm(dim=-1, keepdim=True).clamp_min(1e-15)
    signed = (gradient.detach() * direction).sum(-1)
    angular = gradient.detach() - signed[:, None] * direction
    return {"signed_radial_gradient": signed,
            "angular_gradient_norm": angular.norm(dim=-1),
            "total_gradient_norm": gradient.detach().norm(dim=-1)}


def gradient_cosine(left, right):
    norm = left.norm() * right.norm()
    return float((left * right).sum() / norm) if float(norm) else None


def validate_checkpoint_protocol(saved):
    """Reject silently replaying another protocol with this fixed V6 loss."""
    if saved.get("format") != CHECKPOINT_FORMAT or saved.get("model_selection_only", False):
        raise ValueError("Require an original complete V6-H20 checkpoint format")
    config = saved.get("training_config", {})
    for key, expected in SOURCE_PROTOCOL.items():
        if config.get(key) != expected:
            raise ValueError(f"Unsupported checkpoint geometry/protocol: {key}")
    for key in ("self_negative", "extra_HIER_tangent_cap", "HIER_backward_hook", "smoke"):
        if config.get(key) is not False:
            raise ValueError(f"Unsupported checkpoint protocol flag: {key}")
    epoch, completed = saved.get("epoch"), saved.get("completed_epochs")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= config["epochs"]:
        raise ValueError("Invalid original V6 checkpoint epoch")
    if completed != epoch:
        raise ValueError("Checkpoint completed_epochs must equal epoch")


def validate_inputs(cache_path, checkpoint_path, metadata_path):
    hashes = {"cache": sha_file(cache_path), "checkpoint": sha_file(checkpoint_path),
              "cache_metadata": sha_file(metadata_path)}
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("Cache metadata must report complete")
    if metadata["checkpoint"]["sha256"] != hashes["checkpoint"]:
        raise ValueError("Cache and checkpoint SHA256 do not match")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_protocol(saved)
    if not all(k in saved for k in (
            "proxy", "net", "training_config", "train_ids", "split_sha256", "epoch")):
        raise ValueError("Require a complete same-epoch V6 model/proxy checkpoint")
    config = saved["training_config"]
    if metadata["checkpoint"].get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Cache metadata must identify the original complete V6 format")
    if saved["epoch"] != metadata["checkpoint"]["epoch"]:
        raise ValueError("Cache/checkpoint epoch mismatch")
    data_meta = metadata["dataset"]
    if data_meta.get("split") != "train_ids" or data_meta.get("test_read") is not False:
        raise ValueError("Only the verified clean training cache is supported")
    if data_meta.get("checkpoint_split_sha256") != saved["split_sha256"]:
        raise ValueError("Checkpoint split identity mismatch")
    shards = data_meta.get("training_shards_sorted")
    if not shards or list(shards) != sorted(shards) or len(set(shards)) != len(shards):
        raise ValueError("Require unique sorted-shard identity in cache metadata")
    with np.load(cache_path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in ("mu", "proxy_ball", "sample_ids", "labels")}
    ids = np.asarray(arrays["sample_ids"], dtype=np.int64).reshape(-1)
    labels = np.asarray(arrays["labels"], dtype=np.int64).reshape(-1)
    if len(ids) != 8856 or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected 8856 distinct V6 train IDs")
    np.testing.assert_array_equal(ids, np.asarray(saved["train_ids"], dtype=np.int64))
    if sha_array(ids) != data_meta["ids_sha256"] or sha_array(labels) != data_meta["labels_sha256"]:
        raise ValueError("Cache IDs/labels checksum mismatch")
    if arrays["mu"].dtype != np.float32 or arrays["proxy_ball"].dtype != np.float32:
        raise ValueError("Production replay requires original FP32 caches")
    if arrays["mu"].shape != (8856, 256) or arrays["proxy_ball"].shape != (512, 256):
        raise ValueError("Unexpected cache shape")
    if set(labels.tolist()) != set(range(40)):
        raise ValueError("Expected all 40 ModelNet classes")
    whole = torch.from_numpy(arrays["mu"])
    cached_proxy = torch.from_numpy(arrays["proxy_ball"])
    proxy_state = {key.removeprefix("module."): value for key, value in saved["proxy"].items()}
    tangent = proxy_state["tangent_proxies"].detach().cpu().float().clone()
    if tangent.shape != (512, 256):
        raise ValueError("Unexpected tangent proxy shape")
    mapped = expmap0_c1(tangent)
    torch.testing.assert_close(mapped, cached_proxy, atol=2e-6, rtol=1e-5)
    # Validate all cached whole/proxy rows, not only sampled plans.
    for name, points in (("whole", whole), ("proxy", mapped)):
        if not bool(torch.isfinite(points).all()) or not bool((points.square().sum(-1) < 1).all()):
            raise ValueError(f"Invalid {name} ball vectors")
    return whole, torch.from_numpy(labels), torch.from_numpy(ids), tangent, saved, {
        "hashes": hashes, "epoch": int(saved["epoch"]), "training_commit": saved.get("commit"),
        "split_sha256": saved["split_sha256"], "ids_sha256": sha_array(ids),
        "labels_sha256": sha_array(labels), "training_shards_sorted": shards,
        "proxy_cache_mapping_max_abs_difference": float((mapped - cached_proxy).abs().max()),
        "feature_input": data_meta.get("feature_input"), "source_training_config": config,
    }


def make_plans(labels, ids, seed, count, plan_epoch):
    sampler = SourceClassBatchSampler(labels.tolist(), seed=seed, steps=max(count, 1), rank=0)
    sampler.set_epoch(plan_epoch)
    plans = []
    for plan in sampler.iter_global_plans():
        rows = torch.tensor(plan.indices, dtype=torch.long)
        if len(rows) != 64 or len(set(plan.classes)) != 32:
            raise AssertionError("Plan must have 32 distinct classes x2")
        plans.append({"rows": rows, "metadata": {"epoch": plan.epoch, "step": plan.step,
            "classes": list(plan.classes), "labels": list(plan.labels),
            "sample_ids": ids[rows].tolist(), "rank0_ids": ids[rows[:32]].tolist(),
            "rank1_ids": ids[rows[32:]].tolist(), "duplicate_id_positions": 64 - len(set(ids[rows].tolist()))}})
    return plans, sampler.state_dict()


def mine_for_mode(whole, labels, ids, proxies, mode, seed, mining_radius):
    mining_whole = whole.detach()
    if mode == "sample_mining_equal_radius":
        norms = mining_whole.norm(dim=-1, keepdim=True)
        if bool((norms == 0).any()):
            raise ValueError("Equal-radius mining undefined for zero whole direction")
        mining_whole = mining_radius * mining_whole / norms
    sample = mine_sample_triplets(mining_whole, labels, topk=20, t_per_anchor=50,
        seed=seed, exclude_self_negative=True, data_ids=ids)
    proxy = mine_reciprocal_triplets(torch.exp(-poincare_distance(proxies.detach())), 20,
        50, seed + 1, True)
    if sample["stats"]["self_k_triplets"] or proxy["stats"]["self_k_triplets"]:
        raise AssertionError("V6 self-k exclusion failed")
    return sample, proxy


def forward_terms(whole, tangent, labels, ids, mode, seed, mining_radius, proxy_rng_state=None):
    proxies = expmap0_c1(tangent)
    sample, proxy = mine_for_mode(whole, labels, ids, proxies, mode, seed, mining_radius)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_loss, sample_stats, sample_details = ghhc_loss(poincare_distance(whole, proxies),
        sample["triplets"], .1, .1, generator, return_details=True)
    baseline_proxy_rng = generator.get_state().clone()
    if proxy_rng_state is not None:
        # Hold proxy randomness fixed when changed sample triplet count would
        # otherwise advance a shared source generator by a different amount.
        generator.set_state(proxy_rng_state)
    proxy_loss, proxy_stats, proxy_details = ghhc_loss(poincare_distance(proxies),
        proxy["triplets"], .1, .1, generator, return_details=True)
    return {"whole": whole, "tangent": tangent, "proxies": proxies,
        "losses": {"sample": sample_loss, "proxy": proxy_loss, "combined": sample_loss + proxy_loss},
        "mining": {"sample": sample, "proxy": proxy},
        "stats": {"sample": sample_stats, "proxy": proxy_stats},
        "details": {"sample": sample_details, "proxy": proxy_details},
        "proxy_rng_state_after_original_sample": baseline_proxy_rng}


def endpoint_overlap(triplets, details):
    result = {}
    pair, triple = details["pair_proxy_idx"], details["triple_proxy_idx"]
    valid = pair != triple
    active = valid & (details["per_draw_loss"].detach() > 0)
    for domain, mask in (("all_draws", torch.ones_like(valid)), ("noncollision", valid),
                         ("active_noncollision", active)):
        pair_hit = (pair[:, None] == triplets[:, :2]).any(-1)
        triple_hit = (triple[:, None] == triplets).any(-1)
        result[domain] = {"draws": int(mask.sum()), "pair_endpoint_hits": int((pair_hit & mask).sum()),
                          "triple_endpoint_hits": int((triple_hit & mask).sum())}
    return result


class ProbeAccumulator:
    def __init__(self):
        self.structure = StructureAccumulator()
        self.gradients = defaultdict(dict)
        self.losses = defaultdict(dict)
        self.endpoint = defaultdict(lambda: defaultdict(int))
        self.draw_stats = defaultdict(lambda: defaultdict(int))
        self.mining_stats = defaultdict(lambda: defaultdict(int))
        self.proxy_gradient_sums = {}
        self.whole_gradient_by_id = {}
        self.gradient_conflict = defaultdict(list)
        self.steps = 0

    def update(self, terms, labels, is_low_radius, gradients, data_ids=None):
        self.steps += 1
        for component, loss in terms["losses"].items():
            add_moments(self.losses[component], moments([float(loss.detach())]))
        structure = {}
        for component in ("sample", "proxy"):
            mined, details = terms["mining"][component], terms["details"][component]
            points = terms["whole"] if component == "sample" else terms["proxies"]
            structure[component] = ancestor_summary(points, terms["proxies"], mined["triplets"], details)
            for key in ("triplets", "collisions", "noncollision_triplets", "active_triplets",
                        "active_self_k_triplets", "active_distinct_k_triplets"):
                self.draw_stats[component][key] += terms["stats"][component][key]
            for key in ("batch_size", "eligible_anchors", "mutual_positive_edges", "triplets", "self_k_triplets",
                        "same_class_j_triplets", "same_class_k_triplets", "all_same_class_triplets",
                        "equal_data_id_ik_triplets", "equal_data_id_ij_triplets"):
                if key in mined["stats"]:
                    self.mining_stats[component][key] += mined["stats"][key]
        self.structure.update(structure)
        for domain, row in endpoint_overlap(terms["mining"]["proxy"]["triplets"], terms["details"]["proxy"]).items():
            for key, value in row.items():
                self.endpoint[domain][key] += value
        for component, by_space in gradients.items():
            for space, gradient in by_space.items():
                points = terms[{"whole_ball": "whole", "proxy_ball": "proxies", "proxy_tangent": "tangent"}[space]]
                values = gradient_values(points, gradient)
                groups = {"all": torch.ones(len(points), dtype=torch.bool)}
                if space == "whole_ball":
                    groups.update({"bottom_radius_1pct": is_low_radius,
                                   "other_whole": ~is_low_radius})
                    groups.update({f"class/{NAMES[int(c)]}": labels == c for c in labels.unique().tolist()})
                else:
                    saturated = torch.tanh(terms["tangent"].detach().norm(dim=-1)) > .999
                    groups.update({"numerically_projected": saturated, "not_projected": ~saturated})
                    for name, value in values.items():
                        key = f"{component}/{space}/{name}"
                        self.proxy_gradient_sums.setdefault(key, torch.zeros_like(value, dtype=torch.float64))
                        self.proxy_gradient_sums[key] += value.double()
                for group, mask in groups.items():
                    for name, value in values.items():
                        add_moments(self.gradients[f"{component}/{space}/{group}/{name}"], moments(value[mask]))
                if component == "sample" and space == "whole_ball" and data_ids is not None:
                    for position, data_id in enumerate(data_ids.tolist()):
                        record = self.whole_gradient_by_id.setdefault(str(data_id), {
                            "label": int(labels[position]), "positions": 0,
                            "depth": float(depth(points[position:position + 1])[0]),
                            "is_bottom_radius_1pct": bool(is_low_radius[position]),
                            "signed_radial_gradient_sum": 0., "angular_gradient_norm_sum": 0.})
                        record["positions"] += 1
                        record["signed_radial_gradient_sum"] += float(values["signed_radial_gradient"][position])
                        record["angular_gradient_norm_sum"] += float(values["angular_gradient_norm"][position])
        for space in ("proxy_tangent", "proxy_ball"):
            self.gradient_conflict[space].append(gradient_cosine(gradients["sample"][space], gradients["proxy"][space]))

    def summary(self):
        endpoints = {}
        for domain, value in self.endpoint.items():
            endpoints[domain] = {**value,
                "pair_endpoint_hit_fraction": value["pair_endpoint_hits"] / value["draws"] if value["draws"] else None,
                "triple_endpoint_hit_fraction": value["triple_endpoint_hits"] / value["draws"] if value["draws"] else None}
        mining = {}
        for component, value in self.mining_stats.items():
            mining[component] = {**value, "pooled_anchor_coverage": value["eligible_anchors"] / value["batch_size"]}
            if component == "sample":
                mining[component]["same_class_j_draw_fraction"] = value["same_class_j_triplets"] / value["triplets"] if value["triplets"] else None
        whole_by_id = {key: {**value,
            "signed_radial_gradient_mean": value["signed_radial_gradient_sum"] / value["positions"],
            "angular_gradient_norm_mean": value["angular_gradient_norm_sum"] / value["positions"]}
            for key, value in self.whole_gradient_by_id.items()}
        return {"batches": self.steps, "loss_unweighted_batch_distributions": {k: finish_moments(v) for k, v in self.losses.items()},
            "draw_counts": dict(self.draw_stats), "actual_cpu_replay_ancestor_structure": self.structure.summary(),
            "pooled_mining_counts": mining,
            "proxy_ancestor_endpoint_identity_overlap": endpoints,
            "weighted_partial_gradients": {k: finish_moments(v) for k, v in self.gradients.items()},
            "per_proxy_partial_gradient_mean_over_batches": {key: (value / self.steps).tolist() for key, value in self.proxy_gradient_sums.items()},
            "observed_whole_sample_partial_gradient_by_data_id": whole_by_id,
            "sample_proxy_gradient_cosines_by_batch": dict(self.gradient_conflict)}


def component_gradients(terms, weight):
    variables = (terms["tangent"], terms["proxies"], terms["whole"])
    result = {}
    for component in COMPONENTS:
        gradients = torch.autograd.grad(weight * terms["losses"][component], variables,
            allow_unused=True, retain_graph=True)
        result[component] = {name: torch.zeros_like(var) if grad is None else grad.detach()
            for name, var, grad in zip(("proxy_tangent", "proxy_ball", "whole_ball"), variables, gradients)}
    for space in result["combined"]:
        torch.testing.assert_close(result["combined"][space], result["sample"][space] + result["proxy"][space], atol=2e-5, rtol=2e-4)
    if bool((result["proxy"]["whole_ball"] != 0).any()):
        raise AssertionError("Rproxy must have no direct whole gradient")
    return result


def describe_geometry(whole, tangent):
    proxies = expmap0_c1(tangent)
    def q(value):
        return torch.quantile(value.detach().double(), torch.tensor([0., .01, .1, .5, .9, .99, 1.], dtype=torch.float64)).tolist()
    return {"whole_depth_quantiles": q(depth(whole)), "proxy_depth_quantiles": q(depth(proxies)),
            "proxy_tangent_norm_quantiles": q(tangent.norm(dim=-1)),
            "proxy_preprojection_radius_quantiles": q(tangent.norm(dim=-1).tanh()),
            "projected_proxy_count": int((tangent.norm(dim=-1).tanh() > .999).sum())}


def make_proxy_optimizer(parameter, saved, args, state_kind):
    """Match checkpoint hyperparameters even when discarding historical m/v."""
    optimizer = torch.optim.AdamW([parameter], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    if "proxy_optimizer" not in saved:
        raise ValueError("Optional optimizer arms require saved proxy_optimizer hyperparameters/state")
    optimizer.load_state_dict(copy.deepcopy(saved["proxy_optimizer"]))
    if state_kind == "fresh":
        optimizer.state.clear()
    if args.proxy_lr is not None:
        optimizer.param_groups[0]["lr"] = args.proxy_lr
    return optimizer


@torch.no_grad()
def evaluate_proxy_structure(args, whole, labels, ids, tangent, plans):
    """Common original-mining no-update endpoint for every optional arm."""
    structure = StructureAccumulator()
    losses = defaultdict(dict)
    endpoints = defaultdict(lambda: defaultdict(int))
    draw_counts = defaultdict(lambda: defaultdict(int))
    for step, plan in enumerate(plans):
        at = plan["rows"]
        seed = args.seed + args.plan_epoch * 10000019 + step * 10007
        terms = forward_terms(whole[at], tangent, labels[at], ids[at], "original", seed, args.mining_radius)
        reports = {}
        for component in ("sample", "proxy"):
            points = terms["whole"] if component == "sample" else terms["proxies"]
            reports[component] = ancestor_summary(points, terms["proxies"],
                terms["mining"][component]["triplets"], terms["details"][component])
            for key in ("triplets", "collisions", "noncollision_triplets", "active_triplets"):
                draw_counts[component][key] += terms["stats"][component][key]
        structure.update(reports)
        for component, loss in terms["losses"].items():
            add_moments(losses[component], moments([float(loss)]))
        for domain, record in endpoint_overlap(terms["mining"]["proxy"]["triplets"], terms["details"]["proxy"]).items():
            for key, value in record.items():
                endpoints[domain][key] += value
        del terms
    result_endpoints = {}
    for domain, value in endpoints.items():
        result_endpoints[domain] = {**value,
            "pair_endpoint_hit_fraction": value["pair_endpoint_hits"] / value["draws"] if value["draws"] else None,
            "triple_endpoint_hit_fraction": value["triple_endpoint_hits"] / value["draws"] if value["draws"] else None}
    return {"optimizer_updates": 0, "evaluation_mining": "original",
        "evaluation_plan_and_seed_rule": "exact same plans and seed + plan_epoch*10000019 + plan_step*10007 as default probe",
        "batches": len(plans), "draw_counts": dict(draw_counts),
        "loss_unweighted_batch_distributions": {key: finish_moments(value) for key, value in losses.items()},
        "actual_cpu_replay_ancestor_structure": structure.summary(),
        "proxy_ancestor_endpoint_identity_overlap": result_endpoints,
        "geometry": describe_geometry(whole, tangent)}


def optimizer_arms(args, whole, labels, ids, tangent, saved, plans):
    results = {}
    whole_sha_before, tangent_sha_before = sha_tensor(whole), sha_tensor(tangent)
    for mode in MODES:
        results[mode] = {}
        for component in COMPONENTS:
            parameter = torch.nn.Parameter(tangent.clone())
            optimizer = make_proxy_optimizer(parameter, saved, args, args.optimizer_state)
            rows = []
            for step in range(args.proxy_steps):
                plan = plans[step % len(plans)]
                at = plan["rows"]
                seed = args.seed + 6000001 + step * 10007
                proxy_rng = None
                if mode != "original":
                    # Obtain the raw-mode proxy stream at the same evolving
                    # proxy state, holding only the mining intervention apart.
                    with torch.no_grad():
                        reference = forward_terms(whole[at], parameter, labels[at], ids[at], "original", seed, args.mining_radius)
                    proxy_rng = reference["proxy_rng_state_after_original_sample"]
                    del reference
                terms = forward_terms(whole[at], parameter, labels[at], ids[at], mode, seed, args.mining_radius, proxy_rng)
                optimizer.zero_grad(set_to_none=True)
                loss = args.weight * terms["losses"][component]
                loss.backward()
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Nonfinite optional proxy optimizer gradient")
                before = parameter.detach().clone()
                optimizer.step()
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError("Nonfinite optional proxy optimizer state")
                if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.proxy_steps:
                    rows.append({"step": step + 1, "weighted_loss": float(loss.detach()),
                                 "geometry": describe_geometry(whole, parameter.detach()),
                                 "actual_optimizer_update": proxy_update_stats(before, parameter.detach())})
                del terms, loss
            results[mode][component] = {"steps": args.proxy_steps, "optimizer_initial_state": args.optimizer_state,
                "optimizer_param_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
                "history_every10_steps": rows, "final_tangent_sha256": sha_tensor(parameter),
                "total_displacement": proxy_update_stats(tangent, parameter.detach()),
                "common_original_mining_terminal_evaluation": evaluate_proxy_structure(args, whole, labels, ids, parameter.detach(), plans),
                "interpretation": "Frozen-whole response with fixed LR and AdamW decay; checkpoint m/v include the historical combined objective. This is not a pure instantaneous component-gradient effect."}
            np.savez_compressed(args.output_dir / f"proxy_only_{mode}_{component}.npz",
                tangent_proxies=parameter.detach().numpy(), proxy_ball=expmap0_c1(parameter).detach().numpy())
            print(f"Optional frozen-whole AdamW arm complete: {mode}/{component}", flush=True)
    original_proxy_hash = results["original"]["proxy"]["final_tangent_sha256"]
    controlled_proxy_hash = results["sample_mining_equal_radius"]["proxy"]["final_tangent_sha256"]
    if original_proxy_hash != controlled_proxy_hash:
        raise RuntimeError("Proxy-only raw/equal mining controls must produce bit-identical tangent parameters; check random-stream isolation")
    results["checks"] = {"proxy_only_raw_equal_tangent_sha256_identical": True}
    # Explicit zero gradients permit AdamW momentum and weight decay updates;
    # grad=None would skip the parameter and would not provide this control.
    controls = {}
    for name, state_kind in (("checkpoint_zero_gradient", "checkpoint"), ("fresh_zero_gradient_decay_only", "fresh")):
        parameter = torch.nn.Parameter(tangent.clone())
        optimizer = make_proxy_optimizer(parameter, saved, args, state_kind)
        rows = []
        for step in range(args.proxy_steps):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.zeros_like(parameter)
            before = parameter.detach().clone()
            optimizer.step()
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError("Nonfinite zero-gradient AdamW control")
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.proxy_steps:
                rows.append({"step": step + 1, "actual_optimizer_update": proxy_update_stats(before, parameter.detach())})
        controls[name] = {"steps": args.proxy_steps, "loss_weight": 0., "gradient_policy": "explicit all-zero tensor",
            "optimizer_initial_state": state_kind,
            "optimizer_param_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
            "history_every10_steps": rows, "final_tangent_sha256": sha_tensor(parameter),
            "total_displacement": proxy_update_stats(tangent, parameter.detach()),
            "common_original_mining_terminal_evaluation": evaluate_proxy_structure(args, whole, labels, ids, parameter.detach(), plans),
            "interpretation": "Historical m/v plus decay" if state_kind == "checkpoint" else "Decay only with checkpoint-matched hyperparameters and zero m/v"}
        np.savez_compressed(args.output_dir / f"proxy_only_control_{name}.npz",
            tangent_proxies=parameter.detach().numpy(), proxy_ball=expmap0_c1(parameter).detach().numpy())
        print(f"Optional zero-gradient AdamW control complete: {name}", flush=True)
    results["zero_gradient_controls"] = controls
    if sha_tensor(whole) != whole_sha_before or sha_tensor(tangent) != tangent_sha_before:
        raise RuntimeError("Optional optimizer arms modified source whole/initial tangent tensors")
    results["checks"]["whole_and_initial_tangent_tensors_unchanged"] = True
    results["total_optimizer_updates"] = args.proxy_steps * (len(MODES) * len(COMPONENTS) + len(controls))
    results["protocol"] = {"loss_weight": args.weight, "warmup_steps": 0,
        "LR_policy": "held at checkpoint next LR unless explicitly overridden; no scheduler steps",
        "primary_arm_initial_momentum": args.optimizer_state,
        "fresh_policy": "discard m/v while retaining the checkpoint LR/betas/eps/decay",
        "endpoint_evaluation": "same original-mining fixed plans/seeds for all six objective arms and both zero-gradient controls"}
    return results


def self_test():
    valid_protocol = {"format": CHECKPOINT_FORMAT, "model_selection_only": False,
        "epoch": 300, "completed_epochs": 300, "training_config": {**SOURCE_PROTOCOL,
            "self_negative": False, "extra_HIER_tangent_cap": False, "HIER_backward_hook": False, "smoke": False}}
    validate_checkpoint_protocol(valid_protocol)
    for key, wrong_value in (("sample_K", 10), ("proxy_K", 10), ("T", 5), ("tau", .2),
                             ("margin", .2), ("self_negative", True)):
        wrong = copy.deepcopy(valid_protocol)
        wrong["training_config"][key] = wrong_value
        try:
            validate_checkpoint_protocol(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Protocol whitelist accepted incorrect {key}")
    for key, wrong_value in (("format", "legacy"), ("completed_epochs", 299)):
        wrong = copy.deepcopy(valid_protocol)
        wrong[key] = wrong_value
        try:
            validate_checkpoint_protocol(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Identity whitelist accepted incorrect {key}")
    generator = torch.Generator().manual_seed(314)
    labels = torch.arange(40).repeat_interleave(3)
    ids = torch.arange(len(labels))
    whole = expmap0_c1(torch.randn(len(labels), 8, generator=generator) * .15)
    tangent = torch.randn(32, 8, generator=generator) * .3
    plans, _ = make_plans(labels, ids, 22, 2, 0)
    at = plans[0]["rows"]
    model = HIERLoss(num_proxies=32, dim=8, seed=22)
    with torch.no_grad():
        model.tangent_proxies.copy_(tangent)
    terms = forward_terms(whole[at].clone().requires_grad_(), tangent.clone().requires_grad_(), labels[at], ids[at], "original", 123, .98)
    source_loss, source_stats = model(whole[at], labels[at], topk=20, seed=123,
        data_ids=ids[at], generator=torch.Generator().manual_seed(123), return_components=True)
    torch.testing.assert_close(terms["losses"]["combined"], source_loss, atol=0, rtol=0)
    for component in ("sample", "proxy"):
        torch.testing.assert_close(terms["details"][component]["pair_proxy_idx"], source_stats["_components"][f"{component}_details"]["pair_proxy_idx"], atol=0, rtol=0)
    gradients = component_gradients(terms, .5)
    accumulator = ProbeAccumulator()
    accumulator.update(terms, labels[at], torch.zeros(64, dtype=torch.bool), gradients)
    accumulator.summary()
    controlled = forward_terms(whole[at].clone().requires_grad_(), tangent.clone().requires_grad_(), labels[at], ids[at],
        "sample_mining_equal_radius", 123, .98, terms["proxy_rng_state_after_original_sample"])
    torch.testing.assert_close(controlled["losses"]["proxy"], terms["losses"]["proxy"], atol=0, rtol=0)
    torch.testing.assert_close(controlled["details"]["proxy"]["pair_proxy_idx"], terms["details"]["proxy"]["pair_proxy_idx"], atol=0, rtol=0)
    # Direct endpoint identity test: ancestor index and endpoint position use
    # the same namespace only for proxy triplets.
    overlap = endpoint_overlap(torch.tensor([[0, 1, 2]]), {"pair_proxy_idx": torch.tensor([0]),
        "triple_proxy_idx": torch.tensor([2]), "per_draw_loss": torch.tensor([1.])})
    assert overlap["noncollision"]["pair_endpoint_hits"] == 1
    assert overlap["noncollision"]["triple_endpoint_hits"] == 1
    # A small optimizer test verifies the zero-gradient control and matching
    # LR when fresh moments are requested, without executing optional100.
    historical_parameter = torch.nn.Parameter(tangent.clone())
    historical_optimizer = torch.optim.AdamW([historical_parameter], lr=.003, weight_decay=.01)
    historical_parameter.grad = torch.randn(tangent.shape, generator=generator)
    historical_optimizer.step()
    optimizer_saved = {"proxy_optimizer": copy.deepcopy(historical_optimizer.state_dict())}
    initial_parameter = historical_parameter.detach().clone()
    test_args = argparse.Namespace(proxy_lr=None, seed=22, plan_epoch=0, mining_radius=.98)
    zero_endpoints = []
    for state_kind in ("checkpoint", "fresh"):
        parameter = torch.nn.Parameter(initial_parameter.clone())
        optimizer = make_proxy_optimizer(parameter, optimizer_saved, test_args, state_kind)
        assert optimizer.param_groups[0]["lr"] == .003
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        zero_endpoints.append(parameter.detach().clone())
    assert not torch.equal(zero_endpoints[0], zero_endpoints[1])
    assert not torch.equal(zero_endpoints[1], initial_parameter)
    proxy_only_end_hashes = []
    for mode in MODES:
        parameter = torch.nn.Parameter(initial_parameter.clone())
        optimizer = make_proxy_optimizer(parameter, optimizer_saved, test_args, "checkpoint")
        for step in range(2):
            at = plans[step]["rows"]
            seed = 22 + 6000001 + step * 10007
            reference_rng = None
            if mode != "original":
                with torch.no_grad():
                    reference = forward_terms(whole[at], parameter, labels[at], ids[at], "original", seed, .98)
                reference_rng = reference["proxy_rng_state_after_original_sample"]
                del reference
            current = forward_terms(whole[at], parameter, labels[at], ids[at], mode, seed, .98, reference_rng)
            optimizer.zero_grad(set_to_none=True)
            (.5 * current["losses"]["proxy"]).backward()
            optimizer.step()
            del current
        proxy_only_end_hashes.append(sha_tensor(parameter))
        terminal = evaluate_proxy_structure(test_args, whole, labels, ids, parameter.detach(), plans)
        assert terminal["optimizer_updates"] == 0 and terminal["batches"] == 2
    assert proxy_only_end_hashes[0] == proxy_only_end_hashes[1]
    print(json.dumps({"self_test": "passed", "checks": ["protocol/format/epoch whitelist rejects incorrect identities", "balanced64 plan", "source exact loss/selection replay",
        "component gradient additivity", "no direct Rproxy whole gradient", "fixed proxy Gumbel control", "endpoint identity",
        "zero-gradient momentum versus matched-LR decay control", "proxy-only two-step hash invariance", "common terminal no-update structure evaluation"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cache-metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--plan-epoch", type=int, default=0, help="Same explicit sampler epoch across checkpoint probes")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mining-radius", type=float, default=.98)
    parser.add_argument("--weight", type=float, default=.5)
    parser.add_argument("--save-selections", action="store_true", help="Private full triplet/ancestor replay arrays")
    parser.add_argument("--proxy-steps", type=int, choices=(0, 100), default=0)
    parser.add_argument("--optimizer-state", choices=("checkpoint", "fresh"), default="checkpoint")
    parser.add_argument("--proxy-lr", type=float, default=None, help="Optional explicit LR override, separately recorded")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.batches <= 64 or not 0 < args.mining_radius < 1 or args.weight <= 0:
        parser.error("Require positive threads/weight, batches1..64 and mining radius in(0,1)")
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    if any(value is None for value in (args.cache, args.checkpoint, args.output_dir)):
        parser.error("--cache, --checkpoint and --output-dir required")
    args.cache_metadata = args.cache_metadata or args.cache.with_name("summary.json")
    if args.output_dir.exists():
        raise FileExistsError("Output must be a new directory; existing results are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "format": "hier-proxy-objective-probe-v1",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    json_write(args.output_dir / "summary.json", report)
    try:
        whole, labels, ids, tangent, saved, identity = validate_inputs(args.cache, args.checkpoint, args.cache_metadata)
        input_hashes = identity["hashes"]
        plans, sampler_identity = make_plans(labels, ids, args.seed, args.batches, args.plan_epoch)
        json_write(args.output_dir / "batch_plans.json", {"sampler": sampler_identity, "plans": [p["metadata"] for p in plans]})
        plan_sha = sha_file(args.output_dir / "batch_plans.json")
        lowest = torch.zeros(len(whole), dtype=torch.bool)
        lowest[torch.argsort(whole.norm(dim=-1), stable=True)[:int(np.ceil(.01 * len(whole)))]] = True
        accumulators = {mode: ProbeAccumulator() for mode in MODES}
        mining_controls = []
        selections = {}
        for step, plan in enumerate(plans):
            at = plan["rows"]
            seed = args.seed + args.plan_epoch * 10000019 + step * 10007
            original_rng, original_mutual, original_proxy_choice = None, None, None
            for mode in MODES:
                terms = forward_terms(whole[at].clone().requires_grad_(), tangent.clone().requires_grad_(),
                    labels[at], ids[at], mode, seed, args.mining_radius, original_rng)
                gradients = component_gradients(terms, args.weight)
                accumulators[mode].update(terms, labels[at], lowest[at], gradients, ids[at])
                mutual = terms["mining"]["sample"]["mutual"]
                if mode == "original":
                    original_rng = terms["proxy_rng_state_after_original_sample"]
                    original_mutual = mutual
                    original_proxy_choice = terms["details"]["proxy"]["pair_proxy_idx"].clone()
                else:
                    torch.testing.assert_close(terms["details"]["proxy"]["pair_proxy_idx"], original_proxy_choice, atol=0, rtol=0)
                    union = int((mutual | original_mutual).sum())
                    mining_controls.append({"batch": step, "sample_mutual_intersection": int((mutual & original_mutual).sum()),
                        "sample_mutual_union": union, "sample_mutual_jaccard": int((mutual & original_mutual).sum()) / union if union else None,
                        "controlled_sample_draws": len(terms["mining"]["sample"]["triplets"]), "proxy_randomness_held_fixed": True})
                if args.save_selections:
                    for component in ("sample", "proxy"):
                        key = f"batch{step:03d}_{mode}_{component}"
                        selections[key + "_triplets"] = terms["mining"][component]["triplets"].numpy()
                        selections[key + "_pair"] = terms["details"][component]["pair_proxy_idx"].numpy()
                        selections[key + "_triple"] = terms["details"][component]["triple_proxy_idx"].numpy()
                        selections[key + "_loss"] = terms["details"][component]["per_draw_loss"].detach().numpy()
                del terms, gradients
            print(f"Frozen cache CPU probe: {step + 1}/{len(plans)} paired batches", flush=True)
        if args.save_selections:
            np.savez_compressed(args.output_dir / "cpu_replay_selections.npz", **selections)
        report.update(identity=identity, sampler_identity=sampler_identity, batch_plan_sha256=plan_sha,
            geometry=describe_geometry(whole, tangent), configuration={
                "seed": args.seed, "batches": args.batches, "plan_epoch": args.plan_epoch, "K": 20, "T": 50,
                "c": 1, "dtype": "float32", "margin": .1, "tau": .1, "loss_weight": args.weight,
                "mining_radius": args.mining_radius, "proxy_steps": args.proxy_steps, "torch_threads": args.threads},
            modes={mode: accumulator.summary() for mode, accumulator in accumulators.items()},
            mining_controls=mining_controls,
            limitations=["Clean eval cache replay; no augmentation, BN forward, backbone gradient or original training Gumbel replay",
                "Source FP32 CPU functions; topK ties and random kernels can differ from historical CUDA",
                "Equal radius affects sample mining only; original whole/proxy geometry remains in both losses",
                "Raw mode uses production shared generator sequence; controlled proxy generator resets to raw sample post-state",
                "Proxy endpoint identity overlap is meaningful; whole object IDs and proxy IDs have different namespaces",
                "Positive signed radial partial gradient points outward; gradient descent subtracts it; AdamW displacement may differ",
                "Component gradients contain loss weight and are Euclidean partials, not shared-parameter optimizer effects",
                "Optional checkpoint AdamW state contains historical combined-objective first/second moments; continuation arms are not pure component-gradient effects",
                "No independent morphology labels or geometry truth used"],
            runtime={"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
            checks={"cache_checkpoint_identity": True, "all_ID_label_hashes": True, "proxy_mapping_matches_cache": True,
                "source_self_k_exclusion": True, "gradient_additivity_and_finite": True,
                "Rproxy_no_direct_whole_gradient": True, "controlled_proxy_pair_choices_identical": True},
            optimizer_updates=0)
        if args.proxy_steps:
            report["optional_proxy_optimizer_arms"] = optimizer_arms(args, whole, labels, ids, tangent, saved, plans)
            report["optimizer_updates"] = report["optional_proxy_optimizer_arms"]["total_optimizer_updates"]
        for key, path in (("cache", args.cache), ("checkpoint", args.checkpoint), ("cache_metadata", args.cache_metadata)):
            if sha_file(path) != input_hashes[key]:
                raise RuntimeError(f"Source {key} changed during diagnostic")
        try:
            report["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            report["code_commit"] = None
        report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - started, source_files_unchanged=True,
                      diagnostic_script_sha256=sha_file(__file__))
        json_write(args.output_dir / "summary.json", report)
        print(json.dumps({"status": "complete", "epoch": identity["epoch"], "batches": args.batches,
                          "optimizer_updates": report["optimizer_updates"], "wall_seconds": report["wall_seconds"]}), flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - started)
        json_write(args.output_dir / "summary.json", report)
        raise


if __name__ == "__main__":
    main()
