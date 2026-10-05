"""Read-only single-GPU HyCoRe/HIER whole-gradient and BN diagnosis.

Global64 is assembled from two sequential local32 forwards. Every group
starts with the checkpoint's rank0 buffers, emulating the outer DDP buffer
broadcast; part then whole update ordinary BN twice. Low-memory VJPs replay
one local32 activation graph at a time. No optimizer or parameter update.

The saved derivatives are Euclidean coordinate partials, before optimizer
preconditioning, clipping and momentum. They are not measured training moves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "classification_ModelNet40", ROOT / "inter_hierarchy_MN40"):
    sys.path.insert(0, str(_path))

NAMES = "airplane bathtub bed bench bookshelf bottle bowl car chair cone cup curtain desk door dresser flower_pot glass_box guitar keyboard lamp laptop mantel monitor night_stand person piano plant radio range_hood sink sofa stairs stool table tent toilet tv_stand vase wardrobe xbox".split()
COMPONENTS = ("ce", "weighted_contrastive", "weighted_radial", "weighted_sample_st", "weighted_sample_fixed")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def module_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(array_hash(value).encode())
    return h.hexdigest()


def save_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_selected(seed, device):
    """Do not initialize CUDA contexts on GPUs running the main B0 job."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(int(seed))


def capture_selected_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_selected_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def snapshot_buffers(module):
    return {name: value.detach().clone() for name, value in module.named_buffers()}


@torch.no_grad()
def restore_buffers(module, saved):
    current = dict(module.named_buffers())
    if current.keys() != saved.keys():
        raise RuntimeError("Buffer schema changed")
    for name, value in saved.items():
        current[name].copy_(value)


def set_condition(module, condition):
    # These conditions isolate BN, keeping all other stochastic modules in
    # eval. The optional source_train condition instead enables full train.
    module.eval()
    if condition == "source_train":
        module.train()
    elif condition in ("train_bn", "train_bn_regrouped"):
        for item in module.modules():
            if isinstance(item, nn.modules.batchnorm._BatchNorm):
                item.train()
    elif condition != "eval":
        raise ValueError("Unknown condition " + condition)


def condition_groups(condition):
    if condition == "train_bn_regrouped":
        # Retain original global positions and negative mapping, while moving
        # one instance from every class to each local32 group.
        return [torch.arange(0, 64, 2), torch.arange(1, 64, 2)]
    return [torch.arange(0, 32), torch.arange(32, 64)]


def forward_fixed(backbone, cloud, whole_count, child_count, whole_centers, child_centers):
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.base_protocol import get_children_alias

    source = cloud.transpose(1, 2).contiguous()
    _, whole, _ = get_children_alias(source, whole_count, centers=whole_centers)
    before = whole.detach().clone()
    _, child, _ = get_children_alias(whole, child_count, centers=child_centers)
    if child.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr():
        raise RuntimeError("Part/whole alias lost")
    if whole.untyped_storage().data_ptr() != source.untyped_storage().data_ptr():
        raise RuntimeError("Whole/source alias lost")
    if torch.equal(whole, before):
        raise RuntimeError("Part did not overwrite the whole prefix")
    nu, _ = backbone(child, emb=True)
    mu, logits = backbone(whole)
    return {"mu": mu, "nu": nu, "logits": logits}


def fixed_ancestor_loss(distances, triplets, pair_idx, triple_idx, margin=.1):
    """Same selected IDs and scalar hinges, with selection derivatives off."""
    if not len(triplets):
        return distances.sum() * 0
    i, j, k = triplets.unbind(-1)
    valid = (pair_idx != triple_idx).to(distances.dtype)
    hi = F.relu(distances[i, pair_idx] - distances[i, triple_idx] + margin)
    hj = F.relu(distances[j, pair_idx] - distances[j, triple_idx] + margin)
    hk = F.relu(distances[k, triple_idx] - distances[k, pair_idx] + margin)
    return ((hi + hj + hk) * valid).mean()


def empty_like_grad(gradient, value):
    return torch.zeros_like(value) if gradient is None else gradient.detach()


def radial_partials(points, gradient):
    radius = points.norm(dim=-1)
    direction = points / radius[:, None].clamp_min(1e-15)
    radial = (gradient * direction).sum(-1)
    angular = gradient - radial[:, None] * direction
    return {"radial": radial, "angular_norm": angular.norm(dim=-1),
            "partial_wrt_depth": radial * (1 - radius.square()) / 2,
            "norm": gradient.norm(dim=-1)}


def comparison(a, b):
    na, nb = a.norm(), b.norm()
    return {"norm_a": float(na), "norm_b": float(nb),
            "norm_ratio": float(na / nb) if float(nb) else None,
            "cosine": float(torch.dot(a.flatten(), b.flatten()) / (na * nb)) if float(na * nb) else None}


def finite_tensor(value, description):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("Nonfinite " + description)


def leaf_components(outputs, labels, ids, proxy_points, relation_seed, classifier=None, fixed_draws=None):
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.base_protocol import intra_losses, smoothed_ce
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import mine_sample_triplets, poincare_distance
    from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss

    mu, nu, logits = (outputs[key].detach().clone().requires_grad_(True) for key in ("mu", "nu", "logits"))
    leaves = {"mu": mu, "nu": nu, "logits": logits}
    base = intra_losses(mu, nu, outputs["child_count"])
    mining = mine_sample_triplets(mu, labels, topk=20, t_per_anchor=50,
                                 seed=relation_seed, exclude_self_negative=True, data_ids=ids)
    triplets = mining["triplets"]
    distances = poincare_distance(mu, proxy_points.detach())
    gen = torch.Generator(device=mu.device).manual_seed(relation_seed)
    sample, sample_stats, details = ghhc_loss(distances, triplets, generator=gen, return_details=True)
    direct = fixed_ancestor_loss(distances, triplets, details["pair_proxy_idx"].detach(),
                                details["triple_proxy_idx"].detach())
    if not torch.allclose(sample.detach(), direct.detach(), atol=2e-6, rtol=2e-5):
        raise RuntimeError("Fixed-ancestor and straight-through scalar losses differ")
    objectives = {"ce": smoothed_ce(logits, labels),
                  "weighted_contrastive": .01 * base["intra_contrastive"],
                  "weighted_radial": .01 * base["intra_radial"],
                  "weighted_sample_st": .5 * sample, "weighted_sample_fixed": .5 * direct}
    gradients = {}
    for name, objective in objectives.items():
        values = torch.autograd.grad(objective, tuple(leaves.values()), allow_unused=True, retain_graph=True)
        gradients[name] = {key: empty_like_grad(value, leaves[key]) for key, value in zip(leaves, values)}
        for key, value in gradients[name].items():
            finite_tensor(value, name + "/" + key)
    # CE belongs to the logits output for replay. This additional mu partial
    # describes the actual Mobius classifier but is NEVER also fed to VJP.
    ce_mu = torch.zeros_like(mu)
    if classifier is not None:
        tied_logits = classifier(mu)
        if not torch.allclose(tied_logits.detach(), logits.detach(), atol=2e-5, rtol=1e-4):
            raise RuntimeError("Classifier(mu) does not reproduce whole logits")
        ce_mu = torch.autograd.grad(smoothed_ce(tied_logits, labels), mu)[0].detach()
        finite_tensor(ce_mu, "CE through classifier wrt mu")
    elif len(mu):
        # Tiny-network self tests do not use a tied classification branch.
        ce_mu = gradients["ce"]["mu"]
    detached_details = {key: value.detach() for key, value in details.items()}
    fixed_choices = None
    if fixed_draws is not None:
        fixed_gen = torch.Generator(device=mu.device).manual_seed(relation_seed)
        with torch.no_grad():
            _, _, selected = ghhc_loss(distances.detach(), fixed_draws, generator=fixed_gen, return_details=True)
        fixed_choices = {key: selected[key].detach() for key in ("pair_proxy_idx", "triple_proxy_idx")}
    return {"losses": {name: float(value.detach()) for name, value in objectives.items()},
            "leaf_gradients": gradients, "ce_mu": ce_mu, "mining": mining,
            "sample_stats": sample_stats, "details": detached_details,
            "fixed_draw_choices": fixed_choices}


def collect_global(backbone, cloud, centers, counts, condition, initial_buffers, group_seed, device):
    groups = condition_groups(condition)
    outputs, replay_states, bn_deltas = {}, [], []
    set_condition(backbone, condition)
    bn_layers = [item for item in backbone.modules() if isinstance(item, nn.modules.batchnorm._BatchNorm)]
    try:
        for group_number, group_cpu in enumerate(groups):
            restore_buffers(backbone, initial_buffers)
            seed_selected(group_seed + group_number * 1000003, device)
            replay_states.append(capture_selected_rng(device))
            before = [int(item.num_batches_tracked) for item in bn_layers]
            group = group_cpu.to(device)
            with torch.no_grad():
                local = forward_fixed(backbone, cloud[group].clone(), *counts,
                                      centers[0][group_cpu].tolist(), centers[1][group_cpu].tolist())
            for key, value in local.items():
                finite_tensor(value, condition + "/" + key)
                if key not in outputs:
                    outputs[key] = torch.empty((64, *value.shape[1:]), device=device, dtype=value.dtype)
                outputs[key][group] = value.detach()
            delta = [int(item.num_batches_tracked) - previous for item, previous in zip(bn_layers, before)]
            expected = 0 if condition == "eval" else 2
            if any(item != expected for item in delta):
                raise RuntimeError("BN update count differs from condition")
            bn_deltas.append({"min": min(delta), "max": max(delta), "expected": expected})
    finally:
        restore_buffers(backbone, initial_buffers)
    outputs["child_count"] = counts[1]
    return outputs, replay_states, bn_deltas


def replay_parameter_vjps(backbone, cloud, centers, counts, condition, initial_buffers,
                          replay_states, outputs, gradients, device):
    """Sum group VJPs with global-mean leaf gradients; no extra /2."""
    named = [(name, parameter) for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    parameters = [value for _, value in named]
    vectors = {component: None for component in COMPONENTS}
    replay_checks = []
    set_condition(backbone, condition)
    try:
        for group_number, group_cpu in enumerate(condition_groups(condition)):
            restore_buffers(backbone, initial_buffers)
            restore_selected_rng(replay_states[group_number], device)
            group = group_cpu.to(device)
            local = forward_fixed(backbone, cloud[group].clone(), *counts,
                                  centers[0][group_cpu].tolist(), centers[1][group_cpu].tolist())
            errors = {}
            for key, value in local.items():
                expected = outputs[key][group]
                errors[key] = float((value.detach() - expected).abs().max())
                if not torch.allclose(value.detach(), expected, atol=2e-5, rtol=1e-4):
                    raise RuntimeError("Replay mismatch: " + condition + "/" + key)
            replay_checks.append(errors)
            for index, component in enumerate(COMPONENTS):
                surrogate = sum((local[key] * gradients[component][key][group]).sum()
                                for key in ("mu", "nu", "logits"))
                values = torch.autograd.grad(surrogate, parameters, allow_unused=True,
                                            retain_graph=index < len(COMPONENTS) - 1)
                vector = torch.cat([torch.zeros_like(parameter).flatten() if value is None
                                    else value.detach().flatten() for parameter, value in zip(parameters, values)]).cpu()
                finite_tensor(vector, "parameter VJP " + component)
                if vectors[component] is None:
                    vectors[component] = vector
                else:
                    vectors[component].add_(vector)
            del local
    finally:
        restore_buffers(backbone, initial_buffers)
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError("Read-only VJP accumulated parameter .grad")
    return vectors, named, replay_checks


def parameter_summary(vectors, named):
    ranges = {key: [] for key in ("all_model", "shared_encoder", "euclidean_features", "mobius_embedding", "classifier")}
    start = 0
    for name, parameter in named:
        stop = start + parameter.numel()
        ranges["all_model"].append((start, stop))
        if name.startswith("classifier."):
            ranges["classifier"].append((start, stop))
        else:
            ranges["shared_encoder"].append((start, stop))
            ranges["mobius_embedding" if name.startswith("emb.") else "euclidean_features"].append((start, stop))
        start = stop
    report = {}
    for group, selected in ranges.items():
        if not selected:
            continue
        component = {key: torch.cat([value[a:b] for a, b in selected]) for key, value in vectors.items()}
        base = component["ce"] + component["weighted_contrastive"] + component["weighted_radial"]
        st, fixed = component["weighted_sample_st"], component["weighted_sample_fixed"]
        report[group] = {"elements": len(base), "component_norms": {key: float(value.norm()) for key, value in component.items()},
                         "base_norm": float(base.norm()), "combined_preclip_norm": float((base + st).norm()),
                         "sample_st_vs_base": comparison(st, base),
                         "sample_st_vs_ce": comparison(st, component["ce"]),
                         "sample_st_vs_intra": comparison(st, base - component["ce"]),
                         "sample_fixed_vs_base": comparison(fixed, base),
                         "selection_path_vs_direct": comparison(st - fixed, fixed),
                         "sample_st_vs_fixed": comparison(st, fixed)}
    return report


def geometry(outputs):
    mu, nu = outputs["mu"], outputs["nu"]
    wd = 2 * torch.atanh(mu.norm(dim=-1))
    pd = 2 * torch.atanh(nu.norm(dim=-1))
    return {"whole_depth_mean": float(wd.mean()), "part_depth_mean": float(pd.mean()),
            "whole_minus_part_mean": float((wd - pd).mean()),
            "whole_deeper_fraction": float((wd > pd).float().mean()),
            "radial_margin_violation_fraction": float((wd - pd < 1000. / outputs["child_count"]).float().mean())}


def relation_comparison(reference, current, reference_leaf, current_leaf):
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance

    old = reference_leaf["mining"]["mutual"]
    new = current_leaf["mining"]["mutual"]
    union = int((old | new).sum())
    fixed = current_leaf["fixed_draw_choices"]
    original = reference_leaf["details"]
    result = {"mutual_graph_jaccard": int((old & new).sum()) / union if union else None,
              "prediction_changed_fraction": float((reference["logits"].argmax(-1) != current["logits"].argmax(-1)).float().mean()),
              "whole_abs_depth_change_mean": float((2 * torch.atanh(reference["mu"].norm(dim=-1)) -
                                                      2 * torch.atanh(current["mu"].norm(dim=-1))).abs().mean()),
              "whole_hyperbolic_movement_mean": float(poincare_distance(reference["mu"], current["mu"]).diagonal().mean()),
              "negative_global_positions_unchanged": True,
              "fixed_draw_semantics": "Reference ordered triplets and identical fresh Gumbel noise; each condition may have a different natural reciprocal graph."}
    for role in ("pair", "triple"):
        key = role + "_proxy_idx"
        result["fixed_draw_" + role + "_ancestor_agreement"] = float((fixed[key] == original[key]).float().mean()) if len(original[key]) else None
    return result


def make_plan(sampler, step, epoch, train_ids, labels, hotspot_ids):
    plan = sampler.global_plan(step, epoch)
    indices = np.asarray(plan.indices, dtype=np.int64).copy()
    classes = list(plan.classes)
    modified = []
    if hotspot_ids:
        by_id = {int(identity): index for index, identity in enumerate(train_ids)}
        target_ids = [hotspot_ids[(2 * step + i) % len(hotspot_ids)] for i in range(2)]
        if any(identity not in by_id for identity in target_ids):
            raise ValueError("Hotspot ID outside checkpoint train split")
        target_class = int(labels[target_ids[0]])
        if any(int(labels[identity]) != target_class for identity in target_ids):
            raise ValueError("Hotspot panel must contain a single class")
        block = classes.index(target_class) if target_class in classes else 0
        modified = [{"block": block, "original_class": classes[block], "forced_class": target_class,
                     "original_ids": train_ids[indices[2 * block:2 * block + 2]].tolist(), "forced_ids": target_ids}]
        classes[block] = target_class
        indices[2 * block:2 * block + 2] = [by_id[identity] for identity in target_ids]
    ids = np.asarray(train_ids[indices], dtype=np.int64)
    gold = np.asarray(labels[ids], dtype=np.int64)
    if len(set(classes)) != 32 or any(np.count_nonzero(gold == label) != 2 for label in classes):
        raise RuntimeError("Balanced64 plan invalid")
    if np.any(gold == gold[::-1]):
        raise RuntimeError("Global flip negative shares the anchor class")
    return indices, ids, gold, modified


def prepare_inputs(points, ids, seed, epoch, step, input_mode, counts):
    cloud = points[ids, :1024].copy()
    if input_mode == "augmented":
        for position in range(64):
            rng = np.random.RandomState((seed + 31000003 * epoch + 10007 * step + 1009 * position) % (2 ** 32))
            cloud[position] = (cloud[position] * rng.uniform(2 / 3, 3 / 2, size=3) +
                               rng.uniform(-.2, .2, size=3)).astype(np.float32)
            rng.shuffle(cloud[position])
    whole_centers, child_centers = [], []
    for rank in range(2):
        rng = random.Random(seed + epoch * 10000019 + step * 1009 + rank * 1000003)
        whole_centers.extend(rng.randint(0, 1023) for _ in range(32))
        child_centers.extend(rng.randint(0, counts[0] - 1) for _ in range(32))
    return torch.from_numpy(cloud), (torch.tensor(whole_centers), torch.tensor(child_centers))


def whole_mu_components(leaf):
    mu_gradients = {name: value["mu"] for name, value in leaf["leaf_gradients"].items()}
    mu_gradients["ce"] = leaf["ce_mu"]
    mu_gradients["base"] = mu_gradients["ce"] + mu_gradients["weighted_contrastive"] + mu_gradients["weighted_radial"]
    mu_gradients["selection_path"] = mu_gradients["weighted_sample_st"] - mu_gradients["weighted_sample_fixed"]
    mu_gradients["base_plus_sample_st"] = mu_gradients["base"] + mu_gradients["weighted_sample_st"]
    return mu_gradients


def whole_partial_summary(outputs, leaf, labels, ids, hotspot_ids):
    masks = {"all": np.ones(64, dtype=bool)}
    for label in np.unique(labels):
        masks["class/" + NAMES[int(label)]] = labels == label
    if hotspot_ids:
        masks["hotspots"] = np.isin(ids, hotspot_ids)
    result = {}
    for component, gradient in whole_mu_components(leaf).items():
        partials = {key: value.detach().cpu().numpy() for key, value in radial_partials(outputs["mu"], gradient).items()}
        result[component] = {}
        for name, mask in masks.items():
            radial = partials["radial"][mask]
            nonzero = partials["norm"][mask] > 0
            result[component][name] = {"positions": int(mask.sum()), "nonzero_partial_fraction": float(nonzero.mean()) if len(radial) else None,
                "radial_mean": float(radial.mean()) if len(radial) else None,
                "radial_median": float(np.median(radial)) if len(radial) else None,
                "positive_radial_fraction": float((radial > 0).mean()) if len(radial) else None,
                "negative_radial_fraction": float((radial < 0).mean()) if len(radial) else None,
                "angular_norm_mean": float(partials["angular_norm"][mask].mean()) if len(radial) else None}
    return result


def write_rows(writer, batch, condition, outputs, leaf, ids, labels, hotspot_ids):
    mu_gradients = whole_mu_components(leaf)
    radius = outputs["mu"].norm(dim=-1).cpu().numpy()
    correct = (outputs["logits"].argmax(-1).cpu().numpy() == labels)
    for component, gradient in mu_gradients.items():
        partial = {key: value.detach().cpu().numpy() for key, value in radial_partials(outputs["mu"], gradient).items()}
        for position in range(64):
            writer.writerow({"batch": batch, "condition": condition, "position": position,
                             "id": int(ids[position]), "class": NAMES[int(labels[position])],
                             "negative_position": 63 - position, "negative_id": int(ids[63 - position]),
                             "hotspot": int(ids[position] in hotspot_ids), "correct": int(correct[position]),
                             "radius": float(radius[position]), "depth": float(2 * np.arctanh(radius[position])),
                             "component": component,
                             "radial_partial": float(partial["radial"][position]),
                             "angular_partial_norm": float(partial["angular_norm"][position]),
                             "partial_wrt_depth": float(partial["partial_wrt_depth"][position]),
                             "partial_norm": float(partial["norm"][position])})


def self_test():
    """CPU-only checks for chain rule, grouping and fixed-selection semantics."""
    torch.set_num_threads(2)
    torch.manual_seed(19)
    toy = nn.Linear(3, 2)
    inputs = torch.randn(64, 3)
    target = torch.randn(64, 2)
    full_loss = (toy(inputs) - target).square().mean()
    full = torch.autograd.grad(full_loss, tuple(toy.parameters()))
    leaf = toy(inputs).detach().requires_grad_(True)
    partial = torch.autograd.grad((leaf - target).square().mean(), leaf)[0]
    for groups in ([torch.arange(32), torch.arange(32, 64)], condition_groups("train_bn_regrouped")):
        total = [torch.zeros_like(parameter) for parameter in toy.parameters()]
        for group in groups:
            gradients = torch.autograd.grad((toy(inputs[group]) * partial[group]).sum(), tuple(toy.parameters()))
            for expected, gradient in zip(total, gradients):
                expected.add_(gradient)
        for expected, gradient in zip(full, total):
            torch.testing.assert_close(expected, gradient, atol=2e-6, rtol=1e-5)
    # Check ordinary local32 BN with distinct buffers and shared parameters.
    # This is the small CPU analogue of two DDP replicas, not global64 BN.
    import copy
    bn_toy = nn.Sequential(nn.Linear(3, 5), nn.BatchNorm1d(5), nn.Tanh(), nn.Linear(5, 2))
    replica = copy.deepcopy(bn_toy)
    for name, parameter in bn_toy.named_parameters():
        module_name, field = name.rsplit(".", 1)
        setattr(replica.get_submodule(module_name), field, parameter)
    initial = snapshot_buffers(bn_toy)
    for groups in ([torch.arange(32), torch.arange(32, 64)], condition_groups("train_bn_regrouped")):
        restore_buffers(bn_toy, initial)
        restore_buffers(replica, initial)
        first, second = bn_toy(inputs[groups[0]]), replica(inputs[groups[1]])
        index = torch.cat(groups)
        order = torch.argsort(index)
        global_output = torch.cat((first, second))[order]
        loss = (global_output - target).square().mean() + .03 * (global_output - global_output.flip(0)).square().mean()
        expected = torch.autograd.grad(loss, tuple(bn_toy.parameters()))
        frozen = global_output.detach().requires_grad_(True)
        local_loss = (frozen - target).square().mean() + .03 * (frozen - frozen.flip(0)).square().mean()
        partial = torch.autograd.grad(local_loss, frozen)[0]
        total = [torch.zeros_like(parameter) for parameter in bn_toy.parameters()]
        for group in groups:
            restore_buffers(bn_toy, initial)
            value = bn_toy(inputs[group])
            torch.testing.assert_close(value.detach(), frozen.detach()[group], atol=2e-6, rtol=1e-5)
            gradient = torch.autograd.grad((value * partial[group]).sum(), tuple(bn_toy.parameters()))
            for accumulated, contribution in zip(total, gradient):
                accumulated.add_(contribution)
        for direct, replay in zip(expected, total):
            torch.testing.assert_close(direct, replay, atol=2e-6, rtol=1e-5)
    distance = torch.tensor([[3., 2., 1.], [1.5, 2.5, 3.5], [4., 2., 1.]], requires_grad=True)
    triples = torch.tensor([[0, 1, 2], [0, 1, 2]])
    fixed = fixed_ancestor_loss(distance, triples, torch.tensor([0, 1]), torch.tensor([2, 1]))
    assert abs(float(fixed) - 1.05) < 1e-6
    fixed.backward()
    torch.testing.assert_close(distance.grad, torch.tensor([[.5, 0., -.5], [0., 0., 0.], [0., 0., 0.]]))
    point = torch.tensor([[.4, 0.]])
    partial = radial_partials(point, torch.tensor([[2., 3.]]))
    torch.testing.assert_close(partial["radial"], torch.tensor([2.]))
    torch.testing.assert_close(partial["angular_norm"], torch.tensor([3.]))
    torch.testing.assert_close(partial["partial_wrt_depth"], torch.tensor([.84]))
    assert all(parameter.grad is None for parameter in toy.parameters())
    assert all(parameter.grad is None for parameter in bn_toy.parameters())
    from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss
    generator = torch.Generator().manual_seed(53)
    costs = (torch.rand(3, 5, generator=generator) * 4 + .2).requires_grad_(True)
    draws = torch.tensor([[0, 1, 2]] * 64)
    actual, _, selected = ghhc_loss(costs, draws, tau=1.2, generator=torch.Generator().manual_seed(11), return_details=True)
    direct = fixed_ancestor_loss(costs, draws, selected["pair_proxy_idx"].detach(), selected["triple_proxy_idx"].detach())
    torch.testing.assert_close(actual.detach(), direct.detach(), atol=2e-6, rtol=1e-5)
    actual_gradient = torch.autograd.grad(actual, costs, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(direct, costs)[0]
    assert torch.isfinite(actual_gradient).all() and torch.isfinite(direct_gradient).all()
    assert float((actual_gradient - direct_gradient).norm()) > 1e-6
    print(json.dumps({"self_test": "passed", "checks": ["global mean two-group VJP", "regroup VJP", "local32 BN shared-parameter VJP", "cross-group objective chain rule", "fixed ancestor collision mask and derivative", "Gumbel scalar equivalence and selection gradient distinction", "radial/depth chain rule", "no parameter grad accumulation"]}))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="Logical index among CUDA_VISIBLE_DEVICES; caller must inspect physical idle GPU")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--plan-epoch", type=int, default=0)
    parser.add_argument("--input-mode", choices=("clean", "augmented"), default="augmented")
    parser.add_argument("--conditions", default="eval,train_bn,train_bn_regrouped")
    parser.add_argument("--hotspot-ids", default="", help="Optional conditional balanced64 panel, e.g. 2883,6009,7985,9123")
    parser.add_argument("--features-only", action="store_true", help="Skip activation replay/VJP while retaining leaf gradients and relation checks")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        if not all((args.checkpoint, args.data_dir, args.run_dir)):
            parser.error("checkpoint, data-dir and run-dir are required")
        if not 1 <= args.batches <= 32 or args.plan_epoch < 0:
            parser.error("batches must be 1..32 and plan-epoch nonnegative")
        args.conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
        if not args.conditions or len(set(args.conditions)) != len(args.conditions):
            parser.error("conditions must be nonempty and unique")
        if any(value not in ("eval", "train_bn", "train_bn_regrouped", "source_train") for value in args.conditions):
            parser.error("Unknown condition")
        args.hotspot_ids = [int(value) for value in args.hotspot_ids.split(",") if value.strip()]
        if args.hotspot_ids and (len(args.hotspot_ids) < 2 or len(set(args.hotspot_ids)) != len(args.hotspot_ids)):
            parser.error("hotspot-ids must contain at least two distinct IDs")
    return args


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.distributed.is_initialized():
        raise RuntimeError("This diagnostic must run as one ordinary process, without DDP")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "starting", "diagnostic_only": True, "optimizer_updates": 0,
                "script_sha256": digest(__file__), "started_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": digest(args.checkpoint),
                "checkpoint": str(args.checkpoint), "command": [sys.executable, *sys.argv],
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_gpu": args.gpu,
                "global_batch": 64, "local_forward_batch": 32, "conditions": args.conditions,
                "input_mode": args.input_mode, "seed": args.seed, "plan_epoch": args.plan_epoch,
                "hotspot_ids": args.hotspot_ids, "panel": "conditional hotspot balanced64" if args.hotspot_ids else "ordinary source balanced64 plans",
                "semantics": ["Fixed diagnostic inputs, centers, global positions and global-flip negative IDs across conditions.",
                              "Augmentation seeds are diagnostic occurrence seeds, not a claimed replay of historical DataLoader workers.",
                              "Each local32 group restores checkpoint rank0 buffers before part and whole; BN is never global64.",
                              "VJP sums local32 contributions to global-mean objectives with no additional world-size factor.",
                              "Leaf partials and parameter norms are before optimizer metric correction, clipping and momentum.",
                              "CE mu partial uses the tied Mobius classifier for interpretation; CE VJP uses only logits, avoiding double counting.",
                              "Rproxy has no direct dependence on whole, and is excluded from whole VJPs."]}
    try:
        manifest["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["commit"] = None
    save_json(args.run_dir / "manifest.json", manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for production pointMLP; self-test is CPU-only")
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
        from models.pointmlp import Hype_pointMLP
        from inter_hierarchy_MN40.hier_proxy_v2_online.train import load_shards
        from inter_hierarchy_MN40.hier_proxy_scratch_v3.train import freeze_curvature
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.distributed import shared_crop_counts
        from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import HIERLoss
        from inter_hierarchy_MN40.hier_proxy_scratch_v6.structure_monitor import ancestor_summary

        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "net" not in saved or "proxy" not in saved or "train_ids" not in saved:
            raise RuntimeError("Checkpoint must include matching H20 net, proxy and train IDs")
        seed_selected(args.seed, device)
        backbone = Hype_pointMLP().to(device)
        backbone.load_state_dict(saved["net"], strict=True)
        freeze_curvature(backbone)
        proxy = HIERLoss(seed=args.seed).to(device)
        proxy.load_state_dict(saved["proxy"], strict=True)
        with torch.no_grad():
            proxy_points = proxy.proxies().detach()
        train_ids = np.asarray(saved["train_ids"], dtype=np.int64)
        rank0_buffers = snapshot_buffers(backbone)
        # net in a complete H20 checkpoint is saved from rank0. If rank_states
        # exist, verify their rank0 copy before using it as the broadcast state.
        if saved.get("rank_states"):
            rank0_saved = saved["rank_states"][0]["BN_buffers"]
            for name, value in rank0_buffers.items():
                if name not in rank0_saved or not torch.equal(value.cpu(), rank0_saved[name].cpu()):
                    raise RuntimeError("Checkpoint net/rank0 buffer mismatch: " + name)
        expected_hash = module_hash(backbone)
        proxy_hash = module_hash(proxy)
        points, labels, shards = load_shards(args.data_dir, "train")
        if len(np.unique(train_ids)) != len(train_ids) or np.any(train_ids < 0) or np.any(train_ids >= len(labels)):
            raise RuntimeError("Invalid train ID mapping")
        shard_hashes = {name: digest(args.data_dir / name) for name in shards}
        sampler = SourceClassBatchSampler(labels[train_ids], seed=args.seed, steps=max(200, args.batches))
        dropout_modules = [{"name": name, "p": float(module.p)} for name, module in backbone.named_modules()
                           if isinstance(module, nn.modules.dropout._DropoutNd)]
        manifest.update(status="running", checkpoint_epoch=int(saved.get("completed_epochs", saved.get("epoch", -1))),
                        checkpoint_format=saved.get("format"), checkpoint_source_commit=saved.get("commit"),
                        split_sha256=saved.get("split_sha256"), train_ids_sha256=array_hash(train_ids),
                        input_shard_sha256=shard_hashes, labels_sha256=array_hash(labels),
                        sampler=sampler.state_dict(), backbone_initial_sha256=expected_hash,
                        proxy_initial_sha256=proxy_hash, dropout_modules=dropout_modules,
                        gpu_name=torch.cuda.get_device_name(device), torch_version=torch.__version__)
        save_json(args.run_dir / "manifest.json", manifest)
        del saved
        original_rng = capture_selected_rng(device)
        fieldnames = ["batch", "condition", "position", "id", "class", "negative_position", "negative_id",
                      "hotspot", "correct", "radius", "depth", "component", "radial_partial",
                      "angular_partial_norm", "partial_wrt_depth", "partial_norm"]
        all_reports = []
        with (args.run_dir / "whole_partials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(args.batches):
                indices, ids, gold, modified = make_plan(sampler, step, args.plan_epoch, train_ids, labels, args.hotspot_ids)
                counts = shared_crop_counts(args.seed, args.plan_epoch, step)
                cloud_cpu, centers = prepare_inputs(points, ids, args.seed, args.plan_epoch, step, args.input_mode, counts)
                cloud = cloud_cpu.to(device)
                targets, identities = torch.from_numpy(gold).to(device), torch.from_numpy(ids).to(device)
                group_seed = args.seed + args.plan_epoch * 10000019 + step * 1009
                relation_seed = args.seed + args.plan_epoch * 10000019 + step * 10007
                report = {"batch": step, "ids": ids.tolist(), "labels": gold.tolist(), "subset_indices": indices.tolist(),
                          "global_flip_negative_ids": ids[::-1].tolist(), "whole_points": counts[0], "part_points": counts[1],
                          "input_cloud_sha256": array_hash(cloud_cpu), "whole_centers": centers[0].tolist(),
                          "part_centers": centers[1].tolist(), "relation_seed": relation_seed,
                          "forced_panel_changes": modified, "conditions": {}}
                reference, reference_leaf = None, None
                for condition in args.conditions:
                    torch.cuda.reset_peak_memory_stats(device)
                    outputs, rng_states, bn_deltas = collect_global(backbone, cloud, centers, counts, condition,
                                                                  rank0_buffers, group_seed, device)
                    fixed_draws = None if reference_leaf is None else reference_leaf["mining"]["triplets"]
                    leaf = leaf_components(outputs, targets, identities, proxy_points, relation_seed,
                                           classifier=backbone.classifier, fixed_draws=fixed_draws)
                    condition_report = {"geometry": geometry(outputs), "losses": leaf["losses"],
                                        "mining": leaf["mining"]["stats"], "sample": leaf["sample_stats"],
                                        "whole_leaf_partials": whole_partial_summary(outputs, leaf, gold, ids, args.hotspot_ids),
                                        "BN_updates": bn_deltas,
                                        "ancestor": ancestor_summary(outputs["mu"], proxy_points, leaf["mining"]["triplets"], leaf["details"]),
                                        "non_BN_stochastic_modules_enabled": condition == "source_train" and bool(dropout_modules)}
                    if not args.features_only:
                        vectors, named, replay_checks = replay_parameter_vjps(backbone, cloud, centers, counts, condition,
                            rank0_buffers, rng_states, outputs, leaf["leaf_gradients"], device)
                        condition_report["parameter_groups"] = parameter_summary(vectors, named)
                        condition_report["replay_max_abs_errors"] = replay_checks
                        del vectors
                    write_rows(writer, step, condition, outputs, leaf, ids, gold, args.hotspot_ids)
                    arrays = {key: outputs[key].detach().cpu().numpy() for key in ("mu", "nu", "logits")}
                    arrays.update(ids=ids, labels=gold, triplets=leaf["mining"]["triplets"].cpu().numpy(),
                                  mutual=leaf["mining"]["mutual"].cpu().numpy(), ce_mu_partial=leaf["ce_mu"].cpu().numpy())
                    for key in ("pair_proxy_idx", "triple_proxy_idx"):
                        arrays[key] = leaf["details"][key].cpu().numpy()
                    for component, gradients in leaf["leaf_gradients"].items():
                        for key, value in gradients.items():
                            arrays[component + "_" + key + "_partial"] = value.cpu().numpy()
                    np.savez_compressed(args.run_dir / f"batch_{step:03d}_{condition}.npz", **arrays)
                    if reference is None:
                        reference, reference_leaf = outputs, leaf
                        condition_report["comparison_reference"] = condition
                    else:
                        condition_report["comparison_to_first_condition"] = relation_comparison(reference, outputs, reference_leaf, leaf)
                    condition_report["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    report["conditions"][condition] = condition_report
                    save_json(args.run_dir / f"batch_{step:03d}.json", report)
                    print(json.dumps({"batch": step, "condition": condition, "status": "passed",
                                      "losses": condition_report["losses"],
                                      "peak_allocated_MiB": condition_report["peak_allocated_MiB"]}), flush=True)
                stream.flush()
                all_reports.append(report)
                manifest["completed_batches"] = step + 1
                save_json(args.run_dir / "manifest.json", manifest)
                del cloud, reference, reference_leaf, outputs, leaf
        restore_buffers(backbone, rank0_buffers)
        restore_selected_rng(original_rng, device)
        if module_hash(backbone) != expected_hash or module_hash(proxy) != proxy_hash:
            raise RuntimeError("Diagnostic changed model/proxy state")
        if digest(args.checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during diagnosis")
        if {name: digest(args.data_dir / name) for name in shards} != shard_hashes:
            raise RuntimeError("Source HDF5 shards changed during diagnosis")
        if any(parameter.grad is not None for parameter in backbone.parameters()) or any(parameter.grad is not None for parameter in proxy.parameters()):
            raise RuntimeError("Diagnostic accumulated parameter gradients")
        summary = {"manifest": manifest, "batches": all_reports,
                   "sign_convention": "Positive radial_partial means a negative Euclidean mu-gradient step points inward; this is not an actual backbone/RiemannianSGD update.",
                   "parameter_gradient_semantics": "Sum of local32 VJPs for one global64 mean objective; preclip Euclidean parameter gradients; Rproxy not included.",
                   "state_checks": {"model_unchanged": True, "proxy_unchanged": True,
                                    "checkpoint_unchanged": True, "input_shards_unchanged": True,
                                    "BN_restored": True, "parameter_grad_buffers_empty": True}}
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter() - started, completed_batches=args.batches)
        save_json(args.run_dir / "summary.json", summary)
        save_json(args.run_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        save_json(args.run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
