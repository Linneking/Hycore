"""Bounded fixed-feature V5 K comparison; no model/proxy optimization.

All cached TRAINING cases receive graph diagnostics. Only an evenly spread
subset of view-0 cases receive live HIER gradients and losses. Paired views
must contain the same ordered sample IDs. Class/ID eligibility frequencies
are conditional on appearances in this cache, not dataset coverage rates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from .hier_loss import HIERLoss, ghhc_loss, shadow_geometry
from .relations import mine_reciprocal_triplets, poincare_distance


def distribution(values):
    """Finite descriptive statistics; empty samples are explicitly nullable."""
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite diagnostic values")
    if not len(x):
        return {"count": 0, "mean": None, "min": None, "p10": None,
                "median": None, "p90": None, "max": None}
    quantiles = torch.quantile(x, torch.tensor([.1, .5, .9], dtype=x.dtype))
    return {"count": len(x), "mean": float(x.mean()), "min": float(x.min()),
            "p10": float(quantiles[0]), "median": float(quantiles[1]),
            "p90": float(quantiles[2]), "max": float(x.max())}


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float((a * b).sum() / denominator) if denominator else None


def norm_ratio(a, b):
    denominator = float(b.norm())
    return float(a.norm() / denominator) if denominator else None


def _average_ranks(values):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    ordered, indices = torch.sort(values, stable=True)
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    ends = counts.cumsum(0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1
    ordered_ranks = torch.repeat_interleave((starts + ends) / 2, counts)
    ranks = torch.empty_like(values)
    ranks[indices] = ordered_ranks
    return ranks


def spearman(a, b):
    if len(a) < 2:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    return cosine(ra - ra.mean(), rb - rb.mean())


def _case_key(case):
    return (str(case.get("stage", "unspecified")), str(case.get("pair_id", case.get("step", 0))))


def _view(case):
    return int(case.get("view", 0))


def _validate_case(case, dim=256):
    mu = torch.as_tensor(case["mu"]).detach().cpu()
    labels = torch.as_tensor(case["gold"]).reshape(-1).long().cpu()
    ids = torch.as_tensor(case["ids"]).reshape(-1).long().cpu()
    if mu.ndim != 2 or mu.shape[1] != dim or len(labels) != len(mu) or len(ids) != len(mu):
        raise ValueError("Cache requires aligned mu[N,256], gold[N], ids[N]")
    if not mu.is_floating_point():
        raise ValueError("mu must be floating point")
    if not bool(torch.isfinite(mu).all()) or not bool((mu.square().sum(-1) < 1).all()):
        raise ValueError("mu must be finite and strictly inside c=1 ball")
    return mu, labels, ids


def _cross_distance_ranks(distances, labels):
    """One-based raw-distance ranks among cross-class candidates only.

    Label similarity boost and the self diagonal are excluded. Stable sort
    resolves equal distances by position for a deterministic description.
    This rank is not an independent semantic/geometric quality label.
    """
    cross = labels[:, None] != labels[None, :]
    n = len(labels)
    ranks = torch.zeros((n, n), dtype=torch.long)
    percentages = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        candidates = cross[i].nonzero(as_tuple=False).flatten()
        order = torch.argsort(distances[i, candidates], stable=True)
        nearest = candidates[order]
        ranks[i, nearest] = torch.arange(1, len(nearest) + 1)
        if len(nearest):
            percentages[i, nearest] = torch.arange(1, len(nearest) + 1, dtype=torch.float64) / len(nearest)
    return cross, ranks, percentages


def geometric_graph_report(distances, labels, radius, mined, ranks, percentages,
                           baseline_mutual, previous_mutual):
    """Describe cross edges and raw cross-positive/cross-negative gaps.

    Gap = d(i,k)-d(i,j), over eligible i and all candidate j/k Cartesian
    pairs with j and k both cross-class. This excludes self-k and automatic
    same-class positive boosts. The gap uses the same learned embedding as
    mining and therefore measures internal ordering, not morphology truth.
    """
    cross = labels[:, None] != labels[None, :]
    mutual = mined["mutual"]
    cross_edges = mutual & cross
    new_edges = cross_edges & ~baseline_mutual
    incremental_edges = cross_edges & ~previous_mutual
    eligible_cross_edges = cross_edges & mined["eligible"][:, None]
    gaps = []
    for i in mined["eligible"].nonzero(as_tuple=False).flatten().tolist():
        positive = (mutual[i] & cross[i]).nonzero(as_tuple=False).flatten()
        negative = (mined["negative"][i] & cross[i]).nonzero(as_tuple=False).flatten()
        if len(positive) and len(negative):
            gaps.append((distances[i, negative][None, :] - distances[i, positive][:, None]).flatten())
    gaps = torch.cat(gaps) if gaps else distances.new_empty(0)

    def edges(mask):
        endpoint_radius = radius[:, None].expand_as(distances)[mask]
        neighbor_radius = radius[None, :].expand_as(distances)[mask]
        return {"directed_count": int(mask.sum()),
                "distance": distribution(distances[mask]),
                "cross_distance_rank": distribution(ranks[mask]),
                "cross_distance_rank_fraction": distribution(percentages[mask]),
                "anchor_ball_radius": distribution(endpoint_radius),
                "neighbor_ball_radius": distribution(neighbor_radius)}

    return {"cross_reciprocal_edges": edges(cross_edges),
            "eligible_cross_reciprocal_edges": edges(eligible_cross_edges),
            "new_cross_edges_vs_minimum_k": edges(new_edges),
            "new_cross_edges_vs_previous_k": edges(incremental_edges),
            "eligible_ball_radius": distribution(radius[mined["eligible"]]),
            "ineligible_ball_radius": distribution(radius[~mined["eligible"]]),
            "radius_mutual_degree_spearman": spearman(radius, mutual.sum(-1)),
            "radius_eligible_indicator_spearman": spearman(radius, mined["eligible"].float()),
            "cross_only_candidate_gap": {"gap": distribution(gaps),
                 "positive_gap_fraction": fraction(int((gaps > 0).sum()), len(gaps)),
                 "negative_gap_fraction": fraction(int((gaps < 0).sum()), len(gaps)),
                 "zero_gap_fraction": fraction(int((gaps == 0).sum()), len(gaps))}}


def gradient_report(case, proxy, mined, seed=22, lambda_hier=.5):
    """One live draw, global mean gradients; nu is held fixed for mu audit."""
    mu = case["mu"].detach().cpu().clone().requires_grad_(True)
    loss, loss_stats, details = proxy.sample_loss(
        mu, mined["triplets"],
        generator=torch.Generator(device="cpu").manual_seed(seed), return_details=True)
    gradient = torch.autograd.grad(loss, mu, retain_graph=True)[0]
    if len(mined["triplets"]):
        self_loss = (details["per_draw_loss"] * details["self_k_mask"]).sum() / len(mined["triplets"])
        self_gradient = torch.autograd.grad(self_loss, mu)[0]
    else:
        self_gradient = torch.zeros_like(mu)
    weighted = lambda_hier * gradient
    norms = gradient.detach().norm(dim=-1)
    energies = norms.square()
    energy_sum = float(energies.sum())
    eligible = mined["eligible"]
    triples = mined["triplets"]
    role_count = torch.bincount(triples.flatten(), minlength=len(mu)) if len(triples) else torch.zeros(len(mu), dtype=torch.long)
    zeros = norms <= 1e-12
    top_count = max(1, (len(mu) + 9) // 10)
    report = {**loss_stats, "sample_loss": float(loss.detach()),
              "sample_mu_gradient_norm": float(gradient.norm()),
              "weighted_sample_mu_gradient_norm": float(weighted.norm()),
              "self_k_mu_gradient_norm": float(self_gradient.norm()),
              "self_k_mu_gradient_norm_ratio": norm_ratio(self_gradient, gradient),
              "per_position_gradient_norm": distribution(norms),
              "eligible_position_gradient_norm": distribution(norms[eligible]),
              "ineligible_position_gradient_norm": distribution(norms[~eligible]),
              "zero_gradient_positions": int(zeros.sum()),
              "zero_gradient_fraction": float(zeros.float().mean()),
              "eligible_zero_gradient_fraction": fraction(int((zeros & eligible).sum()), int(eligible.sum())),
              "ineligible_zero_gradient_fraction": fraction(int((zeros & ~eligible).sum()), int((~eligible).sum())),
              "top_10_percent_position_gradient_energy_fraction": fraction(float(torch.topk(energies, top_count).values.sum()), energy_sum),
              "eligible_position_gradient_energy_fraction": fraction(float(energies[eligible].sum()), energy_sum),
              "gradient_norm_by_position": norms.tolist(),
              "draw_role_count_by_position": role_count.tolist(),
              "radius_gradient_norm_spearman": spearman(mu.detach().norm(dim=-1), norms),
              "ce_comparison": {"available": False}, "intra_comparison": {"available": False}}
    if "ce_mu_grad" in case:
        ce = torch.as_tensor(case["ce_mu_grad"]).detach().cpu().to(mu)
        if ce.shape != mu.shape or not bool(torch.isfinite(ce).all()):
            raise ValueError("Cached ce_mu_grad shape/finite check failed")
        report["ce_comparison"] = {"available": True, "ce_mu_gradient_norm": float(ce.norm()),
            "weighted_hier_to_ce_gradient_norm_ratio": norm_ratio(weighted, ce),
            "hier_ce_gradient_cosine": cosine(weighted, ce),
            "semantics": "Cached CE gradient must be of global mean wrt global mu; no extra DDP world-size factor."}
    if "nu" in case and "child_count" in case:
        from .base_protocol import intra_losses
        nu = torch.as_tensor(case["nu"]).detach().cpu().to(mu)
        if nu.shape != mu.shape:
            raise ValueError("Cached nu shape does not match mu")
        terms = intra_losses(mu, nu, int(case["child_count"]))
        contr = .01 * terms["intra_contrastive"]
        radial = .01 * terms["intra_radial"]
        gc = torch.autograd.grad(contr, mu, retain_graph=True)[0]
        gr = torch.autograd.grad(radial, mu)[0]
        gi = gc + gr
        report["intra_comparison"] = {"available": True,
            "weighted_contrastive_loss": float(contr.detach()), "weighted_radial_loss": float(radial.detach()),
            "weighted_intra_mu_gradient_norm": float(gi.norm()),
            "weighted_hier_to_intra_gradient_norm_ratio": norm_ratio(weighted, gi),
            "hier_intra_gradient_cosine": cosine(weighted, gi),
            "hier_contrastive_gradient_cosine": cosine(weighted, gc),
            "hier_radial_gradient_cosine": cosine(weighted, gr),
            "semantics": "Original global64 intra means and coefficients .01/.01; cached nu held fixed for mu partial derivative."}
        if "ce_mu_grad" in case:
            base = ce + gi
            report["base_comparison"] = {"base_mu_gradient_norm": float(base.norm()),
                "weighted_hier_to_base_gradient_norm_ratio": norm_ratio(weighted, base),
                "hier_base_gradient_cosine": cosine(weighted, base)}
    for value in (gradient, self_gradient):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Nonfinite fixed-feature mu gradient")
    return report


def _select_gradient_cases(cases, maximum):
    """Evenly spread view-0 cases within every stage, with a global cap."""
    if maximum < 0:
        raise ValueError("gradient_cases must be nonnegative")
    by_stage = defaultdict(list)
    for index, case in enumerate(cases):
        if _view(case) == 0:
            by_stage[str(case.get("stage", "unspecified"))].append(index)
    eligible_count = sum(map(len, by_stage.values()))
    maximum = min(maximum, eligible_count)
    selected = []
    allocations = {key: 0 for key in by_stage}
    for _ in range(maximum):
        choices = [key for key in sorted(by_stage) if allocations[key] < len(by_stage[key])]
        key = min(choices, key=lambda value: allocations[value] / len(by_stage[value]))
        allocations[key] += 1
    for key in sorted(by_stage):
        positions = by_stage[key]
        count = allocations[key]
        if count:
            slots = torch.linspace(0, len(positions) - 1, count).round().long().tolist()
            selected.extend(positions[slot] for slot in slots)
    return sorted(selected)


def _conditional_eligibility(cases, row_indices, graphs):
    """Primary view only, avoiding duplicate weighting of paired views."""
    per_class = defaultdict(lambda: {"sampled_positions": 0, "eligible_positions": 0, "batches_present": 0,
                                     "batches_with_any_eligible": 0})
    per_id = defaultdict(lambda: {"sampled_positions": 0, "eligible_positions": 0, "batches_present": 0,
                                  "batches_with_any_eligible": 0})
    used = []
    for index in row_indices:
        if _view(cases[index]) != 0:
            continue
        used.append(index)
        labels, ids = cases[index]["gold"].tolist(), cases[index]["ids"].tolist()
        eligible = graphs[index]["eligible"].tolist()
        classes_in_batch, eligible_classes = set(labels), set()
        ids_in_batch, eligible_ids = set(ids), set()
        for label, sample_id, is_eligible in zip(labels, ids, eligible):
            per_class[label]["sampled_positions"] += 1
            per_class[label]["eligible_positions"] += int(is_eligible)
            per_id[sample_id]["sampled_positions"] += 1
            per_id[sample_id]["eligible_positions"] += int(is_eligible)
            per_id[sample_id]["class"] = label
            if is_eligible:
                eligible_classes.add(label)
                eligible_ids.add(sample_id)
        for label in classes_in_batch:
            per_class[label]["batches_present"] += 1
            per_class[label]["batches_with_any_eligible"] += int(label in eligible_classes)
        for sample_id in ids_in_batch:
            per_id[sample_id]["batches_present"] += 1
            per_id[sample_id]["batches_with_any_eligible"] += int(sample_id in eligible_ids)

    def finish(entries):
        rows = {}
        for key, value in sorted(entries.items()):
            rows[str(key)] = {**value,
                "eligible_fraction_given_sampled_position": fraction(value["eligible_positions"], value["sampled_positions"]),
                "any_eligible_fraction_given_batch_present": fraction(value["batches_with_any_eligible"], value["batches_present"])}
        return rows

    class_rows, id_rows = finish(per_class), finish(per_id)
    appeared_twice = [v for v in id_rows.values() if v["batches_present"] >= 2]
    never_ids = [key for key, value in id_rows.items() if value["eligible_positions"] == 0]
    return {"primary_view_cases": len(used), "conditional_not_full_dataset_coverage": True,
            "per_class": class_rows, "per_id": id_rows,
            "classes_seen": len(class_rows),
            "classes_with_zero_eligible_positions": [key for key, value in class_rows.items() if value["eligible_positions"] == 0],
            "class_conditional_eligible_fraction": distribution([v["eligible_fraction_given_sampled_position"] for v in class_rows.values()]),
            "sample_ids_seen": len(id_rows), "sample_ids_never_eligible": len(never_ids),
            "sample_ids_never_eligible_fraction": fraction(len(never_ids), len(id_rows)),
            "sample_ids_observed_in_multiple_batches": len(appeared_twice),
            "multi_batch_ids_never_eligible_fraction": fraction(sum(v["eligible_positions"] == 0 for v in appeared_twice), len(appeared_twice)),
            "id_conditional_eligible_fraction": distribution([v["eligible_fraction_given_sampled_position"] for v in id_rows.values()])}


def matched_view_report(cases, row_indices, graphs, baseline_graphs=None):
    pairs = defaultdict(dict)
    for index in row_indices:
        key, view = _case_key(cases[index]), _view(cases[index])
        if view in pairs[key]:
            raise ValueError(f"Duplicate cache pair/view: {key}/{view}")
        pairs[key][view] = index
    rows = []
    skipped = 0
    for key, views in sorted(pairs.items()):
        if 0 not in views or len(views) < 2:
            continue
        first = views[0]
        for view, second in sorted(views.items()):
            if view == 0:
                continue
            a, b = cases[first], cases[second]
            if not torch.equal(a["ids"], b["ids"]) or not torch.equal(a["gold"], b["gold"]):
                skipped += 1
                continue
            ga, gb = graphs[first], graphs[second]
            cross = a["gold"][:, None] != a["gold"][None, :]
            ma, mb = ga["mutual"], gb["mutual"]
            ca, cb = ma & cross, mb & cross
            ea, eb = ga["eligible"], gb["eligible"]
            cross_union = int((ca | cb).sum())
            cross_counts = cross.sum(-1).clamp_min(1)
            expected_intersection = float((ca.sum(-1) * cb.sum(-1) / cross_counts).sum())
            expected_union = float(ca.sum() + cb.sum()) - expected_intersection
            if baseline_graphs is None:
                newa, newb = ca, cb
            else:
                newa = ca & ~baseline_graphs[first]["mutual"]
                newb = cb & ~baseline_graphs[second]["mutual"]
            new_union = int((newa | newb).sum())
            new_total = int(newa.sum() + newb.sum())
            rows.append({"stage": key[0], "pair_id": key[1], "other_view": view,
                "cross_edge_intersection": int((ca & cb).sum()), "cross_edge_union": cross_union,
                "cross_edge_jaccard": fraction(int((ca & cb).sum()), cross_union) if cross_union else None,
                "cross_edge_degree_matched_independent_expected_jaccard": fraction(expected_intersection, expected_union) if expected_union else None,
                "new_vs_minimum_k_edge_intersection": int((newa & newb).sum()),
                "new_vs_minimum_k_edge_union": new_union,
                "new_vs_minimum_k_edge_jaccard": fraction(int((newa & newb).sum()), new_union) if new_union else None,
                "new_vs_minimum_k_edges_total": new_total,
                "new_vs_minimum_k_edges_retained_in_other_full_graph": int((newa & cb).sum() + (newb & ca).sum()),
                "all_edge_intersection": int((ma & mb).sum()), "all_edge_union": int((ma | mb).sum()),
                "eligible_intersection": int((ea & eb).sum()), "eligible_union": int((ea | eb).sum()),
                "eligible_jaccard": fraction(int((ea & eb).sum()), int((ea | eb).sum())) if bool((ea | eb).any()) else None,
                "eligible_state_agreement": float((ea == eb).float().mean()),
                "eligible_view0_fraction": float(ea.float().mean()), "eligible_other_fraction": float(eb.float().mean()),
                "same_id_duplicates_in_batch": len(a["ids"]) - len(set(a["ids"].tolist()))})
    cross_intersection = sum(r["cross_edge_intersection"] for r in rows)
    cross_union = sum(r["cross_edge_union"] for r in rows)
    eligible_intersection = sum(r["eligible_intersection"] for r in rows)
    eligible_union = sum(r["eligible_union"] for r in rows)
    new_union = sum(r["new_vs_minimum_k_edge_union"] for r in rows)
    new_total = sum(r["new_vs_minimum_k_edges_total"] for r in rows)
    return {"matched_pairs": len(rows), "skipped_unmatched_order_or_labels": skipped,
            "cross_edge_undefined_empty_union_pairs": sum(r["cross_edge_jaccard"] is None for r in rows),
            "cross_edge_jaccard_pooled": fraction(cross_intersection, cross_union) if cross_union else None,
            "cross_edge_jaccard": distribution([r["cross_edge_jaccard"] for r in rows if r["cross_edge_jaccard"] is not None]),
            "eligible_jaccard_pooled": fraction(eligible_intersection, eligible_union) if eligible_union else None,
            "eligible_state_agreement": distribution([r["eligible_state_agreement"] for r in rows]),
            "cross_edge_degree_matched_independent_expected_jaccard": distribution([r["cross_edge_degree_matched_independent_expected_jaccard"] for r in rows if r["cross_edge_degree_matched_independent_expected_jaccard"] is not None]),
            "new_vs_minimum_k_edge_jaccard_pooled": fraction(sum(r["new_vs_minimum_k_edge_intersection"] for r in rows), new_union) if new_union else None,
            "new_vs_minimum_k_edges_retained_in_other_full_graph_fraction": fraction(sum(r["new_vs_minimum_k_edges_retained_in_other_full_graph"] for r in rows), new_total) if new_total else None,
            "rows": rows, "comparison_unit": "Aligned batch positions; duplicate data IDs remain separate augmented positions.",
            "chance_baseline": "Descriptive independent cross-candidate choice with matched per-row degree; reciprocal graph dependencies are ignored, so this is not a significance test."}


def _aggregate_stage(cases, indices, graphs, rows, baseline_graphs):
    total_positions = sum(len(cases[index]["mu"]) for index in indices)
    eligible = sum(graphs[index]["stats"]["eligible_anchors"] for index in indices)
    any_role = sum(graphs[index]["stats"]["any_role_positions"] for index in indices)
    draws = sum(graphs[index]["stats"]["triplets"] for index in indices)
    distances = [rows[index]["geometry"]["cross_reciprocal_edges"]["distance"]["mean"] for index in indices
                 if rows[index]["geometry"]["cross_reciprocal_edges"]["distance"]["count"]]
    losses = [rows[index]["gradient"] for index in indices if "gradient" in rows[index]]
    result = {"cases": len(indices), "positions": total_positions,
        "eligible_anchors": eligible, "anchor_fraction": fraction(eligible, total_positions),
        "per_case_anchor_fraction": distribution([graphs[i]["stats"]["anchor_coverage"] for i in indices]),
        "any_role_fraction": fraction(any_role, total_positions), "triplets": draws,
        "self_k_fraction": fraction(sum(graphs[i]["stats"]["self_k_triplets"] for i in indices), draws),
        "cross_j_fraction": fraction(sum(graphs[i]["stats"]["triplets"] - graphs[i]["stats"]["same_class_j_triplets"] for i in indices), draws),
        "cross_edge_case_mean_distance": distribution(distances),
        "conditional_eligibility": _conditional_eligibility(cases, indices, graphs),
        "matched_views": matched_view_report(cases, indices, graphs, baseline_graphs),
        "gradient_cases": len(losses)}
    geometry = [rows[i]["geometry"] for i in indices]
    gap_count = sum(g["cross_only_candidate_gap"]["gap"]["count"] for g in geometry)
    def weighted_mean(section, metric):
        weighted = [(g[section][metric]["count"], g[section][metric]["mean"]) for g in geometry]
        count = sum(n for n, _ in weighted)
        return sum(n * mean for n, mean in weighted if n) / count if count else None
    result["geometric_summary"] = {
        "cross_only_candidate_gap_count": gap_count,
        "cross_only_candidate_positive_gap_fraction": fraction(sum(g["cross_only_candidate_gap"]["positive_gap_fraction"] * g["cross_only_candidate_gap"]["gap"]["count"] for g in geometry), gap_count),
        "cross_only_candidate_gap_mean": sum(g["cross_only_candidate_gap"]["gap"]["mean"] * g["cross_only_candidate_gap"]["gap"]["count"] for g in geometry if g["cross_only_candidate_gap"]["gap"]["count"]) / gap_count if gap_count else None,
        "incremental_new_edge_distance_mean": weighted_mean("new_cross_edges_vs_previous_k", "distance"),
        "incremental_new_edge_cross_rank_fraction_mean": weighted_mean("new_cross_edges_vs_previous_k", "cross_distance_rank_fraction"),
        "incremental_new_edge_neighbor_radius_mean": weighted_mean("new_cross_edges_vs_previous_k", "neighbor_ball_radius"),
        "radius_mutual_degree_spearman": distribution([g["radius_mutual_degree_spearman"] for g in geometry if g["radius_mutual_degree_spearman"] is not None]),
        "radius_eligible_indicator_spearman": distribution([g["radius_eligible_indicator_spearman"] for g in geometry if g["radius_eligible_indicator_spearman"] is not None])}
    if losses:
        total_draws = sum(v["triplets"] for v in losses)
        valid = sum(v["noncollision_triplets"] for v in losses)
        active = sum(v["active_triplets"] for v in losses)
        result["gradient_summary"] = {
            "sample_loss": distribution([v["sample_loss"] for v in losses]),
            "weighted_mu_gradient_norm": distribution([v["weighted_sample_mu_gradient_norm"] for v in losses]),
            "collision_fraction": fraction(sum(v["collisions"] for v in losses), total_draws),
            "active_fraction_all_draws": fraction(active, total_draws),
            "active_fraction_noncollision": fraction(active, valid),
            "zero_gradient_position_fraction": distribution([v["zero_gradient_fraction"] for v in losses]),
            "top10_gradient_energy_fraction": distribution([v["top_10_percent_position_gradient_energy_fraction"] for v in losses]),
            "self_k_gradient_norm_ratio": distribution([v["self_k_mu_gradient_norm_ratio"] for v in losses if v["self_k_mu_gradient_norm_ratio"] is not None]),
            "radius_gradient_norm_spearman": distribution([v["radius_gradient_norm_spearman"] for v in losses if v["radius_gradient_norm_spearman"] is not None])}
        for component, fields in {"ce_comparison": ["weighted_hier_to_ce_gradient_norm_ratio", "hier_ce_gradient_cosine"],
                                  "intra_comparison": ["weighted_hier_to_intra_gradient_norm_ratio", "hier_intra_gradient_cosine"]}.items():
            available = [v[component] for v in losses if v[component].get("available")]
            result["gradient_summary"][component] = {
                "available_cases": len(available),
                **{field: distribution([v[field] for v in available if v[field] is not None]) for field in fields}}
    return result


def analyze_cache(cache, ks=(10, 12, 16, 20), gradient_cases=8, seed=22,
                  lambda_hier=.5, proxy_count=512, t_per_anchor=50):
    """Reusable CPU cache analyzer. Does not mutate cache, model or proxies."""
    ks = tuple(sorted(set(map(int, ks))))
    if not ks or max(ks) > 64 or min(ks) < 3 or gradient_cases > 16:
        raise ValueError("Require K3..64 and gradient_cases <=16")
    if t_per_anchor < 1 or lambda_hier < 0:
        raise ValueError("Invalid T or loss coefficient")
    if not cache.get("cases"):
        raise ValueError("Feature cache is empty")
    cases = []
    for original in cache["cases"]:
        mu, labels, ids = _validate_case(original)
        if max(ks) > len(mu):
            raise ValueError("K exceeds batch size")
        cases.append({**original, "mu": mu, "gold": labels, "ids": ids})
    selected = set(_select_gradient_cases(cases, gradient_cases))
    proxy = HIERLoss(proxy_count, dim=256, c=1, margin=.1, tau=.1, seed=seed)
    stage_indices = defaultdict(list)
    for index, case in enumerate(cases):
        stage_indices[str(case.get("stage", "unspecified"))].append(index)
    all_rows, graph_by_k = {k: [] for k in ks}, {k: [] for k in ks}
    for index, case in enumerate(cases):
        distances = poincare_distance(case["mu"])
        labels = case["gold"]
        similarity = torch.exp(-distances)
        similarity[labels[:, None] == labels[None, :]] += 1
        _, ranks, percentages = _cross_distance_ranks(distances, labels)
        baseline, previous = None, None
        radius = case["mu"].norm(dim=-1)
        # Seed is reset across K for a case. Shapes/eligible anchors differ,
        # so neither relation draws nor Gumbel noise are exactly edge-coupled.
        case_seed = seed + index
        for k in ks:
            mined = mine_reciprocal_triplets(similarity, topk=k, t_per_anchor=t_per_anchor,
                seed=case_seed, labels=labels, data_ids=case["ids"])
            if baseline is None:
                baseline = mined["mutual"]
                previous = baseline
            row = {"case_index": index, "stage": str(case.get("stage", "unspecified")),
                "step": case.get("step", 0), "pair_id": case.get("pair_id", case.get("step", 0)),
                "view": _view(case), "k": k, "case_seed": case_seed,
                "relation": mined["stats"], "eligible_by_position": mined["eligible"].tolist(),
                "geometry": geometric_graph_report(distances, labels, radius, mined, ranks, percentages, baseline, previous),
                "whole_shadow_geometry": shadow_geometry(case["mu"])}
            if index in selected:
                row["gradient"] = gradient_report(case, proxy, mined, case_seed, lambda_hier)
            all_rows[k].append(row)
            graph_by_k[k].append(mined)
            previous = mined["mutual"]
    proxy_reports = []
    for k in ks:
        distances = poincare_distance(proxy.proxies())
        mined = mine_reciprocal_triplets(torch.exp(-distances.detach()), topk=k,
                    t_per_anchor=t_per_anchor, seed=seed + 1)
        with torch.no_grad():
            loss, loss_stats = ghhc_loss(distances.detach(), mined["triplets"], tau=.1, margin=.1,
                           generator=torch.Generator(device="cpu").manual_seed(seed + 1))
        proxy_reports.append({"k": k, "initial_random_proxy_graph": True,
            "relation": mined["stats"], "loss": float(loss), "loss_stats": loss_stats,
            "shadow_geometry": shadow_geometry(proxy.proxies().detach())})
    summaries = []
    for k in ks:
        for stage, indices in sorted(stage_indices.items()):
            summaries.append({"k": k, "stage": stage,
                **_aggregate_stage(cases, indices, graph_by_k[k], all_rows[k], graph_by_k[ks[0]])})
    return {"cache_identity": cache.get("identity"), "cache_config": cache.get("config"),
        "config": {"ks": list(ks), "gradient_cases": gradient_cases,
            "selected_gradient_case_indices": sorted(selected), "seed": seed, "c": 1,
            "dim": 256, "proxy_count": proxy_count, "proxy_initialization": "random_tangent",
            "margin": .1, "tau": .1, "lambda_hier_for_gradient_comparison": lambda_hier,
            "t_per_anchor": t_per_anchor, "extra_hier_cap": False, "extra_hier_backward": False,
            "optimizer_updates": 0, "main_training": False},
        "semantics": {"features": "TRAINING subset cached representations only; no validation/test performance evaluation.",
            "random_matching": "Identical initialization; reset same per-case seed across K. Changed shapes and eligible sets prevent exact draw/noise correspondence.",
            "gradient": "Partial gradient wrt detached global mu; proxy regularizer has no direct mu path. .5 multiplies sample gradient.",
            "gradient_energy": "Squared per-position partial-gradient norms, not additive attribution of parameter gradients.",
            "geometric_gap": "Internal learned-embedding ordering only; both j/k cross-class, excluding self and label-boosted positive pairs.",
            "coverage": "Conditional on sampled cache positions; not an estimate of full training-set or epoch coverage.",
            "proxy": "Each K uses the identical initial random proxy graph; no proxy updates or trained-graph performance claim."},
        "summaries": summaries, "initial_proxy_graphs": proxy_reports,
        "rows": [row for k in ks for row in all_rows[k]]}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 12, 16, 20])
    parser.add_argument("--gradient-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--lambda-hier", type=float, default=.5)
    parser.add_argument("--proxy-count", type=int, default=512)
    parser.add_argument("--t-per-anchor", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {args.output}")
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    torch.set_num_threads(args.threads)
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    report = analyze_cache(cache, args.ks, args.gradient_cases, args.seed,
                           args.lambda_hier, args.proxy_count, args.t_per_anchor)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report["manifest"] = {"commit": commit, "source_module_sha256": _sha256(__file__),
        "cache_sha256": _sha256(args.features), "feature_cache": str(args.features),
        "start_utc": started, "end_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started_clock, "torch": torch.__version__,
        "device": "cpu", "threads": args.threads, "args": {key: str(value) if isinstance(value, Path) else value
                                                               for key, value in vars(args).items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False plus exclusive creation guards invalid/overwritten evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    compact = []
    for summary in report["summaries"]:
        compact.append({"stage": summary["stage"], "k": summary["k"],
            "anchor_fraction": summary["anchor_fraction"],
            "cross_edge_view_jaccard": summary["matched_views"]["cross_edge_jaccard_pooled"],
            "class_min_conditional_eligible": summary["conditional_eligibility"]["class_conditional_eligible_fraction"]["min"],
            "gradient_cases": summary["gradient_cases"]})
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "seconds": report["manifest"]["wall_seconds"], "summaries": compact}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
