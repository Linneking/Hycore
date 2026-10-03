"""CPU-only sampling coverage audit; never loads models or starts training.

Input JSON is either training-subset labels (a list or ``{"labels": [...]}``)
or class counts (a list or ``{"counts": [...]}``). Counts create synthetic
within-class IDs and give a distribution audit, not actual dataset-ID proof.
Outputs belong in an ignored diagnostic directory, never in the code tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .sampler import SAMPLER_VERSION, SourceClassBatchSampler
except ImportError:  # Also support direct invocation of this script.
    from sampler import SAMPLER_VERSION, SourceClassBatchSampler


def theoretical_coverage(class_counts, steps: int, classes_per_batch: int = 32,
                         instances_per_class: int = 2) -> dict:
    counts = {int(label): int(n) for label, n in class_counts.items()}
    if not counts or any(n < 1 for n in counts.values()):
        raise ValueError("nonempty positive class counts required")
    if not 0 < classes_per_batch <= len(counts) or instances_per_class < 1 or steps < 1:
        raise ValueError("invalid batch shape or step count")
    p_selected = classes_per_batch / len(counts)
    per_class = {}
    for label, n in sorted(counts.items()):
        p_missed = 1 - p_selected + p_selected * (1 - 1 / n) ** instances_per_class
        fraction = 1 - p_missed ** steps
        per_class[str(label)] = {
            "class_size": n, "expected_draws": steps * p_selected * instances_per_class,
            "expected_unique": n * fraction, "expected_coverage": fraction,
            "hard_max_coverage": min(1.0, steps * instances_per_class / n),
        }
    dataset_size = sum(counts.values())
    unique = sum(item["expected_unique"] for item in per_class.values())
    draws = steps * classes_per_batch * instances_per_class
    largest = max(counts, key=counts.get)
    return {
        "steps": steps, "draws": draws, "dataset_size": dataset_size,
        "expected_unique": unique, "expected_coverage": unique / dataset_size,
        "expected_repeated_draws": draws - unique,
        "largest_class": largest, "largest_class_expected_coverage":
            per_class[str(largest)]["expected_coverage"],
        "per_class": per_class,
    }


def audit_sampler(sampler: SourceClassBatchSampler, epoch: int = 0,
                  segment_steps: int = 20) -> dict:
    started = time.perf_counter()
    draws_by_class = Counter()
    steps_by_class = defaultdict(list)
    unique_by_class = defaultdict(set)
    duplicate_blocks = 0
    duplicate_draws_inside_blocks = 0
    flip_matches = 0
    flip_pairs_checked = 0
    global_flip_matches = 0
    global_flip_pairs_checked = 0
    block_failures = 0
    rank_overlap_steps = 0
    first_segment_counts = Counter()
    last_segment_counts = Counter()
    plan_hasher = hashlib.sha256()
    for plan in sampler.iter_global_plans(epoch):
        plan_hasher.update(json.dumps(plan.indices, separators=(",", ":")).encode("ascii"))
        plan_hasher.update(b"\n")
        for label in plan.classes:
            steps_by_class[label].append(plan.step)
        for label, index in zip(plan.labels, plan.indices):
            draws_by_class[label] += 1
            unique_by_class[label].add(index)
            if plan.step < segment_steps:
                first_segment_counts[label] += 1
            if plan.step >= sampler.steps - segment_steps:
                last_segment_counts[label] += 1
        rank_classes = [set(plan.rank_classes(rank)) for rank in range(sampler.world_size)]
        if sum(map(len, rank_classes)) != len(set().union(*rank_classes)):
            rank_overlap_steps += 1
        for start in range(0, sampler.global_batch_size, sampler.instances_per_class):
            block_labels = plan.labels[start:start + sampler.instances_per_class]
            block_indices = plan.indices[start:start + sampler.instances_per_class]
            block_failures += int(len(set(block_labels)) != 1)
            repeated = len(block_indices) - len(set(block_indices))
            duplicate_blocks += int(repeated > 0)
            duplicate_draws_inside_blocks += repeated
        for rank in range(sampler.world_size):
            local_labels = plan.rank_labels(rank)
            for label, flipped_label in zip(local_labels, reversed(local_labels)):
                flip_pairs_checked += 1
                flip_matches += int(label == flipped_label)
        for label, flipped_label in zip(plan.labels, reversed(plan.labels)):
            global_flip_pairs_checked += 1
            global_flip_matches += int(label == flipped_label)
    dataset_size = len(sampler.labels)
    total_draws = sampler.steps * sampler.global_batch_size
    total_unique = sum(map(len, unique_by_class.values()))
    per_class = {}
    for label in sampler.classes:
        n = len(sampler.by_class[label])
        unique = len(unique_by_class[label])
        selected_steps = steps_by_class[label]
        per_class[str(label)] = {
            "class_size": n, "draws": draws_by_class[label], "unique": unique,
            "coverage": unique / n, "repeated_draws": draws_by_class[label] - unique,
            "selected_steps": len(selected_steps),
            "first_selected_step": selected_steps[0] if selected_steps else None,
            "last_selected_step": selected_steps[-1] if selected_steps else None,
            "first_segment_draws": first_segment_counts[label],
            "last_segment_draws": last_segment_counts[label],
        }
    return {
        "seed": sampler.seed, "epoch": epoch, "steps": sampler.steps,
        "world_size": sampler.world_size, "classes_per_rank": sampler.classes_per_rank,
        "instances_per_class": sampler.instances_per_class,
        "local_batch_size": sampler.local_batch_size,
        "global_batch_size": sampler.global_batch_size,
        "dataset_size": dataset_size, "draws": total_draws,
        "unique": total_unique, "coverage": total_unique / dataset_size,
        "repeated_draws": total_draws - total_unique,
        "repeat_fraction": 1 - total_unique / total_draws,
        "within_class_block_duplicate_blocks": duplicate_blocks,
        "within_class_block_duplicate_draws": duplicate_draws_inside_blocks,
        "rank_overlap_steps": rank_overlap_steps,
        "class_block_failures": block_failures,
        "flip_same_class_pairs": flip_matches,
        "flip_pairs_checked": flip_pairs_checked,
        "flip_all_strictly_different_classes": flip_matches == 0,
        "flip_scope": "rank_local",
        "global_flip_same_class_pairs": global_flip_matches,
        "global_flip_pairs_checked": global_flip_pairs_checked,
        "global_flip_all_strictly_different_classes": global_flip_matches == 0,
        "segment_steps": min(segment_steps, sampler.steps),
        "first_segment_unique_classes": len(first_segment_counts),
        "last_segment_unique_classes": len(last_segment_counts),
        "plan_sha256": plan_hasher.hexdigest(),
        "per_class": per_class,
        "cpu_elapsed_seconds": time.perf_counter() - started,
    }


def load_labels(path: Path, use_counts: bool) -> tuple[list[int], dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    key = "counts" if use_counts else "labels"
    values = data[key] if isinstance(data, dict) else data
    if use_counts:
        if any(int(n) < 1 for n in values):
            raise ValueError("counts must describe nonempty training classes")
        labels = [label for label, n in enumerate(values) for _ in range(int(n))]
    else:
        labels = [int(label) for label in values]
    provenance = {
        "input_kind": "class_counts" if use_counts else "training_subset_labels",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instance_identity": "synthetic_within_class_ids" if use_counts else
            "training_subset_row_indices",
        "no_dataset_points_or_models_loaded": True,
    }
    if isinstance(data, dict):
        for field in ("provenance", "split_sha256_from_existing_manifest"):
            if field in data:
                provenance[field] = data[field]
    return labels, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--counts-json", type=Path)
    input_group.add_argument("--labels-json", type=Path)
    parser.add_argument("--mode", choices=["source"], default="source")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--classes-per-rank", type=int, default=16)
    parser.add_argument("--instances-per-class", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--compare-steps", type=int, nargs="+", default=[138, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 42, 2026])
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--segment-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, help="New JSON path; existing files are refused")
    args = parser.parse_args()
    if args.segment_steps < 1:
        parser.error("segment-steps must be positive")
    source = args.counts_json or args.labels_json
    labels, provenance = load_labels(source, args.counts_json is not None)
    counts = Counter(labels)
    budgets = sorted(set(args.compare_steps + [args.steps]))
    audits = []
    for steps in budgets:
        for seed in args.seeds:
            sampler = SourceClassBatchSampler(
                labels, world_size=args.world_size, classes_per_rank=args.classes_per_rank,
                instances_per_class=args.instances_per_class, steps=steps, seed=seed)
            for epoch in args.epochs:
                audits.append(audit_sampler(sampler, epoch, args.segment_steps))
    summaries = {}
    largest_class = max(counts, key=counts.get)
    for steps in budgets:
        selected = [audit for audit in audits if audit["steps"] == steps]
        summaries[str(steps)] = {
            "replicates": len(selected),
            "mean_coverage": statistics.mean(audit["coverage"] for audit in selected),
            "min_coverage": min(audit["coverage"] for audit in selected),
            "max_coverage": max(audit["coverage"] for audit in selected),
            "largest_class": largest_class,
            "mean_largest_class_coverage": statistics.mean(
                audit["per_class"][str(largest_class)]["coverage"] for audit in selected),
            "total_flip_same_class_pairs": sum(audit["flip_same_class_pairs"] for audit in selected),
            "total_global_flip_same_class_pairs": sum(
                audit["global_flip_same_class_pairs"] for audit in selected),
            "total_rank_overlap_steps": sum(audit["rank_overlap_steps"] for audit in selected),
            "total_class_block_failures": sum(audit["class_block_failures"] for audit in selected),
        }
    report = {
        "sampler_version": SAMPLER_VERSION,
        "scope": "CPU-only independent-epoch source-sampling diagnostic; no training",
        "provenance": provenance,
        "config": {"world_size": args.world_size, "classes_per_rank": args.classes_per_rank,
                   "instances_per_class": args.instances_per_class, "steps": args.steps,
                   "compare_steps": budgets, "seeds": args.seeds, "epochs": args.epochs},
        "theoretical": [theoretical_coverage(
            counts, steps, args.world_size * args.classes_per_rank,
            args.instances_per_class) for steps in budgets],
        "simulated_summary": summaries, "audits": audits,
        "limitations": [
            "RNG stream is versioned Python sampling, not the original NumPy sequence.",
            "Counts-only inputs use synthetic IDs and cannot audit real subset ordering.",
            "Coverage is unique instance count; total draw/dataset size is not coverage.",
            "CPU plan timing is not GPU training step timing.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"simulated_summary": summaries,
                      "output": str(args.output) if args.output else None},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
