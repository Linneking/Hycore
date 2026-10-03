"""CPU epoch telemetry for the V5 global64 and original-style32 protocols.

Only detached numeric observations are stored. Pair/triple coverage is
pooled over batch-position candidate domains; it is not an epoch-wide set
of unique instance-ID triples. Near-boundary and shadow-cap counters are
observations of final embeddings, not measurements of projection calls.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


def _array(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None,
                "p10": None, "median": None, "p90": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite telemetry observation")
    return {"count": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "p10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max())}


class _Scalar:
    def __init__(self):
        self.values, self.weights = [], []

    def update(self, value, weight=1):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Nonfinite scalar telemetry")
        self.values.append(value)
        self.weights.append(float(weight))

    def summary(self):
        result = _distribution(self.values)
        result["sum"] = float(sum(self.values))
        total_weight = sum(self.weights)
        result["batch_size_weighted_mean"] = _fraction(
            sum(v * w for v, w in zip(self.values, self.weights)), total_weight)
        result["over_1_count"] = sum(v > 1 for v in self.values)
        result["over_1_fraction"] = _fraction(result["over_1_count"], len(self.values))
        result["over_10_count"] = sum(v > 10 for v in self.values)
        result["over_10_fraction"] = _fraction(result["over_10_count"], len(self.values))
        return result


_COUNT_FIELDS = (
    "batch_size", "eligible_anchors", "eligible_anchor_data_ids", "mutual_positive_edges",
    "anchors_with_zero_positive", "anchors_with_one_positive", "triplets", "unique_triplets",
    "repeated_triplets", "candidate_triplets", "expected_unique_triplets",
    "candidate_positive_pairs", "candidate_negative_pairs", "unique_positive_pairs", "unique_negative_pairs",
    "any_role_positions", "batch_unique_data_ids", "any_role_unique_data_ids", "unique_data_id_triples",
    "duplicate_data_id_draws", "self_k_triplets", "equal_data_id_ik_triplets", "equal_data_id_ij_triplets",
    "same_class_j_triplets", "same_class_k_triplets", "all_same_class_triplets", "same_class_nonself_k_triplets",
    "same_class_positive_candidates", "cross_class_positive_candidates", "same_class_negative_candidates",
    "cross_class_negative_candidates", "collisions", "noncollision_triplets", "active_triplets",
    "active_self_k_triplets", "active_distinct_k_triplets", "active_i_hinges", "active_j_hinges", "active_k_hinges",
    "noncollision_self_k_triplets",
)


class _GraphAggregate:
    def __init__(self):
        self.batches = self.loss_evaluated_batches = 0
        self.counts = Counter()
        self.degree_histogram = Counter()
        self.settings = {"topk_including_self": set(), "t_per_anchor": set(), "exclude_self_negative": set()}
        self.per_batch = {}
        self.loss_draws = 0
        self.draw_contribution_sum = Counter()

    def update(self, stats):
        self.batches += 1
        for key in _COUNT_FIELDS:
            if key in stats:
                value = float(stats[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid HIER count {key}: {value}")
                self.counts[key] += value
        for key in self.settings:
            if key in stats:
                self.settings[key].add(stats[key])
        for degree, count in stats.get("mutual_degree_histogram", {}).items():
            self.degree_histogram[str(degree)] += int(count)
        for key in ("anchor_coverage", "any_role_position_coverage", "mutual_degree_mean", "repeat_fraction"):
            if key in stats:
                self.per_batch.setdefault(key, _Scalar()).update(stats[key], stats.get("batch_size", 1))
        if "noncollision_triplets" in stats:
            self.loss_evaluated_batches += 1
            draws = int(stats.get("triplets", 0))
            self.loss_draws += draws
            for key in ("self_k_loss_contribution", "distinct_k_loss_contribution"):
                if key in stats:
                    value = float(stats[key])
                    if not math.isfinite(value):
                        raise ValueError("Nonfinite HIER loss contribution")
                    self.draw_contribution_sum[key] += draws * value

    def summary(self):
        if not self.batches:
            return None
        counts = {key: int(value) if key != "expected_unique_triplets" else float(value)
                  for key, value in self.counts.items()}
        numerator_denominator = {
            "anchor_coverage": ("eligible_anchors", "batch_size"),
            "anchor_data_id_coverage_pooled_batch_domains": ("eligible_anchor_data_ids", "batch_unique_data_ids"),
            "any_role_position_coverage": ("any_role_positions", "batch_size"),
            "any_role_id_coverage_pooled_batch_domains": ("any_role_unique_data_ids", "batch_unique_data_ids"),
            "candidate_triplet_coverage": ("unique_triplets", "candidate_triplets"),
            "positive_pair_coverage": ("unique_positive_pairs", "candidate_positive_pairs"),
            "negative_pair_coverage": ("unique_negative_pairs", "candidate_negative_pairs"),
            "repeat_fraction": ("repeated_triplets", "triplets"),
            "self_k_fraction": ("self_k_triplets", "triplets"),
            "equal_data_id_ik_fraction": ("equal_data_id_ik_triplets", "triplets"),
            "cross_class_j_fraction": ("same_class_j_triplets", "triplets"),
        }
        rates = {key: _fraction(self.counts[n], self.counts[d]) for key, (n, d) in numerator_denominator.items()}
        if rates["cross_class_j_fraction"] is not None:
            rates["cross_class_j_fraction"] = 1 - rates["cross_class_j_fraction"]
        if "same_class_j_triplets" not in self.counts:
            rates["cross_class_j_fraction"] = None
        if self.loss_evaluated_batches:
            rates.update({"collision_rate": _fraction(self.counts["collisions"], self.loss_draws),
                "active_fraction_all_draws": _fraction(self.counts["active_triplets"], self.loss_draws),
                "active_fraction_noncollision": _fraction(self.counts["active_triplets"], self.counts["noncollision_triplets"])})
        else:
            rates.update(collision_rate=None, active_fraction_all_draws=None, active_fraction_noncollision=None)
        return {"relation_batches": self.batches, "loss_evaluated_batches": self.loss_evaluated_batches,
            "counts": counts, "rates": rates,
            "mutual_degree_histogram": dict(sorted(self.degree_histogram.items(), key=lambda item: int(item[0]))),
            "settings_observed": {key: sorted(value) for key, value in self.settings.items()},
            "per_batch": {key: value.summary() for key, value in self.per_batch.items()},
            "per_draw_loss_contribution": {key: _fraction(value, self.loss_draws)
                for key, value in self.draw_contribution_sum.items()},
            "coverage_domain": "Sum of within-batch position/ID candidate domains; not an epoch-global unique triple set."}


class EpochTelemetry:
    def __init__(self, train_ids, train_labels, num_classes=40):
        ids = _array(train_ids, np.int64).reshape(-1).copy()
        labels = _array(train_labels, np.int64).reshape(-1).copy()
        if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Training IDs must be unique and aligned with labels")
        if num_classes < 1 or (len(labels) and (labels.min() < 0 or labels.max() >= num_classes)):
            raise ValueError("Invalid training labels")
        self.train_ids, self.train_labels, self.num_classes = ids, labels, int(num_classes)
        self.id_index = {int(sample_id): i for i, sample_id in enumerate(ids)}
        self.class_sizes = np.bincount(labels, minlength=num_classes)
        self.draw_counts = np.zeros(len(ids), dtype=np.int64)
        self.class_draw_counts = np.zeros(num_classes, dtype=np.int64)
        self.eligible_counts = np.zeros(len(ids), dtype=np.int64)
        self.eligible_batch_counts = np.zeros(len(ids), dtype=np.int64)
        self.relation_draw_counts = np.zeros(len(ids), dtype=np.int64)
        self.role_counts = {role: np.zeros(len(ids), dtype=np.int64) for role in ("i", "j", "k")}
        self.role_position_counts = Counter()
        self.any_role_ids = set()
        self.any_role_positions = 0
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.batches = self.positions = self.relation_batches = 0
        self.loss_evaluated_batches = self.objective_active_batches = 0
        self.objective_activity_known_batches = 0
        self.scalars = {}
        self.geometry_values = {part: {key: [] for key in ("radius", "depth", "inverse_metric_factor")}
                                for part in ("whole", "part")}
        self.geometry_counts = {part: Counter() for part in ("whole", "part")}
        self.proxy_geometry = {}
        self.graphs = {"sample": _GraphAggregate(), "proxy": _GraphAggregate()}

    def _observe_geometry(self, part, embedding):
        points = _array(embedding, np.float64)
        if points.ndim != 2 or not np.isfinite(points).all():
            raise ValueError("Geometry telemetry requires finite [N,D] embeddings")
        radius = np.linalg.norm(points, axis=1)
        if (radius >= 1).any():
            raise ValueError("Geometry points must be strictly inside c=1 ball")
        tangent = np.arctanh(radius)
        factor = (1 - radius ** 2) ** 2 / 4
        self.geometry_values[part]["radius"].extend(radius.tolist())
        self.geometry_values[part]["depth"].extend((2 * tangent).tolist())
        self.geometry_values[part]["inverse_metric_factor"].extend(factor.tolist())
        counts = self.geometry_counts[part]
        counts["positions"] += len(radius)
        counts["near_native_numerical_boundary"] += int((radius >= .99599).sum())
        counts["shadow_tangent_cap2_3"] += int((tangent > 2.3).sum())

    def update(self, *, ids, labels, logits, mu, nu, scalar_metrics, hier_stats=None, sample_mining=None):
        ids = _array(ids, np.int64).reshape(-1)
        labels = _array(labels, np.int64).reshape(-1)
        predictions = _array(logits)
        if predictions.ndim == 2:
            if predictions.shape != (len(ids), self.num_classes) or not np.isfinite(predictions).all():
                raise ValueError("Logits telemetry shape/finite check failed")
            predictions = predictions.argmax(axis=1)
        else:
            predictions = predictions.astype(np.int64).reshape(-1)
        if len(ids) != len(labels) or len(predictions) != len(ids) or not len(ids):
            raise ValueError("Telemetry batch IDs, labels and predictions are not aligned")
        if labels.min() < 0 or labels.max() >= self.num_classes or predictions.min() < 0 or predictions.max() >= self.num_classes:
            raise ValueError("Telemetry class index outside range")
        try:
            indices = np.asarray([self.id_index[int(sample_id)] for sample_id in ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Sample outside declared training split: {error}") from error
        if not np.array_equal(self.train_labels[indices], labels):
            raise ValueError("Telemetry ID/label mapping mismatch")
        self.batches += 1
        self.positions += len(ids)
        np.add.at(self.draw_counts, indices, 1)
        np.add.at(self.class_draw_counts, labels, 1)
        np.add.at(self.confusion, (labels, predictions), 1)
        if len(_array(mu)) != len(ids) or len(_array(nu)) != len(ids):
            raise ValueError("Geometry and input position counts differ")
        self._observe_geometry("whole", mu)
        self._observe_geometry("part", nu)
        for key, value in scalar_metrics.items():
            if value is None:
                continue
            scalar = _array(value)
            if scalar.size != 1:
                raise ValueError(f"Expected scalar telemetry for {key}")
            self.scalars.setdefault(str(key), _Scalar()).update(float(scalar.reshape(-1)[0]), len(ids))
        for key in ("lambda_hier", "hier_weight", "hier_active"):
            if key in scalar_metrics:
                self.objective_activity_known_batches += 1
                self.objective_active_batches += int(float(_array(scalar_metrics[key]).reshape(-1)[0]) > 0)
                break
        if hier_stats is not None:
            for name in ("sample", "proxy"):
                if name in hier_stats:
                    self.graphs[name].update(hier_stats[name])
            if "sample_loss" in hier_stats:
                self.loss_evaluated_batches += 1
                for key in ("sample_loss", "proxy_loss"):
                    if key in hier_stats:
                        self.scalars.setdefault("hier_" + key + "_evaluated", _Scalar()).update(hier_stats[key], len(ids))
            for part, observations in hier_stats.get("geometry", {}).items():
                if isinstance(observations, dict):
                    for key, value in observations.items():
                        if isinstance(value, (int, float, np.number)):
                            self.proxy_geometry.setdefault(part + "." + key, _Scalar()).update(value)
                elif isinstance(observations, (int, float, np.number)):
                    self.proxy_geometry.setdefault(part, _Scalar()).update(observations)
            if sample_mining is None:
                sample_mining = hier_stats.get("_sample_mining")
        if sample_mining is not None:
            self.relation_batches += 1
            eligible = _array(sample_mining["eligible"], bool).reshape(-1)
            triples = _array(sample_mining["triplets"], np.int64).reshape(-1, 3)
            if len(eligible) != len(ids) or (len(triples) and (triples.min() < 0 or triples.max() >= len(ids))):
                raise ValueError("Sample mining telemetry indices do not match batch")
            np.add.at(self.relation_draw_counts, indices, 1)
            np.add.at(self.eligible_counts, indices[eligible], 1)
            self.eligible_batch_counts[np.unique(indices[eligible])] += 1
            for column, role in enumerate(("i", "j", "k")):
                positions = triples[:, column]
                np.add.at(self.role_counts[role], indices[positions], 1)
                self.role_position_counts[role] += len(np.unique(positions))
            touched = np.unique(triples.reshape(-1))
            self.any_role_positions += len(touched)
            self.any_role_ids.update(indices[touched].tolist())
            # Warmup relation-only mining has no evaluated hinge statistics.
            if hier_stats is None or "sample" not in hier_stats:
                self.graphs["sample"].update(sample_mining.get("stats", {}))

    def _id_coverage(self, counts):
        seen = counts > 0
        class_unique = np.bincount(self.train_labels[seen], minlength=self.num_classes)
        return {"unique_ids": int(seen.sum()), "train_split_size": len(self.train_ids),
            "fraction": _fraction(int(seen.sum()), len(self.train_ids)),
            "per_class_unique_ids": class_unique.tolist(),
            "per_class_fraction": [_fraction(int(n), int(size)) for n, size in zip(class_unique, self.class_sizes)]}

    def summary(self):
        class_correct = np.diag(self.confusion)
        class_total = self.confusion.sum(axis=1)
        accuracies = [_fraction(int(correct), int(total)) for correct, total in zip(class_correct, class_total)]
        observed_accuracies = [value for value in accuracies if value is not None]
        seen = self.draw_counts > 0
        scalar_summaries = {key: value.summary() for key, value in self.scalars.items()}
        coverage = {**self._id_coverage(self.draw_counts), "draws": self.positions,
            "repeated_draws": self.positions - int(seen.sum()),
            "repeat_fraction": _fraction(self.positions - int(seen.sum()), self.positions),
            "per_class_draws": self.class_draw_counts.tolist(), "per_class_dataset_sizes": self.class_sizes.tolist(),
            "id_draw_count_histogram_all_training_ids": {str(key): int(value) for key, value in sorted(Counter(self.draw_counts.tolist()).items())},
            "id_draw_counts": {str(int(sample_id)): int(count) for sample_id, count in zip(self.train_ids, self.draw_counts) if count}}
        geometry = {}
        for part, observations in self.geometry_values.items():
            counts = self.geometry_counts[part]
            geometry[part] = {key: _distribution(value) for key, value in observations.items()}
            geometry[part].update({"positions": counts["positions"],
                "near_native_numerical_boundary_threshold": .99599,
                "near_native_numerical_boundary_count": counts["near_native_numerical_boundary"],
                "near_native_numerical_boundary_fraction": _fraction(counts["near_native_numerical_boundary"], counts["positions"]),
                "shadow_tangent_cap_radius": 2.3, "shadow_tangent_cap_count": counts["shadow_tangent_cap2_3"],
                "shadow_tangent_cap_fraction": _fraction(counts["shadow_tangent_cap2_3"], counts["positions"]),
                "near_boundary_is_not_a_projection_trigger_count": True,
                "shadow_is_final_mu_logmap_counterfactual_not_live_extra_clip": True})
        geometry["hier_reported_per_batch"] = {key: value.summary() for key, value in self.proxy_geometry.items()} if self.proxy_geometry else None
        hierarchy = None
        if self.graphs["sample"].batches or self.graphs["proxy"].batches:
            eligible_class_counts = np.bincount(self.train_labels, weights=self.eligible_counts, minlength=self.num_classes).astype(np.int64)
            relation_class_draws = np.bincount(self.train_labels, weights=self.relation_draw_counts, minlength=self.num_classes).astype(np.int64)
            role_reports = {}
            for role, counts in self.role_counts.items():
                role_reports[role] = {**self._id_coverage(counts), "draw_occurrences": int(counts.sum()),
                    "per_class_draw_occurrences": np.bincount(self.train_labels, weights=counts, minlength=self.num_classes).astype(np.int64).tolist(),
                    "batch_positions_touched_sum": self.role_position_counts[role],
                    "pooled_relation_position_coverage": _fraction(self.role_position_counts[role], int(self.relation_draw_counts.sum()))}
            conditional = [_fraction(int(a), int(b)) for a, b in zip(eligible_class_counts, relation_class_draws)]
            conditional_present = [value for value in conditional if value is not None]
            hierarchy = {"relation_monitor_batches": self.relation_batches,
                "loss_evaluated_batches": self.loss_evaluated_batches,
                "objective_activity_known_batches": self.objective_activity_known_batches,
                "objective_active_batches": self.objective_active_batches if self.objective_activity_known_batches else None,
                "sample_graph": self.graphs["sample"].summary(), "proxy_graph": self.graphs["proxy"].summary(),
                "eligible_anchors": {**self._id_coverage(self.eligible_counts),
                    "positions": int(self.eligible_counts.sum()),
                    "per_class_positions": eligible_class_counts.tolist(),
                    "per_class_relation_input_positions": relation_class_draws.tolist(),
                    "per_class_fraction_given_relation_input": conditional,
                    "class_conditional_fraction_distribution": _distribution(conditional_present),
                    "eligible_id_position_counts": {str(int(sample_id)): int(count) for sample_id, count in zip(self.train_ids, self.eligible_counts) if count},
                    "per_id_eligibility_given_relation_input": {str(int(sample_id)): {
                        "input_positions": int(draws), "eligible_positions": int(eligible),
                        "eligible_batches": int(batches), "fraction": float(eligible / draws)}
                        for sample_id, draws, eligible, batches in zip(self.train_ids, self.relation_draw_counts, self.eligible_counts, self.eligible_batch_counts) if draws},
                    "conditional_on_observed_epoch_draws": True},
                "roles": role_reports,
                "any_role": {"unique_ids": len(self.any_role_ids),
                    "train_split_fraction": _fraction(len(self.any_role_ids), len(self.train_ids)),
                    "batch_positions_touched_sum": self.any_role_positions,
                    "pooled_relation_position_coverage": _fraction(self.any_role_positions, int(self.relation_draw_counts.sum()))},
                "warmup_loss_evaluation_is_not_an_optimization_step": True}
        return {"batches": self.batches, "positions": self.positions,
            "training": {"oa": _fraction(int(class_correct.sum()), int(class_total.sum())),
                "aa_observed_classes": float(np.mean(observed_accuracies)) if observed_accuracies else None,
                "observed_classes": int((class_total > 0).sum()), "per_class_accuracy": accuracies,
                "per_class_correct": class_correct.tolist(), "per_class_positions": class_total.tolist(),
                "confusion_matrix": self.confusion.tolist(), "accuracy_unit": "fraction, training-mode sampled positions"},
            "sample_coverage": coverage, "geometry": geometry,
            "scalars": scalar_summaries, "hierarchy": hierarchy,
            "finite": True, "stores_detached_numeric_observations_only": True}
