"""Read-only NumPy geometry for explicitly declared Poincare-ball caches.

Distances are full, unsquared geodesic distances. Coordinate radius, normalized
radius, origin distance and tangent norm are deliberately separate quantities.
No training module, GPU runtime or mutable model implementation is imported.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

QUANTILE_LEVELS = (0., .05, .25, .5, .75, .95, 1.)
QUANTILE_KEYS = ("p00", "p05", "p25", "p50", "p75", "p95", "p100")


def quantiles(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {key: None for key in QUANTILE_KEYS}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value in geometry summary")
    return dict(zip(QUANTILE_KEYS, map(float, np.quantile(values, QUANTILE_LEVELS))))


def curvature(c):
    c = float(c)
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Curvature magnitude c must be finite and positive")
    return c


def matrix(values, name="coordinates"):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite N x D matrix")
    return result


def ball_geometry(points, c):
    points = matrix(points)
    c = curvature(c)
    radius = np.linalg.norm(points, axis=1)
    normalized_radius = np.sqrt(c) * radius
    if np.any(normalized_radius >= 1):
        raise ValueError("Cache contains a point outside the open Poincare ball")
    depth = 2. / np.sqrt(c) * np.arctanh(normalized_radius)
    direction = np.zeros_like(points)
    np.divide(points, radius[:, None], out=direction, where=radius[:, None] > 0)
    return {"points": points, "radius": radius, "normalized_radius": normalized_radius,
            "depth": depth, "mapped_tangent_norm": depth / 2., "direction": direction,
            "direction_defined": radius > 0}


def expmap0(tangent, c, numeric_radius_fraction=None, source_tangent_cap=None,
            source_cap_epsilon=0.):
    """Explicit forward mapping; never infer a cap from curvature or a filename.

    source_cap_epsilon=1e-5 reproduces released HIER's denominator convention.
    V7's post-step parameter constraint is NOT applied here: stored parameters
    must be measured as saved, including any constraint violation.
    """
    tangent = matrix(tangent, "tangent parameters")
    c = curvature(c)
    norm = np.linalg.norm(tangent, axis=1)
    effective = norm.copy()
    if source_tangent_cap is not None:
        cap = float(source_tangent_cap)
        epsilon = float(source_cap_epsilon)
        if cap <= 0 or epsilon < 0:
            raise ValueError("Invalid declared tangent cap")
        effective *= np.minimum(1., cap / np.maximum(norm + epsilon, np.finfo(float).tiny))
    normalized = np.tanh(np.sqrt(c) * effective)
    if numeric_radius_fraction is not None:
        fraction = float(numeric_radius_fraction)
        if not 0 < fraction < 1:
            raise ValueError("numeric_radius_fraction must lie strictly between 0 and 1")
        normalized = np.minimum(normalized, fraction)
    direction = np.zeros_like(tangent)
    np.divide(tangent, norm[:, None], out=direction, where=norm[:, None] > 0)
    points = direction * (normalized / np.sqrt(c))[:, None]
    ball_geometry(points, c)  # fail rather than silently clipping undeclared saturation
    return points


def poincare_distance(x, y, c):
    """Stable general-c asinh distance; validates the open-ball domain."""
    gx, gy = ball_geometry(x, c), ball_geometry(y, c)
    x, y = gx["points"], gy["points"]
    if x.shape[1] != y.shape[1]:
        raise ValueError("Embedding dimension mismatch")
    # Avoid subtracting nearly equal squared norms for nearly coincident points.
    squared = np.zeros((len(x), len(y)), dtype=np.float64)
    for dimension in range(x.shape[1]):
        delta = x[:, dimension, None] - y[None, :, dimension]
        squared += delta * delta
    denominator = ((1. - gx["normalized_radius"] ** 2)[:, None] *
                   (1. - gy["normalized_radius"] ** 2)[None, :])
    return 2. / np.sqrt(float(c)) * np.arcsinh(np.sqrt(float(c) * squared / denominator))


def direction_distance(x, y, c):
    gx, gy = ball_geometry(x, c), ball_geometry(y, c)
    distances = 1. - np.clip(gx["direction"] @ gy["direction"].T, -1., 1.)
    valid = gx["direction_defined"][:, None] & gy["direction_defined"][None, :]
    return np.where(valid, distances, np.inf)


def unique_ids(values, name="sample_ids"):
    ids = np.asarray(values).reshape(-1)
    if ids.dtype.kind not in "iuUS":
        raise ValueError(name + " must contain integer or string identities, never objects/floats")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(name + " contains repeated identities")
    return ids


def pool_hash(ids):
    ids = unique_ids(ids)
    # Sorting makes cache row order irrelevant. Explicit type avoids 1 == '1'.
    canonical = {"kind": "integer" if ids.dtype.kind in "iu" else "string",
                 "ids": np.sort(ids).tolist()}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def topk_neighbours(queries, candidates, candidate_ids, c, k=4,
                    metric="hyperbolic", chunk_size=32, equal_radius_fraction=None):
    queries, candidates = matrix(queries), matrix(candidates)
    candidate_ids = unique_ids(candidate_ids)
    if len(candidate_ids) != len(candidates):
        raise ValueError("Candidate identities and coordinates have different lengths")
    if not 1 <= int(k) <= len(candidates):
        raise ValueError("topk is larger than the declared candidate pool")
    order = np.argsort(candidate_ids, kind="stable")
    candidates = candidates[order]
    if metric == "equal_radius_hyperbolic":
        if equal_radius_fraction is None or not 0 < float(equal_radius_fraction) < 1:
            raise ValueError("Equal-radius control requires explicit radius fraction")
        queries = ball_geometry(queries, c)["direction"] * float(equal_radius_fraction) / np.sqrt(c)
        candidates = ball_geometry(candidates, c)["direction"] * float(equal_radius_fraction) / np.sqrt(c)
        distance = poincare_distance
    elif metric == "hyperbolic":
        distance = poincare_distance
    elif metric == "direction":
        distance = direction_distance
    else:
        raise ValueError("Unknown retrieval metric: " + str(metric))
    rows, values = [], []
    for start in range(0, len(queries), int(chunk_size)):
        distances = distance(queries[start:start + int(chunk_size)], candidates, c)
        selected = np.argsort(distances, axis=1, kind="stable")[:, :int(k)]
        rows.append(order[selected])
        values.append(np.take_along_axis(distances, selected, axis=1))
    if not rows:
        return np.empty((0, int(k)), dtype=np.int64), np.empty((0, int(k)))
    return np.concatenate(rows), np.concatenate(values)


def align_rows(previous_ids, current_ids):
    previous_ids, current_ids = unique_ids(previous_ids), unique_ids(current_ids)
    if pool_hash(previous_ids) != pool_hash(current_ids):
        raise ValueError("Identity pools differ; turnover cannot be measured on unmatched pools")
    lookup = {value: index for index, value in enumerate(current_ids.tolist())}
    return np.array([lookup[value] for value in previous_ids.tolist()], dtype=np.int64)


def displacement(previous, current, c):
    old, new = ball_geometry(previous, c), ball_geometry(current, c)
    if old["points"].shape != new["points"].shape:
        raise ValueError("Aligned coordinate shape mismatch")
    valid = old["direction_defined"] & new["direction_defined"]
    cosine = np.sum(old["direction"] * new["direction"], axis=1)
    angles = np.arccos(np.clip(cosine[valid], -1., 1.))
    return {"radial_depth_change": quantiles(new["depth"] - old["depth"]),
            "angular_displacement_rad": quantiles(angles),
            "angular_valid_count": int(valid.sum()), "count": len(previous),
            "coordinate_displacement": quantiles(np.linalg.norm(new["points"] - old["points"], axis=1))}


def topk_stability(previous_ids, current_ids):
    previous_ids, current_ids = np.asarray(previous_ids), np.asarray(current_ids)
    if previous_ids.shape != current_ids.shape or previous_ids.ndim != 2:
        raise ValueError("Aligned topk arrays must have the same P x k shape")
    jaccard, turnover, same = [], [], []
    for old, new in zip(previous_ids, current_ids):
        a, b = set(old.tolist()), set(new.tolist())
        jaccard.append(len(a & b) / len(a | b))
        turnover.append(1. - len(a & b) / len(a))
        same.append(a == b)
    return {"mean_topk_jaccard": float(np.mean(jaccard)) if jaccard else None,
            "unchanged_fraction": float(np.mean(same)) if same else None,
            "turnover": float(np.mean(turnover)) if turnover else None,
            "jaccard_quantiles": quantiles(jaccard), "proxy_count": len(jaccard)}
