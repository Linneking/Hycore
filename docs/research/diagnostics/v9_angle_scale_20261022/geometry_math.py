"""Exact ideal-geometry pair/triple centers for a common hyperbolic shell.

This module is independent of HIER candidate selection and its numerical
distance operator. It does not estimate a learned proxy's optimal depth.
"""

from __future__ import annotations

import numpy as np


def pair_midpoint_depth(cosine, *, depth: float = 4.6, c: float = 0.1):
    """Depth of a geodesic midpoint of two points on the same shell."""
    if c <= 0 or depth < 0:
        raise ValueError("Expected positive curvature magnitude and nonnegative depth")
    cosine = np.asarray(cosine, dtype=np.float64)
    if not np.isfinite(cosine).all():
        raise ValueError("Nonfinite pair cosine")
    half_cosine = np.sqrt((1 + np.clip(cosine, -1, 1)) / 2)
    root_c = np.sqrt(c)
    return np.arctanh(np.tanh(root_c * depth) * half_cosine) / root_c


def triple_center_from_gram(gram, *, depth: float = 4.6, c: float = 0.1,
                            tolerance: float = 1e-10):
    """Exact same-shell three-point hyperbolic minimax center.

    Input has shape (..., 3, 3), diagonal 1, and is a Gram matrix of unit
    directions. The returned weights form a convex combination of those
    directions. A zero norm means that the minimax center is the origin.
    Vertices and edges handle boundary solutions; KKT handles the full face.
    Eigenvalue validation is deliberately omitted for large batched audits;
    callers must build the Gram matrix from actual normalized directions.
    """
    if c <= 0 or depth < 0:
        raise ValueError("Expected positive curvature magnitude and nonnegative depth")
    original = np.asarray(gram, dtype=np.float64)
    if original.shape[-2:] != (3, 3) or not np.isfinite(original).all():
        raise ValueError("Expected finite (...,3,3) Gram matrix")
    if not np.allclose(original, np.swapaxes(original, -1, -2), atol=1e-7, rtol=0):
        raise ValueError("Gram matrix is not symmetric")
    if not np.allclose(np.diagonal(original, axis1=-2, axis2=-1), 1,
                       atol=1e-7, rtol=0):
        raise ValueError("Gram diagonal must be 1")
    shape = original.shape[:-2]
    flat = original.reshape(-1, 3, 3)
    count = len(flat)
    basic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], dtype=float)
    weights = np.zeros((count, 7, 3), dtype=float)
    weights[:, :6] = basic
    kkt = np.zeros((count, 4, 4), dtype=float)
    kkt[:, :3, :3] = flat
    kkt[:, :3, 3] = 1
    kkt[:, 3, :3] = 1
    rhs = np.broadcast_to([0., 0., 0., 1.], (count, 4))
    nonsingular = np.abs(np.linalg.det(kkt)) > 1e-12
    solutions = np.empty((count, 4), dtype=float)
    if nonsingular.any():
        solutions[nonsingular] = np.linalg.solve(kkt[nonsingular],
                                                 rhs[nonsingular, :, None])[..., 0]
    for idx in np.flatnonzero(~nonsingular):
        solutions[idx] = np.linalg.lstsq(kkt[idx], rhs[idx], rcond=1e-13)[0]
    face = solutions[:, :3]
    residual = np.max(np.abs(np.einsum('bij,bj->bi', kkt, solutions) - rhs), axis=1)
    face_valid = (face.min(axis=1) >= -tolerance) & (residual < 1e-8)
    face = np.maximum(face, 0)
    face /= np.maximum(face.sum(axis=1, keepdims=True), 1e-300)
    weights[:, 6] = face
    norms_sq = np.einsum('bki,bij,bkj->bk', weights, flat, weights)
    norms_sq[:, 6] = np.where(face_valid, norms_sq[:, 6], np.inf)
    chosen = np.argmin(norms_sq, axis=1)
    alpha = weights[np.arange(count), chosen]
    norm_sq = np.clip(norms_sq[np.arange(count), chosen], 0, 1)
    support_cosine = np.sqrt(norm_sq)
    root_c = np.sqrt(c)
    center_depth = np.arctanh(np.tanh(root_c * depth) * support_cosine) / root_c
    cosh_radius = (np.cosh(root_c * depth) * np.cosh(root_c * center_depth)
                   - np.sinh(root_c * depth) * np.sinh(root_c * center_depth)
                   * support_cosine)
    radius = np.arccosh(np.maximum(cosh_radius, 1)) / root_c
    # The Euclidean projection inequality independently certifies optimality:
    # every n_i dot v must be at least ||v||^2.
    projection_residual = np.maximum(norm_sq - np.einsum('bij,bj->bi', flat, alpha).min(axis=1), 0)
    return {
        'weights': alpha.reshape(shape + (3,)),
        'support_cosine': support_cosine.reshape(shape),
        'depth': center_depth.reshape(shape),
        'radius': radius.reshape(shape),
        'active_count': (alpha > 1e-8).sum(axis=1).reshape(shape),
        'projection_residual': projection_residual.reshape(shape),
    }


def triple_center(directions, *, depth: float = 4.6, c: float = 0.1):
    """As above, accepting actual (...,3,D) directions and normalizing them."""
    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape[-2] != 3:
        raise ValueError("Expected (...,3,D) directions")
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise ValueError("Directions must be nonzero and finite")
    unit = directions / norms
    result = triple_center_from_gram(unit @ np.swapaxes(unit, -1, -2),
                                     depth=depth, c=c)
    vector = np.einsum('...i,...ij->...j', result['weights'], unit)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    result['direction'] = np.divide(vector, length, out=np.zeros_like(vector),
                                    where=length > 1e-12)
    return result
