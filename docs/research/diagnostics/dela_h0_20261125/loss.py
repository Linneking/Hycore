"""DMR pair-distance losses for the DeLA-H0 diagnostic."""

import math
from operator import index

import torch


def pair_distance_targets(m: int, *, n: int = 1024) -> tuple[float, float]:
    """Return the whole–part and equal-size part–part targets in the unit ball."""
    m, n = index(m), index(n)
    if not 0 < m < n:
        raise ValueError("Expected a nonempty part smaller than its whole.")

    whole_depth = math.log1p(2 * n)
    part_depth = math.log1p(2 * m / (n + 1 - m))
    return whole_depth - part_depth, 2 * part_depth


def hyperbolic_pair_losses(
    z_w: torch.Tensor, z_p: torch.Tensor, m: int, row_ids, ball, *, n: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply DMR Eqs. (12–14) to aligned whole/part rows using reverse pairing.

    Each row comes from one original object and uses the same part point count.
    The caller supplies a Poincaré ball of curvature −1 and weights the two
    returned batch means together in its training objective.
    """
    if z_w.ndim != 2 or z_w.shape != z_p.shape:
        raise ValueError("Whole and part embeddings must have matching [B, D] shapes.")
    batch_size = z_w.shape[0]
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Reverse pairing requires a positive even batch size.")

    if isinstance(row_ids, torch.Tensor):
        if row_ids.shape != (batch_size,):
            raise ValueError("Expected one object ID per embedding row.")
        ids = row_ids.tolist()
    else:
        ids = tuple(row_ids)
        if len(ids) != batch_size:
            raise ValueError("Expected one object ID per embedding row.")
    if any(row_id == negative_id for row_id, negative_id in zip(ids, reversed(ids))):
        raise ValueError("Reverse pairing matched two rows from the same object.")

    whole_part_target, part_part_target = pair_distance_targets(m, n=n)
    if z_w.dtype in (torch.float16, torch.bfloat16):
        z_w = z_w.float()
    if z_p.dtype in (torch.float16, torch.bfloat16):
        z_p = z_p.float()
    whole_part_dist = ball.dist(z_w, z_p)
    part_part_dist = ball.dist(z_p, z_p.flip(0))

    l_h = (whole_part_dist - whole_part_target).abs().mean()
    l_c = (part_part_dist - part_part_target).abs().mean()
    return l_h, l_c
