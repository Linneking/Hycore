"""Geometry, gradient and denominator contracts for frozen audit95."""
from __future__ import annotations

import math
import numpy as np
import torch

from docs.research.diagnostics.parthier_embedding_audit94_v1.metrics import array, describe
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import Artanh


SCALES = (200, 400, 600)
STATE_NAMES = ('wrong_or_tied', 'ordered_short', 'margin_pass')


def states(gaps, margin):
    x = np.asarray(gaps)
    if not np.isfinite(x).all():
        raise ValueError('Undefined gaps cannot disappear from a denominator')
    threshold = np.asarray(margin, dtype=x.dtype)
    return np.where(x <= 0, 0, np.where(x < threshold, 1, 2)).astype(np.int8)


def angles(a, b):
    """Aligned original-coordinate directions, computed in FP64 degrees."""
    a, b = a.double(), b.double()
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    defined = (an > 1e-6) & (bn > 1e-6)
    cosine = (a * b).sum(-1) / (an * bn).clamp_min(1e-30)
    result = torch.rad2deg(cosine.clamp(-1, 1).acos())
    return torch.where(defined, result, torch.full_like(result, float('nan')))


def angle_matrix(a, b):
    a, b = a.double(), b.double()
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    cosine = (a / an.clamp_min(1e-30)[:, None]) @ (b / bn.clamp_min(1e-30)[:, None]).T
    return torch.where((an[:, None] > 1e-6) & (bn[None, :] > 1e-6),
                       torch.rad2deg(cosine.clamp(-1, 1).acos()), torch.full_like(cosine, float('nan')))


def midpoint(a, b):
    """Continuous c=1 geodesic midpoint via the normalized Lorentz sum."""
    a, b = a.double(), b.double()
    ra, rb = a.norm(dim=-1), b.norm(dim=-1)
    if bool((ra >= 1).any() or (rb >= 1).any()):
        raise ValueError('Midpoint requires ball-interior points')
    ta, tb = (1 + ra.square()) / (1 - ra.square()), (1 + rb.square()) / (1 - rb.square())
    spatial = 2 * a / (1 - ra.square())[:, None] + 2 * b / (1 - rb.square())[:, None]
    temporal = ta + tb
    lorentz_norm = (temporal.square() - spatial.square().sum(-1)).clamp_min(1e-24).sqrt()
    point = spatial / (temporal + lorentz_norm)[:, None]
    h = 2 * point.norm(dim=-1).clamp(max=1-1e-14).atanh()
    return point, h


def aligned_official_distance(x, y):
    """Elementwise literal production Mobius distance, including its epsilon."""
    xy = -(x * y).sum(-1)
    x2, y2 = x.square().sum(-1), y.square().sum(-1)
    numerator = (1 + 2 * xy + y2)[:, None] * (-x) + (1-x2)[:, None] * y
    difference = numerator / (1 + 2 * xy + x2*y2 + 1e-5)[:, None]
    return 2 * Artanh.apply(difference.norm(dim=-1))


def direction_rate(w, p, gw, gp):
    """d(angle)/dt for joint output-coordinate velocities (-gw,-gp)."""
    rw, rp = w.norm(dim=-1), p.norm(dim=-1)
    uw, up = w / rw.clamp_min(1e-30)[:, None], p / rp.clamp_min(1e-30)[:, None]
    cosine = (uw*up).sum(-1).clamp(-1, 1)
    sine = (1-cosine.square()).clamp_min(0).sqrt()
    cosine_rate = -((up-cosine[:, None]*uw)*gw).sum(-1)/rw.clamp_min(1e-30)
    cosine_rate -= ((uw-cosine[:, None]*up)*gp).sum(-1)/rp.clamp_min(1e-30)
    valid = (rw > 1e-6) & (rp > 1e-6) & (sine > 1e-8)
    rate = -cosine_rate / sine.clamp_min(1e-30) * (180/math.pi)
    return torch.where(valid, rate, torch.full_like(rate, float('nan')))


def gradient_probe(whole, part, count):
    """Only detached output leaves; preserve positive/flip-negative sharing."""
    from inter_hierarchy_MN40.hier_proxy_scratch_v9.base_protocol import intra_losses
    with torch.enable_grad():
        w = whole.detach().double().clone().requires_grad_(True)
        p = part.detach().double().clone().requires_grad_(True)
        losses = intra_losses(w, p, count)
        result = {}
        for term in ('contrastive', 'radial'):
            loss = .01 * losses['intra_'+term]
            gw, gp = torch.autograd.grad(loss, (w, p), retain_graph=(term=='contrastive'))
            urw = w.detach() / w.detach().norm(dim=-1, keepdim=True)
            urp = p.detach() / p.detach().norm(dim=-1, keepdim=True)
            rw, rp = (gw*urw).sum(-1), (gp*urp).sum(-1)
            tw, tp = (gw-rw[:, None]*urw).norm(dim=-1), (gp-rp[:, None]*urp).norm(dim=-1)
            result[term] = {k: array(v) for k,v in {
                'angle_rate': direction_rate(w.detach(), p.detach(), gw, gp),
                'whole_radial_gradient': rw, 'part_radial_gradient': rp,
                'whole_tangent_norm': tw, 'part_tangent_norm': tp,
                'whole_gradient_norm': gw.norm(dim=-1), 'part_gradient_norm': gp.norm(dim=-1)}.items()}
            result[term]['weighted_batch_loss'] = float(loss.detach())
        return result


def gradient_summary(columns):
    out = {k: describe(v) for k,v in columns.items()}
    rate = np.asarray(columns['angle_rate'])
    valid = np.isfinite(rate)
    r = rate[valid]
    out.update(total=len(rate), defined=int(valid.sum()), undefined=int((~valid).sum()),
        increase_fraction=float((r > 1e-10).mean()) if len(r) else None,
        decrease_fraction=float((r < -1e-10).mean()) if len(r) else None,
        stationary_fraction=float((np.abs(r) <= 1e-10).mean()) if len(r) else None)
    return out


def feasible_pair(distances, margin=.1):
    """Exact exhaustive HIER-only feasibility in this finite, fixed pool."""
    d = torch.as_tensor(distances)
    if d.ndim != 2 or d.shape[0] != 3:
        raise ValueError('Expected three endpoint-to-candidate rows')
    n = d.shape[1]
    good = ((d[0][None,:]-d[0][:,None]) >= margin)
    good &= ((d[1][None,:]-d[1][:,None]) >= margin)
    good &= ((d[2][:,None]-d[2][None,:]) >= margin)
    good.fill_diagonal_(False)
    number = int(good.sum())
    if not number:
        return {'feasible_pairs': 0, 'exists': False, 'a': None, 'b': None}
    ca = d[:2].amax(0)
    cb = d.amax(0)
    scores = (ca[:,None]+cb[None,:]).masked_fill(~good, float('inf'))
    flat = int(scores.argmin())
    a,b = divmod(flat,n)
    return {'feasible_pairs': number, 'exists': True, 'a': a, 'b': b,
            'hier_gaps': array(torch.stack((d[0,b]-d[0,a],d[1,b]-d[1,a],d[2,a]-d[2,b]))).tolist()}


class Columns:
    def __init__(self): self.data = {}
    def add(self, **items):
        lengths = {len(np.asarray(array(v))) for v in items.values()}
        if len(lengths) != 1: raise ValueError('Unaligned row columns')
        for key,value in items.items(): self.data.setdefault(key,[]).append(array(value))
    def finish(self): return {k:np.concatenate(v,axis=0) for k,v in self.data.items()}


def hotspot(counts, expected, exposures):
    counts, expected, exposures = map(lambda x:np.asarray(x,dtype=np.float64),(counts,expected,exposures))
    exposed = exposures > 0
    a,e = counts[exposed], expected[exposed]
    n = len(a)
    k = max(1,math.ceil(.05*n))
    def concentration(x):
        total=float(x.sum())
        return {'total':total,'top1_fraction':float(x.max()/total) if total else None,
                'top5pct_fraction':float(np.sort(x)[-k:].sum()/total) if total else None}
    return {'exposed_entries':n,'used_entries':int((a>0).sum()),'top5pct_entry_count':k,
            'actual':concentration(a),'expected':concentration(e),
            'per_million_exposure':describe(1e6*a/exposures[exposed])}
