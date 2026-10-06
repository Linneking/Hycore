"""Bounded CPU proxy safety comparison, with frozen clean-eval whole coordinates.

Every arm uses common fixed sample/proxy triples and independent seeded sample
and proxy Gumbel draws. This isolates the proxy mapping and AdamW displacement;
it does not replay training-mode BN, backbone updates or dynamic proxy mining.
No GPU is used and source files remain immutable. Use a new output directory.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from inter_hierarchy_MN40.hier_proxy_scratch_v5.relations import poincare_distance, mine_sample_triplets, mine_reciprocal_triplets
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
from inter_hierarchy_MN40.hier_proxy_scratch_v6.hier_loss import ghhc_loss, expmap0_c1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(u, p):
    s = u.detach().norm(dim=-1).double()
    r = p.detach().norm(dim=-1).double()
    return {'tangent_min': float(s.min()), 'tangent_median': float(s.median()), 'tangent_max': float(s.max()),
            'ball_min': float(r.min()), 'ball_median': float(r.median()), 'ball_max': float(r.max()),
            'depth_median': float((2*torch.atanh(r)).median()),
            'numerical_saturation_count': int((torch.tanh(s) > .999).sum())}


def mapped(u, arm):
    if arm == 'source_forward_cap_hook':
        norm = u.norm(dim=-1, keepdim=True) + 1e-5
        p = expmap0_c1(u * (2.3/norm).clamp(max=1))
        # Equivalent to the released RiemannianGradient at c=1, applied to p.
        if p.requires_grad:
            p.register_hook(lambda gradient: gradient * (1-p.detach().square().sum(-1, keepdim=True)).square()/4)
        return p
    return expmap0_c1(u)


def clamp_parameter(u, cap=3.):
    with torch.no_grad():
        norms = u.norm(dim=-1, keepdim=True)
        count = int((norms > cap).sum())
        u.mul_((cap/norms.clamp_min(1e-15)).clamp(max=1))
    return count


def self_test():
    u = torch.tensor([[5., 0.], [2., 0.]], requires_grad=True)
    p = expmap0_c1(u)
    grad = torch.autograd.grad(p[:, 0].sum(), u)[0]
    assert abs(float(grad[0, 0])) < 1e-6
    assert float(grad[1, 0]) > 0
    clamp_parameter(u)
    assert float(u.norm(dim=-1).max()) <= 3.00001
    p = expmap0_c1(u)
    grad = torch.autograd.grad(p[:, 0].sum(), u)[0]
    assert float(grad[0, 0]) > 0
    ordinary = torch.tensor([[2., 0.]], requires_grad=True)
    hooked = ordinary.detach().clone().requires_grad_()
    grad1 = torch.autograd.grad(expmap0_c1(ordinary).sum(), ordinary)[0]
    grad2 = torch.autograd.grad(mapped(hooked, 'source_forward_cap_hook').sum(), hooked)[0]
    ball = expmap0_c1(ordinary).detach()
    expected = grad1 * (1-ball.square().sum(-1, keepdim=True)).square()/4
    torch.testing.assert_close(grad2, expected, atol=2e-6, rtol=2e-5)
    print(json.dumps({'self_test': 'passed', 'checks': ['saturated radial derivative', 'parameter clamp recovers nonzero derivative', 'source output gradient hook placement']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--t-per-anchor', type=int, default=10)
    parser.add_argument('--init-depth', type=float, default=5.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test(); return
    if args.cache is None or args.output_dir is None or not 1 <= args.steps <= 64 or not 1 <= args.t_per_anchor <= 50:
        parser.error('Require cache, new output-dir, steps1..64, T1..50')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache_sha = digest(args.cache)
    with np.load(args.cache, allow_pickle=False) as cache:
        whole = torch.from_numpy(cache['mu'].copy()).float()
        labels = torch.from_numpy(cache['labels'].reshape(-1).copy()).long()
        ids = torch.from_numpy(cache['sample_ids'].reshape(-1).copy()).long()
    assert whole.shape == (8856, 256) and len(torch.unique(ids)) == 8856
    sampler = SourceClassBatchSampler(labels.tolist(), seed=22, steps=args.steps, rank=0)
    sampler.set_epoch(0)
    generator = torch.Generator().manual_seed(22)
    initial = torch.randn(512, 256, generator=generator)
    initial *= args.init_depth/2 / initial.norm(dim=-1, keepdim=True)
    initial_proxy = expmap0_c1(initial)
    # Fixed graph/triples explicitly isolate mapping and optimizer mechanisms.
    proxy_triplets = mine_reciprocal_triplets(torch.exp(-poincare_distance(initial_proxy)),
        topk=20, t_per_anchor=args.t_per_anchor, seed=23, exclude_self_negative=True)['triplets']
    plans = []
    for index, plan in enumerate(sampler.iter_global_plans()):
        rows = torch.tensor(plan.indices)
        x, y, data_ids = whole[rows], labels[rows], ids[rows]
        triples = mine_sample_triplets(x, y, topk=20, t_per_anchor=args.t_per_anchor,
            seed=22+index*10007, exclude_self_negative=True, data_ids=data_ids)['triplets']
        plans.append((x, triples))
    results = {}
    for arm, lr in [('legacy_unbounded', .01), ('source_forward_cap_hook', .01),
                    ('parameter_cap_depth6', .01), ('parameter_cap_depth6_lr002', .002)]:
        u = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([u], lr=lr, weight_decay=.01)
        snapshots, projection_events = [], 0
        sample_counts, proxy_counts = Counter(), Counter()
        start_p = mapped(u, arm).detach().clone()
        for step, (x, sample_triplets) in enumerate(plans):
            optimizer.zero_grad(set_to_none=True)
            before_u, before_p = u.detach().clone(), mapped(u, arm).detach().clone()
            p = mapped(u, arm)
            sample, sample_stats, sample_details = ghhc_loss(poincare_distance(x, p), sample_triplets,
                tau=.1, margin=.1, generator=torch.Generator().manual_seed(220000+step), return_details=True)
            proxy, proxy_stats, proxy_details = ghhc_loss(poincare_distance(p), proxy_triplets,
                tau=.1, margin=.1, generator=torch.Generator().manual_seed(230000+step), return_details=True)
            loss = .1*(sample+proxy)
            loss.backward()
            assert bool(torch.isfinite(loss)) and bool(torch.isfinite(u.grad).all())
            radial_grad = (u.grad*torch.nn.functional.normalize(u.detach(), dim=-1)).sum(-1)
            gradient_norm = float(u.grad.norm())
            optimizer.step()
            unconstrained = u.detach().clone()
            if arm.startswith('parameter_cap'):
                projection_events += clamp_parameter(u, 3.)
            after_p = mapped(u, arm).detach()
            assert bool(torch.isfinite(u).all()) and bool(torch.isfinite(after_p).all())
            for kind, details, counts in [('sample', sample_details, sample_counts), ('proxy', proxy_details, proxy_counts)]:
                counts.update(details['pair_proxy_idx'].tolist()); counts.update(details['triple_proxy_idx'].tolist())
            snapshots.append({'step': step+1, **describe(u, after_p), 'sample_loss': float(sample.detach()),
                'proxy_loss': float(proxy.detach()), 'sample_active': sample_stats['active_triplets'],
                'proxy_active': proxy_stats['active_triplets'], 'gradient_norm': gradient_norm,
                'radial_gradient_mean': float(radial_grad.mean()),
                'Adam_parameter_displacement_median': float((unconstrained-before_u).norm(dim=-1).median()),
                'mapped_displacement_median': float((after_p-before_p).norm(dim=-1).median()),
                'tangent_outward_rows': int((u.detach().norm(dim=-1)>before_u.norm(dim=-1)).sum())})
        results[arm] = {'lr': lr, 'weight': .1, 'initial': describe(initial, start_p), 'final': snapshots[-1],
            'trajectory': snapshots, 'parameter_projection_events': projection_events,
            'sample_ancestor_unique': len(sample_counts), 'proxy_ancestor_unique': len(proxy_counts),
            'initial_to_final_mapped_displacement_median': float((mapped(u, arm).detach()-start_p).norm(dim=-1).median())}
        print(json.dumps({'arm': arm, 'final': snapshots[-1], 'projection_events': projection_events}), flush=True)
    recovery = None
    checkpoint_sha = None
    if args.checkpoint:
        checkpoint_sha = digest(args.checkpoint)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        old_u = next(v for k, v in saved['proxy'].items() if k.endswith('tangent_proxies')).float()
        old_p = expmap0_c1(old_u)
        projected = old_u.clone()
        touched = clamp_parameter(projected, 3.)
        p = expmap0_c1(projected)
        if bool(torch.isfinite(p).all()):
            test_u = projected.clone().requires_grad_()
            derivative = torch.autograd.grad(expmap0_c1(test_u).norm(dim=-1).sum(), test_u)[0]
            signed = (derivative*torch.nn.functional.normalize(test_u.detach(), dim=-1)).sum(-1)
            recovery = {'source': describe(old_u, old_p), 'post_parameter_cap': describe(projected, p),
                'touched': touched, 'mapped_displacement_median': float((p-old_p).norm(dim=-1).median()),
                'radial_derivative_min_after_cap': float(signed.min()),
                'note': 'No historical optimizer continuation; diagnostic geometry intervention only.'}
        assert digest(args.checkpoint) == checkpoint_sha
    assert digest(args.cache) == cache_sha
    report = {'format': 'hier-v7-proxy-safety-probe-v1', 'status': 'complete', 'cache_sha256': cache_sha,
        'checkpoint_sha256': checkpoint_sha, 'script_sha256': digest(__file__),
        'steps_per_arm': args.steps, 'total_optimizer_steps': args.steps*4, 'T': args.t_per_anchor,
        'P': 512, 'D': 256, 'init_depth': args.init_depth, 'seed': 22, 'device': 'cpu',
        'source_files_unchanged': True, 'arms': results, 'recovery_geometry': recovery,
        'wall_seconds': time.perf_counter()-started,
        'limitations': ['Frozen clean eval whole; no backbone or trainBN update.',
            'Common proxy triples are fixed from the initial geometry; production re-mines every step.',
            'Diagnostic T may be reduced; not a full source training replay or performance selection.',
            'Source arm reproduces cap/hook mechanism at c1, retaining current exact distance.',
            'Source forward cap2.3 changes initial depth5 to4.6; this is explicitly a different effective start.',
            'Fresh AdamW in each arm; no historical optimizer momentum restored.',
            'Short bounded mechanism test cannot establish e300 safety or independent hierarchy validity.']}
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'wall_seconds': report['wall_seconds']}), flush=True)

if __name__ == '__main__':
    main()
