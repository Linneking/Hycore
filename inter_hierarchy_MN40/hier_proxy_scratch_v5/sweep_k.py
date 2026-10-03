"""K diagnostics on saved TRAINING features; no backbone training or test data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .hier_loss import HIERLoss
from .relations import mine_sample_triplets


def examine_case(case, proxy, ks=(3, 6, 8, 10, 20), seed=22):
    mu, labels, ids = case['mu'], case['gold'], case['ids']
    rows = []
    for k in ks:
        gen = torch.Generator(device=mu.device).manual_seed(seed)
        with torch.no_grad():
            _, stats = proxy(mu, labels, topk=k, seed=seed, data_ids=ids, generator=gen)
        rows.append({'stage': case['stage'], 'step': case['step'], 'k': k,
                     'proxy_k': k, **stats})
    return rows


def self_negative_audit(case, proxy, k=8, seed=22):
    """Same live draw: measure the self-k contribution, not a fresh noise draw."""
    mu = case['mu'].detach().clone().requires_grad_(True)
    mined = mine_sample_triplets(mu, case['gold'], topk=k, seed=seed,
                                 data_ids=case['ids'])
    loss, stats, detail = proxy.sample_loss(
        mu, mined['triplets'], generator=torch.Generator(device=mu.device).manual_seed(seed),
        return_details=True)
    total_g = torch.autograd.grad(loss, mu, retain_graph=True)[0]
    if len(mined['triplets']):
        contribution = (detail['per_draw_loss'] * detail['self_k_mask']).sum() / len(mined['triplets'])
        self_g = torch.autograd.grad(contribution, mu)[0]
    else:
        self_g = torch.zeros_like(mu)
    return {'stage': case['stage'], 'step': case['step'], 'k': k,
            **mined['stats'], **stats, 'sample_loss': float(loss.detach()),
            'sample_mu_gradient_norm': float(total_g.norm()),
            'self_k_mu_gradient_norm': float(self_g.norm()),
            'self_k_mu_gradient_ratio': float(self_g.norm() / total_g.norm().clamp_min(1e-30))}


def proxy_lr_audit(case, *, rates=(0.0017777778, 0.005, 0.01, 0.02), steps=12,
                   seed=22, count=512, k=8):
    """Bounded proxy-only updates on the same detached features and initialization."""
    reports = []
    for rate in rates:
        proxy = HIERLoss(count, dim=256, seed=seed).to(case['mu'].device)
        initial = proxy.tangent_proxies.detach().clone()
        opt = torch.optim.AdamW(proxy.parameters(), lr=rate, weight_decay=0.01, eps=1e-8)
        rows = []
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss, stats = proxy(case['mu'].detach(), case['gold'], topk=k,
                                seed=seed+step, data_ids=case['ids'],
                                generator=torch.Generator(device=case['mu'].device).manual_seed(seed+step))
            loss.backward()
            gradient = proxy.tangent_proxies.grad
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise RuntimeError(f'Nonfinite proxy LR diagnostic: {rate}')
            rows.append({'step': step, 'loss': float(loss.detach()),
                         'gradient_norm': float(gradient.norm()),
                         'shadow_gradient_elements_over_10': int((gradient.abs()>10).sum()),
                         'sample': stats['sample'], 'proxy': stats['proxy'],
                         'geometry': stats['geometry']})
            # HIER extra clipping/preconditioning remain disabled.
            opt.step()
        reports.append({'lr': rate, 'steps': steps, 'proxy_only': True,
                        'tangent_parameter_change_norm': float((proxy.tangent_proxies-initial).norm()),
                        'rows': rows})
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[3,6,8,10,20])
    p.add_argument('--seed', type=int, default=22)
    args = p.parse_args()
    torch.set_num_threads(2)
    cache = torch.load(args.features, map_location='cpu', weights_only=False)
    proxy = HIERLoss(cache['proxy_count'], dim=256, seed=args.seed)
    rows = [row for case in cache['cases'] for row in examine_case(case, proxy, args.ks, args.seed)]
    report = {'cache_identity': cache['identity'], 'config': cache['config'], 'rows': rows,
              'self_negative': [self_negative_audit(case, proxy) for case in cache['cases']],
              'not_a_classification_performance_test': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
