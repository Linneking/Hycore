"""Detached observations of actual synchronized proxy gradients and AdamW steps.

The measured .grad includes lambda=.5 and the existing cap/GS Jacobian. Epsilon
does not modify that gradient. Fixed-moment epsilon comparisons below are
algebraic counterfactual updates, not separate optimizer interventions.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

EPOCH_FILE_TEMPLATE = 'observer_epoch_{epoch:03d}.json'


def summary(values):
    x = torch.as_tensor(values).detach().double().cpu().reshape(-1).numpy()
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty observer values")
    q = np.quantile(x, [.01, .1, .5, .9, .99])
    return dict(count=int(x.size), mean=float(x.mean()), std_population=float(x.std()),
                min=float(x.min()), p01=float(q[0]), p10=float(q[1]), median=float(q[2]),
                p90=float(q[3]), p99=float(q[4]), max=float(x.max()))


def mapped_depth(parameter):
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball, depth
    with torch.no_grad():
        return depth(to_ball(parameter.detach(), c=.1, clip_r=2.3, hook=False), c=.1).double().cpu()


def array_sha(value):
    value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


@torch.no_grad()
def capture_before(parameter, optimizer):
    group, = [g for g in optimizer.param_groups if any(p is parameter for p in g['params'])]
    if parameter.grad is None or group.get('amsgrad', False) or group.get('maximize', False):
        raise ValueError('Expected actual dense minimizing non-AMSGrad proxy gradient')
    state = optimizer.state.get(parameter, {})
    return dict(parameter=parameter.detach().clone(), gradient=parameter.grad.detach().clone(),
                m=state.get('exp_avg', torch.zeros_like(parameter)).detach().clone(),
                v=state.get('exp_avg_sq', torch.zeros_like(parameter)).detach().clone(),
                step=int(state.get('step', 0)), lr=float(group['lr']), eps=float(group['eps']),
                betas=tuple(group['betas']), weight_decay=float(group['weight_decay']))


@torch.no_grad()
def measure_step(before, parameter_after, optimizer):
    p, g, m, v = [before[k].double() for k in ('parameter', 'gradient', 'm', 'v')]
    after = parameter_after.detach().double()
    b1, b2 = before['betas']; step = before['step'] + 1
    mhat = (b1*m + (1-b1)*g)/(1-b1**step)
    sqrt_vhat = ((b2*v + (1-b2)*g.square())/(1-b2**step)).sqrt()
    effective = -before['lr']*mhat/(sqrt_vhat+before['eps'])
    decay = -before['lr']*before['weight_decay']*p
    actual = after-p
    pnorm = p.norm(dim=-1)
    if bool((pnorm <= 0).any()): raise ValueError('Undefined zero proxy direction')
    direction = p/pnorm[:, None]
    radial_g = (g*direction).sum(-1)
    tangent_g = (g-radial_g[:,None]*direction).norm(dim=-1)
    radial_move = (actual*direction).sum(-1)
    small = -before['lr']*mhat/(sqrt_vhat+1e-8)
    large = -before['lr']*mhat/(sqrt_vhat+1e-4)
    ratio = float(large.norm()/small.norm()) if float(small.norm()) else None
    def cosine(a,b):
        denominator=a.norm()*b.norm()
        return float((a*b).sum()/denominator) if float(denominator) else None
    after_norm = after.norm(dim=-1)
    per_proxy = dict(gradient_norm=g.norm(dim=-1).cpu(), signed_radial_gradient=radial_g.cpu(),
                     tangent_gradient_norm=tangent_g.cpu(), actual_update_norm=actual.norm(dim=-1).cpu(),
                     signed_radial_update=radial_move.cpu(), raw_norm_delta=(after_norm-pnorm).cpu())
    result = dict(
        lr=before['lr'], eps=before['eps'], optimizer_step=step,
        proxy_gradient_norm=float(g.norm()), proxy_gradient_per_proxy_median=float(g.norm(dim=-1).median()),
        proxy_gradient_coordinate_abs_median=float(g.abs().median()),
        actual_update_norm=float(actual.norm()), weight_decay_update_norm=float(decay.norm()),
        adam_effective_update_norm=float((actual-decay).norm()),
        sqrt_vhat_median=float(sqrt_vhat.median()),
        eps_dominance_fraction=float((sqrt_vhat<before['eps']).double().mean()),
        update_attenuation_median=float((sqrt_vhat/(sqrt_vhat+before['eps'])).median()),
        fixed_moment_eps1e4_vs_eps1e8_update_norm_ratio=ratio,
        fixed_moment_eps1e4_vs_eps1e8_update_cosine=cosine(large,small),
        fixed_moment_eps1e4_vs_eps1e8_net_update_cosine=cosine(large+decay,small+decay),
        fixed_moment_eps1e8_net_radial_step_mean=float(((small+decay)*direction).sum(-1).mean()),
        fixed_moment_eps1e4_net_radial_step_mean=float(((large+decay)*direction).sum(-1).mean()),
        fixed_moment_eps1e8_update_norm=float(small.norm()), fixed_moment_eps1e4_update_norm=float(large.norm()),
        prediction_error_max_coordinate=float((actual-effective-decay).abs().max()),
        before_cap_count=int((pnorm>2.3).sum()), cap_count=int((after_norm>2.3).sum()),
        crossing_events=int(((pnorm<=2.3)&(after_norm>2.3)).sum()),
        return_events=int(((pnorm>2.3)&(after_norm<=2.3)).sum()),
        gradient_per_proxy=summary(per_proxy['gradient_norm']),
        signed_radial_gradient_per_proxy=summary(radial_g), tangent_gradient_per_proxy=summary(tangent_g),
        actual_update_per_proxy=summary(per_proxy['actual_update_norm']),
        signed_radial_update_per_proxy=summary(radial_move),
        gradient_semantics='actual synchronized optimizer-input proxy .grad; lambda0.5 and cap/GS included; eps has not altered this gradient',
        fixed_moment_semantics='same actual mhat/vhat; hypothetical adaptive update only; excludes weight decay and is not another training run')
    state = optimizer.state[parameter_after]
    result['actual_optimizer_step'] = int(state['step'])
    if result['prediction_error_max_coordinate'] > 2e-6 or result['actual_optimizer_step'] != step:
        raise RuntimeError('Actual AdamW update differs from detached accounting')
    return result, per_proxy


class Observer:
    def __init__(self, namespace):
        self.ns = namespace

    def bind(self, proxy, optimizer, run_dir, rank, manifest, config):
        self.proxy, self.optimizer, self.run_dir, self.rank = proxy, optimizer, Path(run_dir), rank
        self.config, self.manifest = config, manifest
        self.epoch, self.step = 0, 0
        if rank:
            return
        self.initial_depth = mapped_depth(proxy.tangent_proxies)
        self.initial_norm = proxy.tangent_proxies.detach().double().norm(dim=-1).cpu()
        self.previous_depth, self.previous_norm = self.initial_depth.clone(), self.initial_norm.clone()
        self.ever_cap = self.initial_norm > 2.3
        self.ever_returned = torch.zeros_like(self.ever_cap)
        self.rows, self.per_proxy = [], []
        self.first_crossings, self.crossing_events, self.return_events = 0, 0, 0
        self.write_epoch(0)

    def begin(self, epoch, step):
        self.epoch, self.step = epoch, step
        if self.rank == 0 and step == 1:
            self.rows, self.per_proxy = [], []
            self.first_crossings, self.crossing_events, self.return_events = 0, 0, 0

    def before_step(self, proxy, optimizer):
        return capture_before(proxy.tangent_proxies, optimizer) if self.rank == 0 else None

    def after_step(self, before, proxy, optimizer):
        if self.rank:
            return
        result, per_proxy = measure_step(before, proxy.tangent_proxies, optimizer)
        norm_before = before['parameter'].detach().double().norm(dim=-1).cpu()
        norm_after = proxy.tangent_proxies.detach().double().norm(dim=-1).cpu()
        cap = norm_after>2.3
        self.first_crossings += int((cap & ~self.ever_cap).sum())
        self.ever_cap |= cap
        self.ever_returned |= (norm_before>2.3)&~cap
        self.crossing_events += result['crossing_events']; self.return_events += result['return_events']
        result.update(epoch=self.epoch, step=self.step, ever_cap_count=int(self.ever_cap.sum()),
                      returned_ever_count=int(self.ever_returned.sum()))
        if self.epoch == 1 and self.step == 1:
            result.update(initial_proxy_parameter_sha256=array_sha(before['parameter']),
                          initial_proxy_gradient_sha256=array_sha(before['gradient']))
        self.ns['append_jsonl'](self.run_dir/'proxy_steps.jsonl', result)
        self.rows.append(result); self.per_proxy.append(per_proxy)

    def write_epoch(self, epoch):
        if self.rank:
            return
        depth = mapped_depth(self.proxy.tangent_proxies)
        norm = self.proxy.tangent_proxies.detach().double().norm(dim=-1).cpu()
        delta_epoch, delta_init = depth-self.previous_depth, depth-self.initial_depth
        result = dict(schema='b05-eps-proxy-epoch-1', epoch=epoch, eps=float(self.optimizer.param_groups[0]['eps']),
                      proxy_count=len(depth), c=.1, depth_units='origin hyperbolic distance',
                      proxy_depth=depth.tolist(), raw_tangent_norm=norm.tolist(),
                      depth_delta_epoch=delta_epoch.tolist(), depth_delta_init=delta_init.tolist(),
                      raw_norm_delta_epoch=(norm-self.previous_norm).tolist(), raw_norm_delta_init=(norm-self.initial_norm).tolist(),
                      cap_count=int((norm>2.3).sum()), ever_cap_count=int(self.ever_cap.sum()),
                      returned_ever_count=int(self.ever_returned.sum()), first_crossings_epoch=self.first_crossings,
                      crossing_events_epoch=self.crossing_events, returns_epoch=self.return_events,
                      proxy_depth_summary=summary(depth), raw_tangent_norm_summary=summary(norm),
                      depth_delta_epoch_summary=summary(delta_epoch), depth_delta_init_summary=summary(delta_init),
                      depth_movement_tolerance=1e-5,
                      inward_epoch_count=int((delta_epoch < -1e-5).sum()), outward_epoch_count=int((delta_epoch>1e-5).sum()),
                      stationary_epoch_count=int((delta_epoch.abs()<=1e-5).sum()),
                      initial_proxy_parameter_sha256=array_sha(self.proxy.tangent_proxies) if epoch==0 else None,
                      step_count=len(self.rows), metrics={})
        if self.rows:
            keys=('proxy_gradient_norm','proxy_gradient_per_proxy_median','proxy_gradient_coordinate_abs_median',
                  'actual_update_norm','weight_decay_update_norm','adam_effective_update_norm','sqrt_vhat_median',
                  'eps_dominance_fraction','update_attenuation_median','fixed_moment_eps1e4_vs_eps1e8_update_norm_ratio')
            keys += ('fixed_moment_eps1e4_vs_eps1e8_update_cosine','fixed_moment_eps1e4_vs_eps1e8_net_update_cosine',
                     'fixed_moment_eps1e8_net_radial_step_mean','fixed_moment_eps1e4_net_radial_step_mean')
            result['metrics']={key:summary([r[key] for r in self.rows if r[key] is not None]) for key in keys}
            result['per_proxy_step_metrics']={key:summary(torch.cat([r[key] for r in self.per_proxy])) for key in self.per_proxy[0]}
            result['gradient_semantics']=self.rows[0]['gradient_semantics']
        self.ns['atomic_json'](self.run_dir/EPOCH_FILE_TEMPLATE.format(epoch=epoch), result)
        self.previous_depth, self.previous_norm = depth.clone(), norm.clone()

    def end_epoch(self, epoch):
        self.write_epoch(epoch)
