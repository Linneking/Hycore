"""Frozen CPU diagnosis of HIER ancestor selection and radial exclusion.

No forward through the encoder, optimizer update, remine under controls, or GPU.
Inputs are explicitly declared existing feature cache and normalized training log.
Output must be a new directory. Per-ID diagnostic arrays stay outside Git.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def distribution(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        return {"n": 0}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite radial statistic")
    q = np.quantile(x, [0, .05, .25, .5, .75, .95, 1])
    return dict(n=len(x), min=float(q[0]), p05=float(q[1]), p25=float(q[2]),
                median=float(q[3]), p75=float(q[4]), p95=float(q[5]), max=float(q[6]),
                mean=float(x.mean()), variance_population=float(x.var()),
                std_population=float(x.std()), iqr=float(q[4]-q[2]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(repo, cache, normalized, epoch, output, query_count=512, repeats=8):
    import torch
    sys.path.insert(0, str(repo))
    from tools.hier_postrun_audit.geometry import ball_geometry, expmap0
    from tools.hier_postrun_audit.mechanisms import make_query_plan, audit_snapshot, _remap, _kernel
    torch.set_num_threads(4)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Analysis output must be new")
    out.mkdir(parents=True)
    cache = Path(cache)
    before = sha(cache)
    with np.load(cache, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    logs = json.loads(Path(normalized).read_text())
    if len(logs) != 1:
        raise ValueError("Exactly one normalized source run is required")
    log = logs[0]
    ids = a['proxy_ids']
    mapping = {int(i): k for k, i in enumerate(ids)}
    usage = [r for r in log['proxy_usage'] if r['epoch'] == epoch]
    target = [r for r in usage if (r['component'], r['domain'], r['role']) == ('sample', 'noncollision', 'combined')]
    if len(target) != 1:
        raise ValueError("Missing or ambiguous activation epoch")
    selected = np.zeros(len(ids), dtype=bool)
    for i, c in zip(target[0]['proxy_ids'], target[0]['counts']):
        selected[mapping[int(i)]] = c > 0
    ball = expmap0(a['proxy_tangent'], 1, numeric_radius_fraction=.999)
    pg, wg = ball_geometry(ball, 1), ball_geometry(a['mu'], 1)
    result = dict(schema='hier_selection_radial_diagnosis_v1', epoch=epoch,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        identity=dict(cache_file=cache.name, cache_sha256=before,
                      normalized_sha256=sha(normalized), run_key=log['run_id']),
        selection_scope='epoch training sample noncollision pair+triple; endpoint snapshot depths',
        inference_scope='fixed clean first1024/eval BN; not augmented training geometry',
        radial_units='full origin hyperbolic distance d0, c=1; d0=2*atanh(radius)',
        geometry=dict(whole=distribution(wg['depth']), all_proxy=distribution(pg['depth']),
                      selected=distribution(pg['depth'][selected]),
                      not_selected=distribution(pg['depth'][~selected])),
        unused_vs_whole=dict(fraction_deeper_than_clean_whole_median=float(np.mean(pg['depth'][~selected]>np.median(wg['depth']))),
                             fraction_deeper_than_clean_whole_max=float(np.mean(pg['depth'][~selected]>wg['depth'].max()))))
    role_rows = []
    for r in usage:
        c = np.zeros(len(ids), dtype=np.int64)
        for i, v in zip(r['proxy_ids'], r['counts']):
            c[mapping[int(i)]] = v
        role_rows.append(dict(component=r['component'], domain=r['domain'], role=r['role'],
             unique_ids=int((c>0).sum()), selected_group_slots=int(c[selected].sum()),
             not_selected_group_slots=int(c[~selected].sum()),
             not_selected_group_id_count=int(((c>0)&~selected).sum()),
             selected_depth_mean_weighted=float(np.dot(c,pg['depth'])/c.sum()) if c.sum() else None,
             selected_id_depth=distribution(pg['depth'][c>0])))
    result['training_roles'] = role_rows
    first_return, ever_return = {}, set()
    for r in log['proxy_usage']:
        if r['epoch']<=epoch or (r['component'],r['domain'],r['role']) != ('sample','noncollision','combined'):
            continue
        for i, c in zip(r['proxy_ids'],r['counts']):
            if c>0 and not selected[mapping[int(i)]]:
                ever_return.add(int(i)); first_return.setdefault(int(i),r['epoch'])
    result['future_reselection'] = dict(not_selected_at_this_epoch=int((~selected).sum()),
        selected_again_later=len(ever_return), never_selected_later=int((~selected).sum())-len(ever_return),
        last_epoch=max(r['epoch'] for r in log['proxy_usage']),
        first_return_epoch_histogram={str(e):list(first_return.values()).count(e) for e in sorted(set(first_return.values()))})
    snapshot = dict(spec=dict(topk=20, proxy_topk=20, t_per_anchor=50, margin=.1,tau=.1,lambda_hier=.1),
        whole=a['mu'], labels=a['labels'],sample_ids=a['sample_ids'],
        tangent=a['proxy_tangent'],proxy_ids=ids,run_key=log['run_id'],epoch=epoch,
        version='v7',exclude_self_negative=True,identity=result['identity'],optimizer=None)
    plan = make_query_plan(snapshot,query_count=query_count,seed=22,sample_batch_size=64)
    _, _, distance, prodmap, _ = _kernel()
    rows = _remap(plan['sample_query_ids'], a['sample_ids'])
    w = torch.tensor(a['mu'],dtype=torch.float32)
    u = torch.tensor(a['proxy_tangent'],dtype=torch.float32)
    un = u.norm(dim=-1,keepdim=True)
    ud = u/un
    lower = float(np.median(pg['depth'][selected])) / 2
    upper = float(np.median(pg['depth'][~selected])) / 2
    variants = {'saved':u,
                'unused_shrunk_to_selected_median':torch.where(torch.tensor(selected[:,None]),u,ud*lower),
                'selected_expanded_to_unused_median':torch.where(torch.tensor(selected[:,None]),ud*upper,u),
                'all_at_selected_median':ud*lower}
    controls = {}
    for name,v in variants.items():
        with torch.no_grad():
            d = distance(w[np.unique(rows)],prodmap(v))
            compact = {int(i):k for k,i in enumerate(np.unique(rows))}
            ii,jj,kk = torch.tensor([[compact[int(i)] for i in row] for row in rows]).unbind(-1)
            pc = torch.maximum(d[ii],d[jj]); tc = torch.maximum(pc,d[kk])
            cr = {}
            for role,cost in [('pair',pc),('triple',tc)]:
                probability = torch.softmax(-cost/.1,-1)
                s = torch.tensor(selected)
                cr[role] = dict(query_count=len(rows),
                     expected_not_selected_group_probability_mean=float(probability[:,~s].sum(-1).mean()),
                     deterministic_argmin_not_selected_fraction=float((~s[cost.argmin(-1)]).float().mean()),
                     best_not_selected_minus_best_selected_cost=distribution((cost[:,~s].min(-1).values-cost[:,s].min(-1).values).numpy()))
            controls[name] = cr
    result['fixed_query_radial_controls'] = controls
    result['controls_scope'] = dict(query_plan_sha256=plan['query_plan_sha256'],sample_batch_size=64,
        query_count=len(rows),seed=22,tau=.1,
        note='Directions and mined query IDs frozen. Exact categorical probabilities softmax(-maxcost/tau); radial controls change geometry, not trained models.')
    probe = audit_snapshot(snapshot,plan,noise_repeats=repeats,seed=22,gradient_repeats=repeats,compare_source_operator=False)
    gradient_groups = {}
    for component in ['sample','proxy']:
        per_repeat = probe['components'][component]['gradient_repeats']
        groups = {}
        for name,mask in [('selected',selected),('not_selected',~selected)]:
            radial, norm = [], []
            for rep in per_repeat:
                values = rep['unweighted_proxy_tangent']['per_id']
                radial.append(np.array([r['signed_radial'] for r in values])[mask])
                norm.append(np.array([r['norm'] for r in values])[mask])
            mean_r = np.mean(radial,axis=0); mean_n = np.mean(norm,axis=0)
            groups[name] = dict(mean_per_id_gradient_norm=distribution(mean_n),
                  mean_per_id_signed_radial_derivative=distribution(mean_r),
                  positive_inward_GD_count_above_1e_8=int((mean_r>1e-8).sum()),
                  negative_outward_GD_count_below_minus_1e_8=int((mean_r<-1e-8).sum()),
                  exact_zero_gradient_id_count=int((mean_n==0).sum()))
        gradient_groups[component] = groups
    result['fixed_query_gradients'] = gradient_groups
    result['gradient_scope'] = 'Eight ST autograd noise repeats, unweighted tangent partials; signed derivative predicts plain GD only, not saved AdamW displacement.'
    result['read_only'] = dict(cache_unchanged=sha(cache)==before,optimizer_updates=0,
                              encoder_forwards=0,test_forwards=0,gpu_used=False)
    if not result['read_only']['cache_unchanged']:
        raise RuntimeError('Cache changed')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (out/'mechanism.json').write_text(json.dumps(probe,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(out/'radial_diagnostic_arrays.npz',proxy_ids=ids,proxy_depth=pg['depth'],
                        whole_depth=wg['depth'],selected_mask=selected)
    print(json.dumps(dict(output=out.name,geometry=result['geometry'],future_reselection=result['future_reselection'],
                          controls=result['fixed_query_radial_controls'],gradients=gradient_groups),allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--normalized',required=True);p.add_argument('--epoch',required=True,type=int)
    p.add_argument('--output',required=True);p.add_argument('--queries',type=int,default=512)
    p.add_argument('--repeats',type=int,default=8)
    args=p.parse_args()
    analyze(args.repo,args.cache,args.normalized,args.epoch,args.output,args.queries,args.repeats)
