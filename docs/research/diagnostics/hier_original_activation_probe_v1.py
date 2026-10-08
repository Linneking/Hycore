"""CPU frozen best-cache replay of the released HIER loss, with role/gradient attribution.

This never loads an encoder, changes a parameter, or performs a training epoch.
It uses the official B90/rank, two-rank gather (B180), IPC2 sampler and HIER
mining/selection formulas. Two independently exported augmentation caches are
controls, not fresh random training augmentations or saved training activations.
Full proxy identities remain in the private output directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data.sampler import Sampler


ROLE_KEYS = [f'{source}_{role}_{condition}'
             for source in ('sample', 'proxy') for role in ('pair', 'triple')
             for condition in ('selected', 'noncollision', 'positive_hinge')]
ROLE_KEYS += [f'proxy_endpoint_{role}_{condition}'
              for role in ('i', 'j', 'k') for condition in ('selected', 'positive_hinge')]
GRAD_KEYS = [f'{source}_{mode}_{component}' for source in ('sample', 'proxy')
             for mode in ('full_st', 'direct')
             for component in ('norm', 'radial_signed', 'angular_norm')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extract_class(path, name, scope):
    """Execute only the audited official class AST; skip unrelated imports/classes."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f'Cannot uniquely locate {name}.')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def official_runtime(reference):
    reference = Path(reference)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference / 'hier'))
    import hyptorch.nn as hypnn
    import hyptorch.pmath as pmath
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'np': np,
             'hypnn': hypnn, 'dist_matrix': pmath.dist_matrix}
    cls = extract_class(reference / 'hier/losses.py', 'HIERLoss', scope)
    class Py38RandomCompatibility:
        @staticmethod
        def randint(a, b):
            # Released code supplies 1e6 (float), accepted by its Python 3.8
            # runtime but rejected by Python 3.12. Same integral bounds/distribution.
            return random.randint(int(a), int(b))
    sampler_cls = extract_class(reference / 'hier/sampler.py', 'UniqueClassSampler',
                                {'torch': torch, 'np': np, 'random': Py38RandomCompatibility,
                                 'Sampler': Sampler})
    return cls, hypnn, pmath, sampler_cls


def official_loss_object(cls, hypnn, pmath, tangent, c=.1, cap=2.3, tau=.1, margin=.1):
    # The released constructor uses .cuda(); initialize its identical state on CPU.
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    obj.nb_proxies, obj.sz_embed = tangent.shape
    obj.hyp_c, obj.clip_r, obj.tau, obj.mrg = c, cap, tau, margin
    obj.lcas = nn.Parameter(tangent.detach().clone())
    obj.to_hyperbolic = hypnn.ToPoincare(c=c, ball_dim=obj.sz_embed, riemannian=True,
                                      clip_r=cap, train_c=False)
    obj.dist_f = lambda x, y: pmath.dist_matrix(x, y, c=c)
    return obj


def block_distance(x, y, pmath, c, block=16, differentiable=False):
    pieces = []
    for start in range(0, len(x), block):
        xx = x[start:start + block]
        if differentiable:
            # Exact official forward and backward, recomputed by row blocks to
            # bound the released expanded N*M*D distance graph's memory.
            pieces.append(checkpoint(lambda a, b: pmath.dist_matrix(a, b, c=c),
                                     xx, y, use_reentrant=False))
        else:
            pieces.append(pmath.dist_matrix(xx, y, c=c))
    return torch.cat(pieces)


def instrumented_hhc(cp, indices, tau=.1, margin=.1, detach_selection=False):
    """Released compute_gHHC formula plus explicit records, no mining correction."""
    i, j, k = [torch.as_tensor(x, dtype=torch.long) for x in indices]
    max_ij = torch.maximum(cp[i], cp[j])
    prob_ij = F.gumbel_softmax(-max_ij / tau, dim=1, hard=True)
    max_ijk = torch.maximum(cp[k], max_ij)
    prob_ijk = F.gumbel_softmax(-max_ijk / tau, dim=1, hard=True)
    ij, ijk = prob_ij.argmax(-1), prob_ijk.argmax(-1)
    if detach_selection:
        prob_ij, prob_ijk = prob_ij.detach(), prob_ijk.detach()
    di_pair, di_triple = (cp[i] * prob_ij).sum(1), (cp[i] * prob_ijk).sum(1)
    dj_pair, dj_triple = (cp[j] * prob_ij).sum(1), (cp[j] * prob_ijk).sum(1)
    dk_pair, dk_triple = (cp[k] * prob_ij).sum(1), (cp[k] * prob_ijk).sum(1)
    hinge = torch.stack([torch.relu(di_pair - di_triple + margin),
                         torch.relu(dj_pair - dj_triple + margin),
                         torch.relu(dk_triple - dk_pair + margin)], dim=1)
    noncollision = ij != ijk
    per_triplet = hinge.sum(1) * noncollision.float()
    loss = per_triplet.mean()
    record = {'pair': ij.detach().numpy(), 'triple': ijk.detach().numpy(),
              'noncollision': noncollision.detach().numpy(),
              'positive_hinge': (per_triplet > 0).detach().numpy(),
              'hinge_components': hinge.detach().numpy(),
              'i': i.numpy(), 'j': j.numpy(), 'k': k.numpy(),
              'loss': float(loss.detach()), 'triplets': len(i),
              'self_k': int((i == k).sum()),
              'collision_count': int((~noncollision).sum())}
    return loss, record


def record_counts(counts, record, source):
    p = len(next(iter(counts.values())))
    for role in ('pair', 'triple'):
        indices = record[role]
        for condition in ('selected', 'noncollision', 'positive_hinge'):
            mask = np.ones(len(indices), bool) if condition == 'selected' else record[condition]
            counts[f'{source}_{role}_{condition}'] += np.bincount(indices[mask], minlength=p)
    if source == 'proxy':
        for role in ('i', 'j', 'k'):
            counts[f'proxy_endpoint_{role}_selected'] += np.bincount(record[role], minlength=p)
            mask = record['positive_hinge']
            counts[f'proxy_endpoint_{role}_positive_hinge'] += np.bincount(record[role][mask], minlength=p)


def gradient_components(gradient, tangent):
    g = gradient.detach().numpy().astype(np.float64)
    u = tangent.detach().numpy().astype(np.float64)
    direction = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-15)
    radial = (g * direction).sum(1)
    return {'norm': np.linalg.norm(g, axis=1), 'radial_signed': radial,
            'angular_norm': np.linalg.norm(g - radial[:, None] * direction, axis=1)}


def frozen_indices(sampler_cls, labels, batch, epoch, seed):
    random.seed(seed + batch)
    batches = []
    for rank in (0, 1):
        sampler = sampler_cls(labels.tolist(), 90, 2, rank=rank, world_size=2)
        sampler.set_epoch(epoch)
        batches.extend(sampler.sample_batch(batch))
    assert len(batches) == 180
    return np.asarray(batches, dtype=np.int64)


def gradient_probe(obj, pmath, z, sample_indices, proxy_indices, seed, reference_check=False,
                   selection_states=None):
    results, vectors, checks = {}, {}, {}
    for source, indices in (('sample', sample_indices), ('proxy', proxy_indices)):
        for mode in ('full_st', 'direct'):
            proxy = obj.to_hyperbolic(obj.lcas)
            x = z if source == 'sample' else proxy
            cp = block_distance(x, proxy, pmath, obj.hyp_c, differentiable=True)
            if selection_states is None:
                torch.manual_seed(seed + (0 if source == 'sample' else 10000))
            else:
                torch.random.set_rng_state(selection_states[source])
            before = torch.random.get_rng_state()
            loss, rec = instrumented_hhc(cp, indices, obj.tau, obj.mrg,
                                          detach_selection=mode == 'direct')
            gradient, = torch.autograd.grad(loss, obj.lcas)
            vectors[f'{source}_{mode}'] = gradient.detach().numpy()
            results[f'{source}_{mode}'] = gradient_components(gradient, obj.lcas)
            if reference_check and mode == 'full_st':
                proxy2 = obj.to_hyperbolic(obj.lcas)
                xx = z if source == 'sample' else proxy2
                cp2 = block_distance(xx, proxy2, pmath, obj.hyp_c, differentiable=True)
                torch.random.set_rng_state(before)
                official = obj.compute_gHHC(xx, proxy2, cp2, indices, None)
                official_grad, = torch.autograd.grad(official, obj.lcas)
                loss_delta = abs(float(official.detach()) - float(loss.detach()))
                grad_delta = float((official_grad - gradient).abs().max())
                checks[source] = {'loss_abs_delta': loss_delta, 'gradient_max_abs_delta': grad_delta}
                assert loss_delta < 1e-7 and grad_delta < 1e-7
            del cp, proxy, loss, gradient
    # Verify gradient additivity on a fresh combined graph and identical draws.
    if reference_check:
        proxy = obj.to_hyperbolic(obj.lcas)
        cp_sample = block_distance(z, proxy, pmath, obj.hyp_c, differentiable=True)
        cp_proxy = block_distance(proxy, proxy, pmath, obj.hyp_c, differentiable=True)
        if selection_states is None:
            torch.manual_seed(seed)
        else:
            torch.random.set_rng_state(selection_states['sample'])
        sample_loss, _ = instrumented_hhc(cp_sample, sample_indices, obj.tau, obj.mrg)
        if selection_states is None:
            torch.manual_seed(seed + 10000)
        else:
            torch.random.set_rng_state(selection_states['proxy'])
        proxy_loss, _ = instrumented_hhc(cp_proxy, proxy_indices, obj.tau, obj.mrg)
        combined, = torch.autograd.grad(sample_loss + proxy_loss, obj.lcas)
        target = vectors['sample_full_st'] + vectors['proxy_full_st']
        delta = float(np.max(np.abs(combined.detach().numpy() - target)))
        checks['gradient_additivity_max_abs_delta'] = delta
        assert delta < 1e-6
    return results, vectors, checks


def basic_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {'min': float(values.min()), 'p05': float(np.quantile(values, .05)),
            'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': float(values.max()), 'mean': float(values.mean()), 'std': float(values.std())}


def summarize_counts(counts, gradients, depth):
    sample_pos = counts['sample_pair_positive_hinge'] + counts['sample_triple_positive_hinge'] > 0
    proxy_ancestor_pos = counts['proxy_pair_positive_hinge'] + counts['proxy_triple_positive_hinge'] > 0
    proxy_endpoint_pos = sum(counts[f'proxy_endpoint_{r}_positive_hinge'] for r in ('i', 'j', 'k')) > 0
    result = {'roles': {k: {'used_proxies': int((v > 0).sum()), 'draws': int(v.sum()),
                            'max_draws': int(v.max()), 'depth': basic_stats(depth[v > 0])
                            if (v > 0).any() else None} for k, v in counts.items()},
              'positive_hinge_source_overlap': {
                  'sample_only': int((sample_pos & ~proxy_ancestor_pos).sum()),
                  'proxy_ancestor_only': int((~sample_pos & proxy_ancestor_pos).sum()),
                  'both_ancestor_sources': int((sample_pos & proxy_ancestor_pos).sum()),
                  'neither_ancestor_source': int((~sample_pos & ~proxy_ancestor_pos).sum()),
                  'proxy_endpoint': int(proxy_endpoint_pos.sum()),
                  'no_sample_but_proxy_endpoint': int((~sample_pos & proxy_endpoint_pos).sum())}}
    result['gradients'] = {}
    result['ancestor_role_overlap'] = {}
    for source in ('sample', 'proxy'):
        pair = counts[f'{source}_pair_positive_hinge']
        triple = counts[f'{source}_triple_positive_hinge']
        p_mask, t_mask = pair > 0, triple > 0
        union = (p_mask | t_mask).sum()
        result['ancestor_role_overlap'][source] = {
            'positive_pair_and_triple_proxies': int((p_mask & t_mask).sum()),
            'positive_role_jaccard': float((p_mask & t_mask).sum() / max(union, 1)),
            'positive_pair_draw_weighted_depth': float(np.average(depth, weights=pair)) if pair.sum() else None,
            'positive_triple_draw_weighted_depth': float(np.average(depth, weights=triple)) if triple.sum() else None}
    for name, values in gradients.items():
        norms = values['norm']
        if not norms:
            continue
        result['gradients'][name] = {
            'batches': len(norms), 'mean_norm_per_proxy': basic_stats(np.mean(norms, axis=0)),
            'nonzero_any': int((np.max(norms, axis=0) > 0).sum()),
            'above_1e-12_any': int((np.max(norms, axis=0) > 1e-12).sum()),
            'above_1e-8_any': int((np.max(norms, axis=0) > 1e-8).sum()),
            'mean_radial_signed': basic_stats(np.mean(values['radial_signed'], axis=0)),
            'mean_angular_norm': basic_stats(np.mean(values['angular_norm'], axis=0)),
            'radial_absolute_fraction_median': float(np.median(np.abs(values['radial_signed']) /
                                                               np.maximum(norms, 1e-30)))}
    return result


def make_figures(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = ('cub', 'cars', 'sop')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, ds in enumerate(datasets):
        data = results[f'{ds}_train']
        depth, counts = data['depth'], data['counts']
        order = np.argsort(depth)
        source_values = [counts['sample_pair_positive_hinge'] + counts['sample_triple_positive_hinge'],
                         counts['proxy_pair_positive_hinge'] + counts['proxy_triple_positive_hinge'],
                         sum(counts[f'proxy_endpoint_{r}_positive_hinge'] for r in ('i', 'j', 'k'))]
        labels = ['Sample ancestor', 'Proxy ancestor', 'Proxy endpoint']
        denominators = [counts['sample_pair_selected'].sum(),
                        counts['proxy_pair_selected'].sum(), counts['proxy_pair_selected'].sum()]
        rates = [v / max(d, 1) * 100000 for v, d in zip(source_values, denominators)]
        matrix = np.stack([np.log10(v[order] + 1) for v in rates])
        im = axes[0, col].imshow(matrix, aspect='auto', vmin=0, vmax=5, cmap='viridis')
        axes[0, col].set_yticks(range(3), labels)
        axes[0, col].set_title(f'{ds.upper()}: positive-hinge role counts')
        axes[0, col].set_xlabel('Proxy index sorted by origin depth')
        for vals, label, color in zip(rates[:2], labels[:2], ('#2271b2', '#d55e00')):
            axes[1, col].scatter(depth, vals + 1, label=label, color=color, s=12, alpha=.55)
        axes[1, col].set_yscale('log')
        axes[1, col].set_xlim(0, 4.65)
        axes[1, col].set_xlabel('Hyperbolic origin depth d0')
        axes[1, col].set_ylabel('Role draws / 100k item triplets + 1')
        axes[1, col].legend(fontsize=8)
    fig.colorbar(im, ax=axes[0, :], label='log10(role draws / 100k triplets + 1)', shrink=.8)
    fig.suptitle('Frozen best-cache replay; no optimizer updates; B180 / K20 / 50 draws per anchor')
    fig.savefig(output / '01_role_sources_depth.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, ds in enumerate(datasets):
        data = results[f'{ds}_train']
        depth = data['depth']
        for source, color in (('sample', '#2271b2'), ('proxy', '#d55e00')):
            full = np.mean(data['gradients'][f'{source}_full_st']['norm'], axis=0)
            direct = np.mean(data['gradients'][f'{source}_direct']['norm'], axis=0)
            axes[0, col].scatter(depth, np.maximum(full, 1e-16), s=12, color=color, alpha=.5, label=source)
            axes[1, col].scatter(np.maximum(direct, 1e-16), np.maximum(full, 1e-16),
                                 s=12, color=color, alpha=.5, label=source)
        axes[0, col].set_yscale('log')
        axes[0, col].set_xlim(0, 4.65)
        axes[0, col].set_ylim(1e-9, 1e-2)
        axes[0, col].set_title(ds.upper())
        axes[0, col].set_xlabel('Hyperbolic origin depth d0')
        axes[0, col].set_ylabel('Raw tangent gradient norm, full ST')
        axes[1, col].set_xscale('log')
        axes[1, col].set_yscale('log')
        axes[1, col].set_xlabel('Direct-only gradient norm')
        axes[1, col].set_ylabel('Full straight-through gradient norm')
        axes[1, col].legend(fontsize=8)
    fig.suptitle('Official cap + Riemannian backward hook; selected is not equivalent to gradient activation')
    fig.savefig(output / '02_gradient_source_depth.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, ds in enumerate(datasets):
        data = results[f'{ds}_train']
        counts, depth = data['counts'], data['depth']
        for row, source in enumerate(('sample', 'proxy')):
            ax = axes[row, col]
            denom = counts[f'{source}_pair_selected'].sum()
            pair = counts[f'{source}_pair_positive_hinge'] * 100000 / denom
            triple = counts[f'{source}_triple_positive_hinge'] * 100000 / denom
            im = ax.scatter(pair, triple, c=depth, s=12, alpha=.7, cmap='viridis', vmin=2.8, vmax=4.6)
            maximum = max(pair.max(), triple.max())
            ax.plot([0, maximum], [0, maximum], color='#666666', lw=1)
            ax.set_title(f'{ds.upper()}: {source} item')
            ax.set_xlabel('Pair ancestor positive-hinge draws / 100k')
            ax.set_ylabel('Triple ancestor positive-hinge draws / 100k')
    fig.colorbar(im, ax=axes.ravel(), label='Proxy origin depth d0', shrink=.8)
    fig.suptitle('Same proxy can serve both ancestor roles; comparison is conditional on the frozen sampler')
    fig.savefig(output / '03_pair_triple_role_overlap.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for col, ds in enumerate(datasets):
        data = results[f'{ds}_train']
        for source, color in (('sample', '#2271b2'), ('proxy', '#d55e00')):
            comp = data['gradients'][f'{source}_full_st']
            fraction = np.abs(comp['radial_signed']) / np.maximum(comp['norm'], 1e-30)
            fraction = np.sort(fraction.ravel())
            axes[col].plot(np.maximum(fraction, 1e-10), np.arange(1, len(fraction) + 1) / len(fraction),
                           color=color, label=f'{source} full ST')
        axes[col].set_xscale('log')
        axes[col].set_xlim(1e-10, 1)
        axes[col].set_ylim(0, 1)
        axes[col].set_title(ds.upper())
        axes[col].set_xlabel('|Radial tangent gradient| / total gradient norm')
        axes[col].set_ylabel('Proxy-batch cumulative fraction')
        axes[col].legend(fontsize=8)
    fig.suptitle('SOP forward tangent clipping suppresses the radial derivative while angular gradients persist')
    fig.savefig(output / '04_gradient_radial_fraction.png', dpi=160)
    plt.close(fig)


def load_probe_output(output):
    """Load private per-proxy arrays for refreshed figures/extended summaries."""
    results = {}
    for folder in Path(output).iterdir():
        path = folder / 'arrays.npz' if folder.is_dir() else None
        if path is None or not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            counts = {k: z[k].copy() for k in ROLE_KEYS}
            gradients = {f'{s}_{m}': {c: list(z[f'{s}_{m}_{c}'].copy())
                                    if f'{s}_{m}_{c}' in z.files else []
                                    for c in ('norm', 'radial_signed', 'angular_norm')}
                         for s in ('sample', 'proxy') for m in ('full_st', 'direct')}
            results[folder.name] = {'depth': z['depth'].copy(), 'counts': counts, 'gradients': gradients}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--converted', required=True)
    ap.add_argument('--handoff', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batches', type=int, default=16)
    ap.add_argument('--full-clean-epoch', action='store_true',
                    help='Clean cache: floor(N/180) official-sampler frozen batches; aug controls use --batches.')
    ap.add_argument('--gradient-batches', type=int, default=2)
    ap.add_argument('--datasets', nargs='+', default=['cub', 'cars', 'sop'])
    ap.add_argument('--variants', nargs='+', default=['train', 'train_aug1', 'train_aug2'])
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--sampler-epoch-offset', type=int, default=0,
                    help='Default: checkpoint completed count equals the next 0-based sampler epoch.')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('A fresh output directory is required.')
    output.mkdir(parents=True)
    cls, hypnn, pmath, sampler_cls = official_runtime(args.reference)
    from hier_original_handoff_loader_v1 import safe_checkpoint
    results, summary, validations = {}, {}, {}
    start = time.time()
    for ds in args.datasets:
        checkpoint_path = Path(args.handoff) / f'logs/run_{ds}_r50/checkpoint_best.pth'
        cp_hash = sha256(checkpoint_path)
        ck = safe_checkpoint(checkpoint_path)
        tangent = ck['cluster_loss']['lcas'].float()
        epoch = int(ck['epoch']) + args.sampler_epoch_offset
        assert ck['args'].topk == 20 and ck['args'].batch_size == 90 and ck['args'].world_size == 2
        obj = official_loss_object(cls, hypnn, pmath, tangent)
        del ck
        with torch.no_grad():
            proxy = obj.to_hyperbolic(obj.lcas)
            proxy_distance = block_distance(proxy, proxy, pmath, .1)
            proxy_sim = torch.exp(-proxy_distance)
        depth = (2 / math.sqrt(.1) * torch.atanh(math.sqrt(.1) * proxy.norm(dim=1))).numpy()
        for variant in args.variants:
            name = f'{ds}_{variant}'
            path = Path(args.converted) / f'{name}.npz'
            cache_hash = sha256(path)
            with np.load(path, allow_pickle=False) as z:
                whole = torch.from_numpy(z['mu'].copy()).float()
                labels = z['labels'].copy()
                assert np.max(np.abs(z['proxy_ball'] - proxy.numpy())) < 2e-6
            counts = {k: np.zeros(512, np.int64) for k in ROLE_KEYS}
            gradients = {f'{s}_{m}': {c: [] for c in ('norm', 'radial_signed', 'angular_norm')}
                         for s in ('sample', 'proxy') for m in ('full_st', 'direct')}
            batch_rows, gradient_vectors = [], {}
            replay_batches = len(labels) // 180 if args.full_clean_epoch and variant == 'train' else args.batches
            for batch in range(replay_batches):
                indices = frozen_indices(sampler_cls, labels, batch, epoch, args.seed)
                zz, yy = whole[indices], torch.from_numpy(labels[indices])
                np.random.seed(args.seed + batch)
                torch.manual_seed(args.seed + batch)
                with torch.no_grad():
                    sample_distance = block_distance(zz, proxy, pmath, .1)
                    within = block_distance(zz, zz, pmath, .1)
                    sample_sim = torch.exp(-within)
                    sample_sim[yy[:, None] == yy[None, :]] += 1
                    sample_indices = obj.get_reciprocal_triplets(sample_sim, topk=20, t_per_anchor=50)
                    sample_state = torch.random.get_rng_state()
                    _, sample_record = instrumented_hhc(sample_distance, sample_indices)
                    proxy_indices = obj.get_reciprocal_triplets(proxy_sim, topk=20, t_per_anchor=50)
                    proxy_state = torch.random.get_rng_state()
                    _, proxy_record = instrumented_hhc(proxy_distance, proxy_indices)
                record_counts(counts, sample_record, 'sample')
                record_counts(counts, proxy_record, 'proxy')
                batch_rows.append({'batch': batch, 'distinct_objects': int(np.unique(indices).size),
                                   'classes': int(np.unique(labels[indices]).size),
                                   **{f'{s}_{k}': r[k] for s, r in (('sample', sample_record), ('proxy', proxy_record))
                                      for k in ('loss', 'triplets', 'self_k', 'collision_count')},
                                   **{f'{s}_positive_hinge_triplets': int(r['positive_hinge'].sum())
                                      for s, r in (('sample', sample_record), ('proxy', proxy_record))}})
                if batch < args.gradient_batches:
                    values, vectors, checks = gradient_probe(obj, pmath, zz, sample_indices, proxy_indices,
                                                             args.seed + batch,
                                                             reference_check=batch == 0 and variant == 'train',
                                                             selection_states={'sample': sample_state,
                                                                               'proxy': proxy_state})
                    for source_mode, components in values.items():
                        for key, arr in components.items():
                            gradients[source_mode][key].append(arr)
                    for key, arr in vectors.items():
                        gradient_vectors[f'b{batch}_{key}'] = arr
                    if checks:
                        validations[name] = checks
                print(json.dumps({'model': name, 'batch': batch + 1,
                                  'sample_loss': sample_record['loss'], 'proxy_loss': proxy_record['loss'],
                                  'elapsed_seconds': round(time.time() - start, 1)}), flush=True)
            result = {'depth': depth, 'counts': counts, 'gradients': gradients}
            results[name] = result
            summary[name] = summarize_counts(counts, gradients, depth)
            summary[name].update(checkpoint_sha256=cp_hash, converted_cache_sha256=cache_hash,
                                 replay_batches=replay_batches, gradient_batches=min(args.gradient_batches, replay_batches),
                                 next_epoch_sampler_number=epoch,
                                 full_global_epoch_batches=len(labels) // 180,
                                 whole_depth=basic_stats((2 / math.sqrt(.1) * torch.atanh(
                                     math.sqrt(.1) * whole.norm(dim=1))).numpy()), proxy_depth=basic_stats(depth))
            folder = output / name
            folder.mkdir()
            np.savez(folder / 'arrays.npz', depth=depth, **counts, **gradient_vectors,
                     **{f'{name}_{key}': np.stack(arr) for name, comps in gradients.items()
                        for key, arr in comps.items() if arr})
            keys = ['proxy_id', 'depth'] + ROLE_KEYS + GRAD_KEYS
            with (folder / 'per_proxy.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for p in range(512):
                    row = {'proxy_id': p, 'depth': float(depth[p]), **{k: int(v[p]) for k, v in counts.items()}}
                    for sm, comps in gradients.items():
                        for key, arr in comps.items():
                            row[f'{sm}_{key}'] = float(np.mean(arr, axis=0)[p]) if arr else ''
                    writer.writerow(row)
            (folder / 'batches.json').write_text(json.dumps(batch_rows, indent=2), encoding='utf-8')
            assert sha256(path) == cache_hash
        assert sha256(checkpoint_path) == cp_hash
    manifest = {'protocol': 'Frozen CPU cache replay, zero optimizer updates, zero encoder/image forwards',
                'selection_batch_size_global': 180, 'per_rank_batch_size': 90, 'world_size': 2,
                'images_per_class': 2, 'K': 20, 'draws_per_anchor': 50, 'tau': .1, 'margin': .1,
                'official_proxy_tangent_cap': 2.3, 'official_Riemannian_backward_hook': True,
                'source_sha256': {file: sha256(Path(args.reference) / file) for file in
                                  ('hier/losses.py', 'hier/sampler.py', 'hier/hyptorch/nn.py', 'hier/hyptorch/pmath.py')},
                'seed': args.seed, 'sampler_epoch_offset': args.sampler_epoch_offset,
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'CPU'},
                'definitions': {'selected': 'hard Gumbel ancestor, including collisions',
                                'noncollision': 'pair ancestor != triple ancestor',
                                'positive_hinge': 'sum of three hinges > 0 after collision mask',
                                'endpoint': 'proxy participates as i/j/k, separately from ancestor choice',
                                'full_st': 'official hard Gumbel straight-through gradient',
                                'direct': 'same hard selection, Gumbel probabilities detached'},
                'limitations': ['Not saved training usage; all tensors exported in eval mode.',
                                'No encoder, PA class proxies, optimizer updates, gradient clipping or AMP.',
                                'Python 3.8 randint integral-float bound cast to int for Python 3.12 compatibility.',
                                'Row-block official FP32 distances can differ by a few rounding ulps from one full matrix GEMM.',
                                'Finite replay window; selected and positive hinge differ from nonzero ST gradient.',
                                'Proxy item has endpoint and ancestor paths; source attribution is total item gradient.',
                                'Fixed seed paired by views; sample mining topology may change the subsequent proxy Gumbel RNG position.'],
                'seconds': time.time() - start, 'results': summary, 'validations': validations}
    (output / 'summary.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if not args.no_figures and set(args.datasets) == {'cub', 'cars', 'sop'} and 'train' in args.variants:
        make_figures(output, results)
    print(json.dumps({'complete': True, 'seconds': time.time() - start, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
