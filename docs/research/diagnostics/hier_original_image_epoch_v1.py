"""Fresh-image, no-update epoch at a supplied original HIER best checkpoint.

The encoder can run on an explicitly requested GPU. HIER roles and tangent
gradients use the independently audited official FP32 CPU implementation.
This is a conditional best-checkpoint diagnostic, never a strict PA resume.
Complete image/proxy identities belong in a private output directory only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hier_original_activation_probe_v1 import (
    GRAD_KEYS, ROLE_KEYS, basic_stats, block_distance, gradient_probe,
    instrumented_hhc, official_loss_object, official_runtime,
    record_counts, sha256, summarize_counts,
)
from hier_original_handoff_loader_v1 import safe_checkpoint


EXPECTED = {'cub': (5864, 100, 'CUB_200_2011'),
            'cars': (8054, 98, 'cars196'),
            'sop': (59551, 11318, 'Stanford_Online_Products')}


class OriginalResnet50(nn.Module):
    """Same modules and aliased state names as released models/resnet.py."""
    def __init__(self, bn_freeze=True):
        super().__init__()
        from torchvision.models import resnet50
        self.model = resnet50(weights=None)
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.fc = nn.Identity()
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2,
                                          self.model.layer3, self.model.layer4])
        self.bn_freeze = bool(bn_freeze)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.bn_freeze:
            for module in self.model.modules():
                if type(module) is nn.BatchNorm2d:
                    module.eval()
        return self

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for block in self.layer_blocks:
            x = block(x)
        return (self.model.gap(x) + self.model.gmp(x)).view(x.shape[0], -1)


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class OriginalWhole(nn.Module):
    def __init__(self, hypnn, margs):
        super().__init__()
        if margs.model != 'resnet50' or not margs.bn_freeze:
            raise ValueError('This runner supports the supplied frozen-BN R50 checkpoints only.')
        self.body = OriginalResnet50(bn_freeze=True)
        last_norm = (nn.LayerNorm(2048, elementwise_affine=False)
                     if margs.use_lastnorm else nn.Identity())
        self.last_layer = nn.Sequential(
            last_norm, nn.Linear(2048, margs.emb),
            nn.Sequential(hypnn.ToPoincare(c=margs.hyp_c, ball_dim=margs.emb,
                                           riemannian=True, clip_r=margs.clip_r)))
        self.norm = NormLayer()

    def forward(self, x):
        return self.last_layer(self.body(x))


def unwrapped_state(state):
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in state.items()}


def state_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf-8'))
        h.update(str(array.dtype).encode('ascii'))
        h.update(str(array.shape).encode('ascii'))
        h.update(array.tobytes())
    return h.hexdigest()


def relative_image_path(path, dataset):
    """Strip only the known source dataset prefix, then reject escaping tails."""
    normalized = str(path).replace('\\', '/')
    marker = EXPECTED[dataset][2] + '/'
    position = normalized.rfind(marker)
    tail = normalized[position + len(marker):] if position >= 0 else normalized
    relative = PurePosixPath(tail)
    if relative.is_absolute() or '..' in relative.parts or ':' in tail:
        raise ValueError('Image path cannot be resolved inside the supplied dataset root.')
    return relative.as_posix()


def read_identity_map(path):
    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == '.json':
        document = json.loads(path.read_text(encoding='utf-8'))
        mapping = document.get('mapping', document)
        if not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values()):
            raise ValueError('Identity JSON must contain mapping: target_path -> source_path.')
        return mapping
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    allowed = {'unique', 'byte_equivalent'}
    return {row['target_path']: row['source_path'] for row in rows
            if row.get('status', 'unique') in allowed}


def resolve_paths(source_paths, dataset, images_root, mapping):
    root = Path(images_root).resolve()
    paths, unresolved = [], []
    for source in source_paths:
        relative = relative_image_path(source, dataset)
        if dataset == 'cars':
            if relative not in mapping:
                unresolved.append(relative)
                continue
            supplied = Path(mapping[relative])
            resolved = (supplied if supplied.is_absolute() else root / supplied).resolve()
        else:
            resolved = root.joinpath(*PurePosixPath(relative).parts).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Resolved image escapes the explicitly supplied image root.')
        if not resolved.is_file():
            unresolved.append(relative)
        paths.append(resolved)
    if unresolved:
        raise FileNotFoundError(f'{len(unresolved)} training identities unresolved; '
                                f'first entries: {unresolved[:5]}')
    if len(paths) != len(source_paths):
        raise AssertionError('Incomplete identity mapping.')
    return paths


def read_sample_map(path, sample_ids, labels):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    lookup = {int(row['sample_id']): row for row in rows}
    if len(lookup) != len(rows) or len(lookup) != len(sample_ids):
        raise ValueError('Sample map is incomplete or has duplicate identities.')
    ordered = [lookup[int(sid)] for sid in sample_ids]
    if any(int(row['label']) != int(label) for row, label in zip(ordered, labels)):
        raise ValueError('Sample map labels differ from the verified clean cache.')
    return [row['image_path'] for row in ordered]


def sampler_plan(sampler_cls, labels, epoch_index, seed):
    """Generate both source rank plans before the epoch, with independent RNGs."""
    old_python, old_numpy = random.getstate(), np.random.get_state()
    ranks = []
    try:
        for rank in (0, 1):
            random.seed(seed)
            np.random.seed(seed)
            sampler = sampler_cls(labels.tolist(), 90, 2, rank=rank, world_size=2)
            sampler.set_epoch(epoch_index)
            ranks.append(np.asarray(list(iter(sampler)), dtype=np.int64))
    finally:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
    batches = len(labels) // 180
    if any(len(plan) != batches * 90 for plan in ranks):
        raise AssertionError('Source sampler epoch length differs from expected drop-last length.')
    return np.concatenate([plan.reshape(batches, 90) for plan in ranks], axis=1)


def build_transforms():
    from torchvision import transforms as T
    interpolation = T.InterpolationMode.BICUBIC
    normalization = T.Normalize((.485, .456, .406), (.229, .224, .225))
    train = T.Compose([T.RandomResizedCrop(224, scale=(.08, 1.), interpolation=interpolation),
                       T.RandomHorizontalFlip(), T.ToTensor(), normalization])
    clean = T.Compose([T.Resize(256, interpolation=interpolation), T.CenterCrop(224),
                       T.ToTensor(), normalization])
    return train, clean


class ImagePool(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            value = self.transform(image.convert('RGB'))
        return value, int(index)


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def forward_images(model, images, device, microbatch, amp=False):
    values = []
    with torch.no_grad():
        for start in range(0, len(images), microbatch):
            part = images[start:start + microbatch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
                values.append(model(part).detach().float().cpu())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Encoder produced nonfinite whole embeddings.')
    return result


def clean_anchor_check(model, paths, clean_transform, cached, device, microbatch,
                       anchors=12, minimum_cosine=.9999, maximum_abs=.01):
    indices = np.unique(np.linspace(0, len(paths) - 1, min(anchors, len(paths)), dtype=int))
    pool = ImagePool(paths, clean_transform)
    images = torch.stack([pool[int(i)][0] for i in indices])
    model.eval()
    actual = forward_images(model, images, device, microbatch, amp=False)
    model.train(True)
    target = torch.from_numpy(cached[indices].copy()).float()
    cosine = F.cosine_similarity(actual, target, dim=1)
    maximum = float((actual - target).abs().max())
    result = {'anchors': len(indices), 'minimum_cosine': float(cosine.min()),
              'median_cosine': float(cosine.median()), 'maximum_abs_coordinate_delta': maximum,
              'required_minimum_cosine': minimum_cosine, 'allowed_maximum_abs_delta': maximum_abs,
              'sample_positions': indices.tolist(), 'precision': 'FP32; no encoder autocast'}
    if float(cosine.min()) < minimum_cosine or maximum > maximum_abs:
        raise AssertionError('Clean image identity/architecture replay failed: ' + json.dumps(result))
    return result


def write_proxy_csv(path, depth, counts, gradients):
    keys = ['proxy_id', 'depth'] + ROLE_KEYS + GRAD_KEYS
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for index in range(len(depth)):
            row = {'proxy_id': index, 'depth': float(depth[index]),
                   **{key: int(value[index]) for key, value in counts.items()}}
            for source_mode, components in gradients.items():
                for component, batches in components.items():
                    row[f'{source_mode}_{component}'] = (float(np.mean(batches, axis=0)[index])
                                                         if batches else '')
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=EXPECTED, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--clean-cache', required=True, help='Verified converted clean train NPZ.')
    parser.add_argument('--sample-map', required=True, help='Complete supplied train sample_id CSV.')
    parser.add_argument('--images-root', required=True, help='Root of this dataset, not the archives directory.')
    parser.add_argument('--identity-map', help='Required for remapped Cars; JSON mapping or identity CSV.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu', help='Explicit cpu or cuda:N; no automatic GPU selection.')
    parser.add_argument('--gpu-identity', default=None, help='Physical GPU identity supplied after idle verification.')
    parser.add_argument('--precision', choices=('source', 'fp32'), default='source')
    parser.add_argument('--microbatch', type=int, default=90)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--gradient-batches', type=int, default=2)
    parser.add_argument('--clean-anchors', type=int, default=12)
    parser.add_argument('--clean-cosine-min', type=float, default=.9999)
    parser.add_argument('--clean-max-abs', type=float, default=.01)
    parser.add_argument('--code-commit', default='unspecified')
    args = parser.parse_args()
    if args.microbatch <= 0 or args.workers < 0 or args.gradient_batches < 0 or args.clean_anchors <= 0:
        raise ValueError('Invalid positive batch/anchor size or nonnegative worker/gradient count.')
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError('--max-batches must be positive when supplied.')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('A fresh private output directory is required.')
    output.mkdir(parents=True)
    started, start_time = time.time(), datetime.now(timezone.utc).isoformat()
    sys.dont_write_bytecode = True
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit CPU or CUDA devices are supported.')
    if device.type == 'cuda' and not args.gpu_identity:
        raise ValueError('CUDA requires --gpu-identity recorded after an external idle-GPU check.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cls, hypnn, pmath, sampler_cls = official_runtime(args.reference)
    checkpoint_hash, cache_hash, map_hash = map(sha256, (args.checkpoint, args.clean_cache, args.sample_map))
    identity_hash = sha256(args.identity_map) if args.identity_map else None
    checkpoint = safe_checkpoint(args.checkpoint)
    source_checkpoint_hash = checkpoint.get('source_checkpoint_sha256', checkpoint_hash)
    margs = checkpoint['args']
    assert margs.dataset.lower() == args.dataset and margs.batch_size == 90 and margs.world_size == 2
    assert margs.topk == 20 and margs.IPC == 2 and margs.global_crops_number == 1
    assert margs.emb == 512 and margs.num_hproxies == 512 and margs.hyp_c == .1 and margs.clip_r == 2.3
    with np.load(args.clean_cache, allow_pickle=False) as cache:
        whole, labels, sample_ids = cache['mu'].copy(), cache['labels'].copy(), cache['sample_ids'].copy()
        metadata = json.loads(str(cache['metadata_json'].item()))
        cached_proxy = cache['proxy_ball'].copy()
    expected_n, expected_classes, _ = EXPECTED[args.dataset]
    assert len(whole) == expected_n and len(np.unique(labels)) == expected_classes
    assert metadata['checkpoint_hash'] == source_checkpoint_hash
    assert metadata['checkpoint_epoch'] == int(checkpoint['epoch'])
    source_paths = read_sample_map(args.sample_map, sample_ids, labels)
    identity = read_identity_map(args.identity_map)
    paths = resolve_paths(source_paths, args.dataset, args.images_root, identity)
    tangent = checkpoint['cluster_loss']['lcas'].detach().float().cpu().clone()
    model = OriginalWhole(hypnn, margs)
    network_state = unwrapped_state(checkpoint['stduent'])
    model.load_state_dict(network_state, strict=True)
    assert len(model.state_dict()) == 632
    assert all(torch.equal(value, network_state[key]) for key, value in model.state_dict().items()), \
        'A saved state value was changed during model loading.'
    model.requires_grad_(False)
    del checkpoint, network_state
    model.to(device).train(True)
    state_before = state_fingerprint(model)
    transform, clean_transform = build_transforms()
    anchor_validation = clean_anchor_check(model, paths, clean_transform, whole, device, args.microbatch,
                                          args.clean_anchors, args.clean_cosine_min, args.clean_max_abs)
    if state_fingerprint(model) != state_before:
        raise AssertionError('Network or BN changed during clean identity validation.')
    obj = official_loss_object(cls, hypnn, pmath, tangent, margin=float(margs.mrg))
    tangent_before = obj.lcas.detach().clone()
    with torch.no_grad():
        proxy = obj.to_hyperbolic(obj.lcas)
        proxy_distance = block_distance(proxy, proxy, pmath, .1)
        proxy_sim = torch.exp(-proxy_distance)
    assert np.max(np.abs(proxy.numpy() - cached_proxy)) < 2e-6
    depth = (2 / math.sqrt(.1) * torch.atanh(math.sqrt(.1) * proxy.norm(dim=1))).numpy()
    # Saved epoch is completed count; the next zero-based loop index is that count.
    epoch_index = int(metadata['checkpoint_epoch'])
    plan = sampler_plan(sampler_cls, labels, epoch_index, args.seed)
    full_batches = len(plan)
    batches = min(full_batches, args.max_batches) if args.max_batches else full_batches
    plan = plan[:batches]
    pool = ImagePool(paths, transform)
    loader = DataLoader(pool, batch_sampler=plan.tolist(), num_workers=args.workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed),
                        persistent_workers=args.workers > 0)
    amp = bool(args.precision == 'source' and margs.use_fp16 and device.type == 'cuda')
    counts = {key: np.zeros(512, np.int64) for key in ROLE_KEYS}
    gradients = {f'{source}_{mode}': {component: [] for component in ('norm', 'radial_signed', 'angular_norm')}
                 for source in ('sample', 'proxy') for mode in ('full_st', 'direct')}
    rows, whole_depth, gradient_vectors = [], [], {}
    for batch, (images, positions) in enumerate(loader):
        positions = positions.numpy()
        assert np.array_equal(positions, plan[batch])
        z = forward_images(model, images, device, args.microbatch, amp=amp)
        y = torch.from_numpy(labels[positions])
        batch_depth = (2 / math.sqrt(.1) * torch.atanh(math.sqrt(.1) * z.norm(dim=1))).numpy()
        if not np.isfinite(batch_depth).all():
            raise FloatingPointError('Whole depth is nonfinite.')
        whole_depth.append(batch_depth)
        np.random.seed(args.seed + batch)
        torch.manual_seed(args.seed + batch)
        with torch.no_grad():
            sample_distance = block_distance(z, proxy, pmath, .1)
            sample_sim = torch.exp(-block_distance(z, z, pmath, .1))
            sample_sim[y[:, None] == y[None, :]] += 1
            sample_indices = obj.get_reciprocal_triplets(sample_sim, topk=20, t_per_anchor=50)
            sample_state = torch.random.get_rng_state()
            _, sample_record = instrumented_hhc(sample_distance, sample_indices, margin=float(margs.mrg))
            proxy_indices = obj.get_reciprocal_triplets(proxy_sim, topk=20, t_per_anchor=50)
            proxy_state = torch.random.get_rng_state()
            _, proxy_record = instrumented_hhc(proxy_distance, proxy_indices, margin=float(margs.mrg))
        record_counts(counts, sample_record, 'sample')
        record_counts(counts, proxy_record, 'proxy')
        row = {'batch': batch, 'distinct_objects': int(np.unique(positions).size),
               'classes': int(np.unique(labels[positions]).size),
               'whole_depth_median': float(np.median(batch_depth)),
               **{f'{source}_{key}': record[key] for source, record in (('sample', sample_record), ('proxy', proxy_record))
                  for key in ('loss', 'triplets', 'self_k', 'collision_count')},
               **{f'{source}_positive_hinge_triplets': int(record['positive_hinge'].sum())
                  for source, record in (('sample', sample_record), ('proxy', proxy_record))}}
        if not np.isfinite(row['sample_loss']) or not np.isfinite(row['proxy_loss']):
            raise FloatingPointError('Nonfinite official HIER loss.')
        rows.append(row)
        if batch < args.gradient_batches:
            components, vectors, _ = gradient_probe(
                obj, pmath, z, sample_indices, proxy_indices, args.seed + batch,
                reference_check=False, selection_states={'sample': sample_state, 'proxy': proxy_state})
            for source_mode, component in components.items():
                for key, value in component.items():
                    gradients[source_mode][key].append(value)
            gradient_vectors.update({f'b{batch}_{key}': value for key, value in vectors.items()})
        print(json.dumps({'dataset': args.dataset, 'batch': batch + 1, 'batches': batches,
                          'sample_loss': row['sample_loss'], 'proxy_loss': row['proxy_loss'],
                          'seconds': round(time.time() - started, 1)}), flush=True)
    assert state_fingerprint(model) == state_before, 'Network/BN changed without authorization.'
    assert all(parameter.grad is None for parameter in model.parameters()), 'Unexpected encoder gradient.'
    assert torch.equal(obj.lcas.detach(), tangent_before), 'HIER tangent parameters changed.'
    assert obj.lcas.grad is None, 'autograd.grad should not accumulate .grad.'
    assert sha256(args.checkpoint) == checkpoint_hash and sha256(args.clean_cache) == cache_hash
    assert sha256(args.sample_map) == map_hash
    if args.identity_map:
        assert sha256(args.identity_map) == identity_hash
    result = summarize_counts(counts, gradients, depth)
    result.update(whole_depth_sampled_slots=basic_stats(np.concatenate(whole_depth)),
                  proxy_depth=basic_stats(depth), distinct_sampled_objects=int(np.unique(plan).size),
                  sampled_image_slots=int(plan.size), sampler_loop_epoch_index=epoch_index,
                  next_human_epoch_number=epoch_index + 1, full_source_epoch_batches=full_batches,
                  replay_batches=batches, full_epoch_completed=batches == full_batches,
                  gradient_batches=min(args.gradient_batches, batches))
    manifest = {'protocol': 'Fresh training augmentation; fixed best encoder/BN/HIER proxies; zero updates',
                'strict_original_resume': False, 'optimizer_updates': 0, 'encoder_gradients': False,
                'start_utc': start_time, 'end_utc': datetime.now(timezone.utc).isoformat(),
                'seconds': time.time() - started, 'dataset': args.dataset, 'seed': args.seed,
                'checkpoint_sha256': checkpoint_hash, 'source_checkpoint_sha256': source_checkpoint_hash,
                'derived_inference_checkpoint': checkpoint_hash != source_checkpoint_hash,
                'clean_cache_sha256': cache_hash, 'sample_map_sha256': map_hash,
                'identity_map_sha256': identity_hash, 'network_and_BN_state_fingerprint': state_before,
                'code_commit': args.code_commit, 'runner_sha256': sha256(__file__),
                'official_source_sha256': {name: sha256(Path(args.reference) / name) for name in
                    ('hier/losses.py', 'hier/sampler.py', 'hier/models/resnet.py', 'hier/models/model.py',
                     'hier/hyptorch/nn.py', 'hier/hyptorch/pmath.py', 'hier/helpers.py', 'hier/utils.py')},
                'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                            'torchvision': __import__('torchvision').__version__, 'PIL': __import__('PIL').__version__,
                            'encoder_device': str(device), 'HIER_device': 'CPU', 'encoder_autocast': amp,
                            'encoder_microbatch': args.microbatch, 'loader_workers': args.workers,
                            'physical_gpu_identity': args.gpu_identity,
                            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
                'configuration': {'global_batch': 180, 'local_batch_source': 90, 'source_world_size': 2,
                                  'IPC': 2, 'topk': 20, 'draws_per_anchor': 50, 'margin': margs.mrg,
                                  'tau': .1, 'c': .1, 'tangent_forward_cap': 2.3,
                                  'source_lambda_HIER': margs.lambda2, 'effective_bn_freeze': True,
                                  'source_checkpoint_use_fp16': bool(margs.use_fp16)},
                'clean_anchor_validation': anchor_validation, 'results': result,
                'limitations': ['Current best conditional usage, not saved historical usage or updated continuation.',
                                'PA classification proxy parameters were not saved; PA loss is not evaluated.',
                                'Fresh seeded RNG is not the missing original RNG/worker state.',
                                'Both source rank plans use independent initialized RNG; encoder is one fixed replica.',
                                'CUDA forward is optional; HIER distances and gradients are CPU FP32.',
                                'Original training AMP and old torchvision/PIL kernels are not bitwise reproduced.',
                                'Per-source gradients are total item gradients, including proxy-query endpoint paths.',
                                'Gradient evidence covers only the declared first gradient batches.']}
    (output / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'batches.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    write_proxy_csv(output / 'per_proxy.csv', depth, counts, gradients)
    with (output / 'batches.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(output / 'arrays.npz', depth=depth, sampled_positions=plan, sampled_ids=sample_ids[plan],
             whole_depth_slots=np.concatenate(whole_depth), **counts, **gradient_vectors,
             **{f'{source_mode}_{component}': np.stack(values)
                for source_mode, components in gradients.items() for component, values in components.items() if values})
    print(json.dumps({'complete': True, 'dataset': args.dataset, 'batches': batches,
                      'seconds': round(time.time() - started, 1), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
