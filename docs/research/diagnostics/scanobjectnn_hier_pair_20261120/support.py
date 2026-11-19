"""Source-matched ScanObjectNN support; no CUDA allocation at import time."""
from __future__ import annotations
import ast
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / 'classification_ScanObject'
sys.path.insert(0, str(REPO))
SEED = 4780
SOURCE_COMMIT = 'ef56a73020e309b1009c486fea1f601d3f7b3252'
TRAIN_SHA = '6b0ee90db0628c5c2e3fd46fee7f656d779171fce316c2afbe65b573c2e1598b'
DATA_FILES = {
    'training': ('training_objectdataset_augmentedrot_scale75.h5', 'bd1ebda43155b112fac0ec9335723e66c4885449022002a2a8d370b56376eed5', 11416),
    'test': ('test_objectdataset_augmentedrot_scale75.h5', 'f23822ee2f1e1d18331e91334d74adb10c8718d72d503bfa76ebe0a9348a7ea1', 2882),
}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_text():
    text = (SOURCE / 'main_pointmlp_hycore.py').read_text(encoding='utf-8')
    if hashlib.sha256(text.encode()).hexdigest() != TRAIN_SHA:
        raise RuntimeError('Official ScanObject training source changed')
    return text


def training_function(text, enabled):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    text = ast.get_source_segment(text, node)
    if enabled:
        old = '        loss = std_loss + 0.01*t_loss + 0.001*h_loss'
        if text.count(old) != 1:
            raise RuntimeError('Missing unique source loss boundary')
        text = text.replace(old, old + '\n        loss = branch.add_loss(loss, parent_mu, label, net.manifold2)', 1)
    return text


def load_original():
    # Published ScanObject modules retain an old package name. A namespace
    # alias resolves only those imports; all source model operations remain.
    sys.path.insert(0, str(REPO / 'pointnet2_ops_lib'))
    sys.path.insert(0, str(SOURCE))
    alias = types.ModuleType('modelsnet')
    alias.__path__ = [str(SOURCE / 'models')]
    sys.modules['modelsnet'] = alias
    original = types.ModuleType('scanobject_source_reference')
    original.__file__ = str(SOURCE / 'main_pointmlp_hycore.py')
    text = source_text()
    assignment = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"'
    if text.count(assignment) != 1:
        raise RuntimeError('GPU assignment boundary changed')
    # The worker is already bound to the verified physical UUID by its queue.
    text = text.replace(assignment, '# Device binding supplied by isolated launcher', 1)
    exec(compile(text, original.__file__, 'exec'), vars(original))
    return original


def seed_source(torch, np):
    # Same source RNG families, overridden to the user's common root seed.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)


def inspect_data(directory):
    import h5py
    import numpy as np
    directory = Path(directory)
    arrays, records = {}, {}
    for partition, (name, expected_hash, expected_rows) in DATA_FILES.items():
        path = directory / name
        actual_hash = sha_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError('Dataset hash mismatch: ' + name)
        with h5py.File(path, 'r') as hdf:
            cloud = hdf['data'][:].astype('float32')
            label = hdf['label'][:].astype('int64')
        if cloud.shape != (expected_rows, 2048, 3) or label.shape != (expected_rows, 1):
            raise RuntimeError('Dataset shape mismatch: ' + name)
        if not np.isfinite(cloud).all() or set(label.reshape(-1).tolist()) != set(range(15)):
            raise RuntimeError('Invalid coordinates or label identity')
        arrays[partition] = (cloud, label)
        records[partition] = dict(file=name, sha256=actual_hash, rows=len(label),
            data_shape=list(cloud.shape), label_shape=list(label.shape),
            class_counts=np.bincount(label.reshape(-1), minlength=15).tolist())
    return arrays, records


class WithIds:
    def __init__(self, source):
        self.source = source
    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        cloud, label = self.source[index]
        return cloud, label, index


class ObservedLoader:
    def __init__(self, loader, observer, limit=None):
        self.loader, self.observer, self.limit = loader, observer, limit
    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit else len(self.loader)
    def __iter__(self):
        import itertools
        rows = iter(self.loader)
        if self.limit:
            rows = itertools.islice(rows, self.limit)
        for cloud, labels, ids in rows:
            self.observer.observe_input(cloud, labels, ids)
            yield cloud, labels


def hier_view(mu, native_manifold):
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
    # Native c1 radius is retained up to a *branch-only* tangent cap. This
    # is differentiable; there is no fixed shell, detach, or replacement mu.
    return to_ball(native_manifold.logmap0(mu), c=.1, clip_r=2.3, hook=True)


class Hierarchy:
    def __init__(self, enabled, output, torch, epochs=300):
        self.enabled, self.output, self.torch = enabled, Path(output), torch
        self.proxy = self.optimizer = self.scheduler = self.generator = None
        self.updates = 0
        self.stats = Counter()
        self.last = {}
        if enabled:
            from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
            self.proxy = HIERLoss(seed=SEED).to('cuda')
            self.optimizer = torch.optim.AdamW(self.proxy.parameters(), lr=.005,
                eps=1e-4, betas=(.9,.999), weight_decay=.01)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=.0005)
            self.generator = torch.Generator(device='cuda').manual_seed(SEED + 300001)
            torch.save(self.proxy.state_dict(), self.output / 'initial_proxies.pth')

    def add_loss(self, base, mu, labels, manifold):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import depth
        if not self.enabled:
            return base
        self.optimizer.zero_grad(set_to_none=True)
        view = hier_view(mu, manifold)
        regularizer, stats = self.proxy(view, labels, topk=20, proxy_topk=20,
            seed=SEED + self.updates * 10007, generator=self.generator,
            data_ids=self.ids, monitor_structure=False)
        if not bool(self.torch.isfinite(regularizer)):
            raise RuntimeError('Nonfinite HIER loss')
        for prefix in ('sample','proxy'):
            if stats[prefix]['self_k_triplets']:
                raise RuntimeError('Invalid self-negative relationship')
            for key in ('triplets','active_triplets','noncollision_triplets','collisions'):
                self.stats[prefix+'_'+key] += stats[prefix][key]
        self.stats['sample_loss_sum'] += stats['sample_loss']
        self.stats['proxy_loss_sum'] += stats['proxy_loss']
        self.last = dict(sample_loss=stats['sample_loss'], proxy_loss=stats['proxy_loss'],
            whole_view=stats['geometry']['whole'], proxy=stats['geometry']['proxy'],
            native_whole_depth_max=float(manifold.dist0(mu.detach()).max()))
        if self.updates == 0:
            self.torch.save(dict(native_mu=mu.detach().cpu(), hier_view=view.detach().cpu(),
                labels=labels.detach().cpu(), proxy_tangent=self.proxy.tangent_proxies.detach().cpu(),
                graph_connected=bool(view.requires_grad), fixed_shell=False), self.output/'first_hier_view.pth')
        return base + .5 * regularizer

    def step(self):
        if self.enabled:
            self.optimizer.step()
            self.proxy.assert_geometry()
        self.updates += 1

    def state(self):
        if not self.enabled:
            return dict(proxy=None, proxy_optimizer=None, proxy_scheduler=None, gumbel_rng=None)
        return dict(proxy=self.proxy.state_dict(), proxy_optimizer=self.optimizer.state_dict(),
            proxy_scheduler=self.scheduler.state_dict(), gumbel_rng=self.generator.get_state())
