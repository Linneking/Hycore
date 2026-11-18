"""Versioned MN20 SH/CB controls and full-MN40 independently sampled HIER."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
PREVIOUS = HERE.parent / 'mn40_ts33_35_20261113/shuffle_runner.py'
spec = importlib.util.spec_from_file_location('mn40_ts35_reused', PREVIOUS)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.ts06
KEEP = (3, 5, 7, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 23, 25, 26, 33, 34, 35, 36)
ARMS = ('TS51', 'TS52', 'TS53', 'TS54', 'TS55')


class WithIds:
    def __init__(self, dataset, ids):
        self.dataset, self.ids = dataset, ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        cloud, label = self.dataset[index]
        return cloud, label, int(self.ids[index])


def replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Missing unique pinned boundary: ' + before)
    return source.replace(before, after, 1)


def main_source(source):
    for before, after in [
        ('"steps_per_epoch": 2 if args.smoke else 307', '"steps_per_epoch": 2 if args.smoke else exp.steps'),
        ('"train_size": 9840', '"train_size": exp.train_size'),
        ('"test_size": 0 if args.smoke else 2468', '"test_size": 0 if args.smoke else exp.test_size'),
        ('original.Hype_pointMLP()', 'original.Hype_pointMLP(num_classes=exp.num_classes)'),
        ('if len(train_set) != 9840:', 'if len(train_set) != exp.train_size:'),
        ('if len(test_set) != 2468:', 'if len(test_set) != exp.test_size:'),
        ('train_loader = DataLoader(train_set, num_workers=8, batch_size=32, shuffle=True, drop_last=True)',
         'train_loader = exp.make_train_loader(train_set)'),
        ('for epoch in range(args.epochs):\n            started',
         'for epoch in range(args.epochs):\n            exp.start_epoch(epoch)\n            started'),
        ('if current["epoch_steps"] != (2 if args.smoke else 307):',
         'if current["epoch_steps"] != (2 if args.smoke else exp.steps):'),
    ]:
        source = replace(source, before, after)
    return source


def train_source(source, experiment):
    import ast
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    source = ast.get_source_segment(source, node)
    source = replace(source, '(data, label) in enumerate(trainloader)', '(data, label, object_ids) in enumerate(trainloader)')
    source = replace(source, '        data, label = data.to(device)',
        '        exp.observe_batch(data, label, object_ids)\n        data, label = data.to(device)')
    source = replace(source, '        pos_mu, _ = net(pos_data, emb=True)',
        '        exp.observe_crops(data, pos_data)\n        pos_mu, _ = net(pos_data, emb=True)')
    source = replace(source, '        loss.backward()',
        '        exp.observe_base(std_loss, t_loss, h_loss)\n        loss.backward()')
    if experiment.enabled and experiment.arm != 'TS53':
        source = replace(source, '        parent_mu, logits = net(data)',
            '        parent_mu, logits = exp.whole(net, data)')
        source = replace(source, '        loss.backward()',
            '        loss = exp.add_loss(loss, label)\n        loss.backward()')
    if experiment.enabled and experiment.arm == 'TS53':
        source = replace(source, '        loss.backward()',
            '        loss.backward()\n        exp.aux_backward(net)')
        source = replace(source, '        train_loss += loss.item()',
            '        train_loss += loss.item() + exp.last_aux_loss')
    source = replace(source, '        optimizer.step()', '        optimizer.step()\n        exp.step()')
    return source


class Experiment(previous.Hierarchy):
    def __init__(self, mode, args, arm):
        super().__init__(mode, args, 'TS35')
        self.arm = arm
        self.enabled = mode == 'enabled' and arm in ('TS52', 'TS53', 'TS55')
        self.filtered = arm != 'TS53'
        self.balanced = arm in ('TS54', 'TS55')
        self.num_classes = 20 if self.filtered else 40
        self.train_size, self.test_size = (4298, 1398) if self.filtered else (9840, 2468)
        self.steps = self.train_size // 32
        self.epoch = self.in_epoch = 0
        self.draws, self.aux_draws = Counter(), Counter()
        self.last_aux_loss = 0.
        self.main_sampler = self.aux_sampler = None
        self.aux_state = None

    def load_data_root(self, module, directory):
        import numpy as np
        loaded = self.original_data_root(module, directory)
        load = module.load_data
        dataset_type = module.ModelNet40

        def filtered_load(partition):
            cloud, labels = load(partition)
            original_size = len(labels)
            ids = np.arange(original_size)
            if self.filtered:
                mask = np.isin(labels.reshape(-1), KEEP)
                ids, cloud, labels = ids[mask], cloud[mask], labels[mask]
                mapping = np.full(40, -1, dtype=np.int64)
                mapping[list(KEEP)] = np.arange(20)
                labels = mapping[labels]
            if partition == 'train':
                self.train_ids, self.train_cloud, self.train_labels = ids, cloud, labels.reshape(-1)
            loaded[partition].update(original_size=original_size, size=len(labels),
                filtered=self.filtered, kept_original_labels=list(KEEP) if self.filtered else list(range(40)),
                selected_indices_sha256=hashlib.sha256(ids.astype('int64').tobytes()).hexdigest(),
                selected_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                class_counts=np.bincount(labels.reshape(-1), minlength=self.num_classes).tolist())
            return cloud, labels

        def dataset(num_points, partition='train'):
            value = dataset_type(num_points=num_points, partition=partition)
            return WithIds(value, self.train_ids) if partition == 'train' else value

        module.load_data, module.ModelNet40 = filtered_load, dataset
        self.data_module = module
        return loaded

    def make_train_loader(self, dataset):
        from torch.utils.data import DataLoader
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        if self.balanced:
            self.main_sampler = SourceClassBatchSampler(self.train_labels, world_size=1,
                classes_per_rank=16, instances_per_class=2, steps=self.steps, seed=4780)
            return DataLoader(dataset, num_workers=8, batch_sampler=self.main_sampler)
        return DataLoader(dataset, num_workers=8, batch_size=32, shuffle=True, drop_last=True)

    def start_epoch(self, epoch):
        self.epoch, self.in_epoch = epoch, 0
        self.draws.clear()
        self.aux_draws.clear()
        if self.main_sampler is not None:
            self.main_sampler.set_epoch(epoch)

    def observe_batch(self, cloud, labels, ids):
        super().observe_input(cloud, labels)
        self.draws.update(ids.tolist())
        self.current_ids = ids.clone()
        self.hier_ids = ids.clone()
        self.counts['main_same_class_flip_positions'] += int((labels.reshape(-1)==labels.reshape(-1).flip(0)).sum())
        if self.observation:
            self.observation.update(ids=ids.tolist(), ids_sha256=previous.tensor_sha(ids))
        self.last_aux_loss = 0.

    def start(self, torch):
        super().start(torch)
        original_forward = self.proxy.forward
        def identified_forward(*args, **kwargs):
            kwargs['data_ids'] = self.hier_ids
            return original_forward(*args, **kwargs)
        self.proxy.forward = identified_forward
        if self.arm == 'TS53':
            import numpy as np
            self.aux_state = dict(python=random.Random(4780 + 500001).getstate(),
                numpy=np.random.RandomState(4780 + 500002).get_state(),
                torch_cpu=torch.Generator(device='cpu').manual_seed(4780 + 500003).get_state(),
                torch_cuda=[torch.Generator(device='cuda').manual_seed(4780 + 500004).get_state()])

    @contextmanager
    def auxiliary_randomness(self):
        import torch
        import numpy as np
        ambient = base.rng_state(torch, np)
        def restore(state):
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])
        restore(self.aux_state)
        try:
            yield
        finally:
            self.aux_state = base.rng_state(torch, np)
            restore(ambient)

    def aux_backward(self, net):
        import torch
        import numpy as np
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        if self.aux_sampler is None:
            self.aux_sampler = SourceClassBatchSampler(self.train_labels, world_size=1,
                classes_per_rank=16, instances_per_class=2, steps=self.steps, seed=4780)
        plan = self.aux_sampler.global_plan(self.in_epoch, epoch=self.epoch)
        ids = np.asarray(plan.indices, dtype=np.int64)
        self.aux_draws.update(ids.tolist())
        buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
        with self.auxiliary_randomness():
            clouds = []
            for index in ids:
                cloud = self.data_module.translate_pointcloud(self.train_cloud[index, :1024])
                np.random.shuffle(cloud)
                clouds.append(cloud)
            data = torch.from_numpy(np.stack(clouds)).to('cuda').permute(0, 2, 1)
            _, whole, count = self.original_module.get_children_np(data, kmin=800, kmax=1024)
            _, part, _ = self.original_module.get_children_np(whole, starting=count, kmin=200, kmax=600)
            labels = torch.as_tensor(self.train_labels[ids], device='cuda')
            self.hier_ids = torch.as_tensor(ids)
            try:
                self.geometry.capture_enabled = True
                net(whole, emb=True)
                self.pre_gs = self.geometry.pre_gs
            finally:
                self.geometry.clear()
            loss = super().add_loss(self.pre_gs.sum() * 0., labels)
            self.last_aux_loss = float(loss.detach())
            loss.backward()
            if self.smoke or self.updates == 0:
                trace = dict(epoch=self.epoch, step=self.in_epoch, ids=ids.tolist(),
                    labels=labels.detach().cpu().tolist(), whole=previous.tensor_sha(whole),
                    part=previous.tensor_sha(part), hier_loss=self.last_aux_loss)
                with (self.output / 'aux_trace.jsonl').open('a') as f:
                    f.write(json.dumps(trace) + '\n')
        with torch.no_grad():
            for name, value in net.named_buffers():
                value.copy_(buffers[name])
            if self.smoke and any(not torch.equal(v, buffers[n]) for n, v in net.named_buffers()):
                raise RuntimeError('Auxiliary forward changed persistent BN state')
        self.counts['aux_forward_updates'] += 1
        self.counts['aux_object_positions'] += 32

    def step(self):
        super().step()
        self.in_epoch += 1

    def checkpoint(self, net, epoch, optimizer, scheduler, metrics, best, torch, np, config):
        metrics['sample_coverage'] = dict(total_positions=sum(self.draws.values()),
            unique_objects=len(self.draws), id_draw_counts=[self.draws[int(i)] for i in self.train_ids])
        if self.arm == 'TS53' and self.enabled:
            metrics['aux_sample_coverage'] = dict(total_positions=sum(self.aux_draws.values()),
                unique_objects=len(self.aux_draws), id_draw_counts=[self.aux_draws[i] for i in range(self.train_size)])
        state = super().checkpoint(net, epoch, optimizer, scheduler, metrics, best, torch, np, config)
        if self.aux_state is not None:
            state['aux_rng'] = self.aux_state
        state['epoch_sampler'] = None if self.main_sampler is None else self.main_sampler.state_dict()
        return state


def canonical(arm):
    filtered = arm != 'TS53'
    sampling = 'CB' if arm in ('TS54', 'TS55') else 'SH'
    steps = 134 if filtered else 307
    dataset = 'MN40F20N130_400' if filtered else 'MN40'
    hier = 'H0' if arm in ('TS51', 'TS54') else 'Hshell-A1-Ks20-Kp20-L0.5-W0'
    if arm == 'TS53':
        hier += '-AuxFreshCB32-BNrestore'
    return f'HC-C01-Cap23-GS1-M500_2-{sampling}32x{steps}-{dataset} / {hier} (S4780)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--mode', choices=('reference', 'off', 'enabled'), default='enabled')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--source-commit')
    options = parser.parse_args()
    if options.mode != 'enabled' and not options.smoke:
        parser.error('Reference/off modes are train-only smoke checks')
    remaining = ['--variant', 'c01-cap', '--data-dir', str(options.data_dir), '--run-dir', str(options.output)]
    if options.smoke:
        remaining += ['--smoke', '--epochs', '2', '--steps-per-epoch', '2']
    if options.source_commit:
        remaining += ['--source-commit', options.source_commit]
    args = base.argument_parser().parse_args(remaining)
    exp = Experiment(options.mode, args, options.arm)
    source = previous.pinned_text(previous.ORIGINAL, previous.TRAIN_SHA)
    adapter = previous.pinned_text(base.__file__, previous.ADAPTER_SHA)
    original_import, original_install = base.import_original, base.install_loss_curvature
    original_identity, original_write = base.source_identity, base.atomic_json
    exp.original_configure, exp.original_checkpoint = base.configure_model, base.checkpoint_state
    exp.original_data_root = base.install_data_root
    base.exp = exp
    # The versioned adapter changes only the recorded task size, class head and loader.
    exec(compile(main_source(adapter), base.__file__, 'exec'), vars(base))

    def import_original(repo):
        original = original_import(repo)
        original.exp = exp
        exec(compile(train_source(source, exp), str(previous.ORIGINAL), 'exec'), vars(original))
        exp.original_module = original
        return original

    def install(hutil):
        original_install(hutil)
        if exp.enabled:
            import torch
            exp.start(torch)

    def identity(repo):
        result = original_identity(repo)
        for path in [Path(__file__), PREVIOUS]:
            result[str(path.relative_to(repo))] = base.digest(path)
        return result

    def write(path, value):
        if path.name in ('manifest.json', 'config.json'):
            config = value.get('training_config', value)
            config.update(experiment_id=options.arm, num_classes=exp.num_classes,
                sampler='SourceClassBatchSampler16x2' if exp.balanced else 'original SH32 no replacement',
                world_size=1, global_batch=32, hier_enabled=exp.enabled, mode=options.mode,
                dataset_task='MN40-F20-N130_400' if exp.filtered else 'MN40',
                selection='none_train_only_smoke' if args.smoke else 'best_rounded_test_OA_original',
                sample_K=20, proxy_K=20, lambda_hier=.5 if exp.enabled else 0.,
                proxy_count=512 if exp.enabled else 0, hier_view_depth=4.6 if exp.enabled else None,
                hier_input='fresh independently sampled CB32' if options.arm=='TS53' else 'current main batch',
                auxiliary_BN='training batch statistics; restore persistent buffers after backward' if options.arm=='TS53' else None,
                auxiliary_stream_seeds=[504781,504782,504783,504784] if options.arm=='TS53' else None,
                proxy_optimizer=dict(lr=.005,eps=1e-4,min_lr=.0005,weight_decay=.01) if exp.enabled else None)
            if path.name=='manifest.json':
                value.update(experiment_id=options.arm,canonical_name=canonical(options.arm),
                    changed_from_baseline=['filtered twenty-class task' if exp.filtered else 'HIER uses separately encoded CB32'],
                    initialization_checkpoint=None,teacher=None)
        original_write(path,value)

    base.import_original, base.install_loss_curvature = import_original, install
    base.configure_model, base.checkpoint_state = exp.configure_model, exp.checkpoint
    base.install_data_root, base.source_identity, base.atomic_json = exp.load_data_root, identity, write
    base.main(remaining)


if __name__ == '__main__':
    main()
