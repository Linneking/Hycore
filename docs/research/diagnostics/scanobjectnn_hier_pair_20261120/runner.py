"""Native ScanObject PointMLP+HyCoRe, optionally with branch-only HIER."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from support import (REPO, SOURCE, SEED, SOURCE_COMMIT, Hierarchy, ObservedLoader,
    WithIds, inspect_data, load_original, now, seed_source, sha_file, source_text,
    state_sha, tensor_sha, training_function, write)


class Observer:
    def __init__(self, output, branch, torch, steps, smoke):
        self.output, self.branch, self.torch = output, branch, torch
        self.steps, self.smoke, self.epoch, self.updates = steps, smoke, 0, 0
        self.reset_epoch()

    def reset_epoch(self):
        self.batch = self.forwards = 0
        self.draws = Counter()
        self.id_hash = hashlib.sha256()
        self.base_sum = Counter()
        self.trace = {}
        self.last_input = {}
        self.branch.stats.clear()

    def observe_input(self, cloud, labels, ids):
        self.batch += 1
        self.forwards = 0
        self.branch.ids = ids
        values = ids.numpy().tobytes()
        self.id_hash.update(values)
        self.draws.update(ids.tolist())
        self.tracing = self.smoke or self.batch in (1, self.steps)
        self.trace = dict(epoch=self.epoch, batch=self.batch, ids_sha256=hashlib.sha256(values).hexdigest())
        if self.tracing:
            self.trace.update(cloud=tensor_sha(cloud), labels=tensor_sha(labels), ids=ids.tolist())

    def forward_pre(self, module, inputs):
        if not module.training:
            return
        self.forwards += 1
        if self.tracing:
            prefix = 'part' if self.forwards == 1 else 'whole'
            self.trace[prefix] = tensor_sha(inputs[0])
            self.trace[prefix+'_points'] = inputs[0].shape[-1]

    def forward_post(self, module, inputs, result):
        if module.training and self.tracing:
            self.trace['part_mu' if self.forwards == 1 else 'whole_mu'] = tensor_sha(result[0])

    def base_criterion(self, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            self.current_base = [float(result[4].detach()), float(result[5].detach())]
            self.base_sum['contrastive'] += self.current_base[0]
            self.base_sum['radial'] += self.current_base[1]
            return result
        return call

    def ce_criterion(self, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            self.base_sum['ce'] += float(result.detach())
            if self.tracing:
                self.trace['base_losses'] = [float(result.detach()), *self.current_base]
                with (self.output/'input_trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(self.trace, allow_nan=False)+'\n')
                self.last_input = dict(self.trace)
            return result
        return call

    def step(self):
        self.branch.step()
        self.updates += 1
        if self.batch % 10 == 0 or self.batch == self.steps:
            write(self.output/'heartbeat.json', dict(status='running', phase='train',
                epoch=self.epoch, completed_epochs=self.epoch-1, batch=self.batch,
                batches=self.steps, optimizer_steps=self.updates, updated_at=now()))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['SO-H0','SO-HIER'], required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--mode', choices=['reference','disabled','enabled'], default=None)
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def run(args):
    import torch
    import numpy as np
    import geoopt
    import h5py
    # Importing HIER before seeding is identical in both arms. Its optional
    # parameter initialization later uses a private CPU generator.
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
    original = load_original()
    arrays, dataset_identity = inspect_data(args.data_dir)
    import ScanObjectNN as data_module
    data_module.load_scanobjectnn_data = lambda partition: arrays[partition]
    mode = args.mode or ('enabled' if args.arm == 'SO-HIER' else 'reference')
    if not args.smoke and mode != ('enabled' if args.arm == 'SO-HIER' else 'reference'):
        raise ValueError('Production arm/mode mismatch')
    if args.output.exists():
        raise FileExistsError('Every run requires a new output: '+str(args.output))
    args.output.mkdir(parents=True)
    seed_source(torch, np)
    model = original.Hype_pointMLP(num_classes=15).to('cuda')
    torch.backends.cudnn.benchmark = True
    initial_sha = state_sha(model)
    torch.save(dict(net=model.state_dict(), model_tensor_sha256=initial_sha), args.output/'initial_model.pth')
    branch = Hierarchy(mode == 'enabled', args.output, torch)
    rows = len(arrays['training'][1])
    steps, epochs = (2, 2) if args.smoke else (rows//32, 300)
    observer = Observer(args.output, branch, torch, steps, args.smoke)
    model.register_forward_pre_hook(observer.forward_pre)
    model.register_forward_hook(observer.forward_post)
    train_dataset = WithIds(original.ScanObjectNN(num_points=1024, partition='training'))
    test_dataset = None if args.smoke else original.ScanObjectNN(num_points=1024, partition='test')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=32, shuffle=True, drop_last=True)
    test_loader = None if args.smoke else DataLoader(test_dataset, num_workers=8, batch_size=32, shuffle=True, drop_last=False)
    class CountedSGD(geoopt.optim.RiemannianSGD):
        def step(self, closure=None):
            result = super().step(closure)
            observer.step()
            return result
    optimizer = CountedSGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=.001)
    if mode != 'reference':
        original.branch = branch
        exec(compile(training_function(source_text(), True), original.__file__, 'exec'), vars(original))
    if mode == 'reference' and original.train.__code__.co_firstlineno != 201:
        raise RuntimeError('Unexpected original training function identity')
    config = dict(backbone='ScanObject official PointMLP+HyCoRe', official_source_commit=SOURCE_COMMIT,
        num_classes=15, base_c=1., base_extra_cap=None, base_output_GS=False,
        batch=32, test_batch=32, workers=8, sampler='SH32 shuffle/drop_last', seed=SEED,
        epochs=300, steps_per_epoch=rows//32, planned_updates=300*(rows//32),
        main_optimizer=dict(name='RiemannianSGD',lr=.1,min_lr=.001,momentum=.9,weight_decay=.0001,clip_norm=1.),
        whole_points=[600,1024], part_points='200..floor((whole_K-1)/2)',
        margin_radial='500/Npart', margin_contrastive=4., weight_contrastive=.01, weight_radial=.001,
        smoothing=.2, test_selection='strictly higher rounded test OA; earliest tie; no early stop',
        hier_enabled=branch.enabled, hier_c=.1 if branch.enabled else None,
        hier_view='native_c1_logmap -> tangent_cap2.3 -> c0.1_expmap -> one_branch_GS' if branch.enabled else None,
        fixed_shell=False, proxy_count=512 if branch.enabled else 0, sample_K=20 if branch.enabled else None,
        proxy_K=20 if branch.enabled else None, T=50 if branch.enabled else None,
        lambda_sample=.5 if branch.enabled else 0., lambda_proxy=.5 if branch.enabled else 0.,
        proxy_optimizer=dict(name='AdamW',lr=.005,min_lr=.0005,eps=.0001,betas=[.9,.999],weight_decay=.01) if branch.enabled else None,
        smoke=args.smoke, mode=mode, benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic, bitwise_replay_claimed=False)
    manifest = dict(status='running', arm=args.arm, started_at=now(), updated_at=now(),
        pid=os.getpid(), commit=args.source_commit, production_source_branch='codex/scanobjectnn-hier-pair-plan-20261120',
        official_source_commit=SOURCE_COMMIT, initialized_model_sha256=initial_sha,
        dataset_identity=dataset_identity, training_config=config, teacher=None, initialization_checkpoint=None,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), physical_gpus=os.environ.get('HYCORE_PHYSICAL_GPUS'),
        physical_gpu_uuids=os.environ.get('HYCORE_PHYSICAL_GPU_UUIDS'),
        source_sha256={str(p.relative_to(REPO)):sha_file(p) for p in [HERE/'runner.py',HERE/'support.py',SOURCE/'main_pointmlp_hycore.py',SOURCE/'models/pointmlp.py',SOURCE/'hutil.py']},
        library_versions=dict(torch=torch.__version__,geoopt=geoopt.__version__,numpy=np.__version__,h5py=h5py.__version__),
        completed_epochs=0, optimizer_steps=0)
    write(args.output/'manifest.json', manifest)
    best = -1.
    for epoch in range(1, epochs+1):
        observer.epoch = epoch
        observer.reset_epoch()
        start = time.monotonic()
        effective_lr = optimizer.param_groups[0]['lr']
        proxy_lr = branch.optimizer.param_groups[0]['lr'] if branch.enabled else None
        loader = ObservedLoader(train_loader, observer, 2 if args.smoke else None)
        result = original.train(model, loader, optimizer, observer.ce_criterion(original.cal_loss),
            observer.base_criterion(original.hype_triplet_losses), 'cuda')
        training = result[-1]
        del result
        if observer.batch != steps or observer.updates != epoch*steps:
            raise RuntimeError('Training update budget mismatch')
        if not args.smoke and len(observer.draws) != steps*32:
            raise RuntimeError('Natural shuffle contains duplicate row IDs')
        write(args.output/'heartbeat.json',dict(status='running',phase='test' if not args.smoke else 'train_only_smoke',
            epoch=epoch, completed_epochs=epoch-1, optimizer_steps=observer.updates, updated_at=now()))
        testing = None if args.smoke else original.validate(model, test_loader, original.cal_loss, 'cuda')
        scheduler.step()
        if branch.enabled:
            branch.scheduler.step()
        selected = not args.smoke and testing['acc'] > best
        if selected:
            best = testing['acc']
            manifest.update(best_epoch=epoch,best_test_OA=best,best_test_AA_same_epoch=testing['acc_avg'])
        finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        if not finite:
            raise RuntimeError('Nonfinite model weights')
        bn = sorted({int(v) for n,v in model.named_buffers() if n.endswith('num_batches_tracked')})
        if bn != [2*epoch*steps]:
            raise RuntimeError('Unexpected BN forward count: '+str(bn))
        metric = dict(epoch=epoch,completed_epochs=epoch,train=training,test=testing,
            optimizer_steps=observer.updates,epoch_optimizer_steps=steps,
            effective_lr=effective_lr,lr_after_scheduler=optimizer.param_groups[0]['lr'],proxy_lr=proxy_lr,
            wall_seconds=time.monotonic()-start,selected_best=selected,finished_at=now(),
            base_loss_means={k:v/steps for k,v in observer.base_sum.items()},
            hierarchy=dict(branch.stats),hier_last=branch.last if branch.enabled else None,
            sample_coverage=dict(total_positions=sum(observer.draws.values()),unique_rows=len(observer.draws),
                ordered_row_ids_sha256=observer.id_hash.hexdigest()),BN_counts=bn,finite=True)
        coverage = [observer.draws[i] for i in range(rows)]
        write(args.output/('sampler_epoch_%03d.json'%epoch),dict(epoch=epoch,draw_counts=coverage,ordered_row_ids_sha256=observer.id_hash.hexdigest()))
        rng = dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
        state = dict(net=model.state_dict(),epoch=epoch,completed_epochs=epoch,optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),metrics=metric,rng=rng,config=config,**branch.state())
        torch.save(state,args.output/'last.pth')
        if selected:
            torch.save(state,args.output/'best.pth')
        if args.smoke or epoch in (1,5,99) or epoch%20==0 or epoch>=291:
            torch.save(state,args.output/('checkpoint_epoch_%03d.pth'%epoch))
        del state
        write(args.output/('metrics_epoch_%03d.json'%epoch),metric)
        with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metric,allow_nan=False)+'\n')
        manifest.update(completed_epochs=epoch,optimizer_steps=observer.updates,updated_at=now())
        write(args.output/'manifest.json',manifest)
        write(args.output/'heartbeat.json',dict(status='running',phase='epoch_complete',completed_epochs=epoch,
            optimizer_steps=observer.updates,updated_at=now()))
        print(json.dumps(dict(arm=args.arm,epoch=epoch,train=training,test=testing,seconds=metric['wall_seconds'])),flush=True)
    manifest.update(status='completed',finished_at=now(),last_sha256=sha_file(args.output/'last.pth'))
    if not args.smoke:
        manifest['best_sha256']=sha_file(args.output/'best.pth')
    write(args.output/'manifest.json',manifest)
    write(args.output/'heartbeat.json',dict(status='completed',completed_epochs=epochs,optimizer_steps=observer.updates,updated_at=now()))


if __name__ == '__main__':
    args = arguments()
    try:
        run(args)
    except BaseException as error:
        if args.output.exists():
            write(args.output/'failure.json',dict(error=repr(error),time=now()))
        raise
