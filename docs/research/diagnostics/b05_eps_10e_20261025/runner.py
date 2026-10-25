"""Paired B05 CB64 train-only diagnostics: LR.005, epsilon1e-8 or1e-4.

The reviewed continuous B05 loop remains the source of all training operations.
This successor removes evaluation, stops at10epochs, retains300-epoch cosines,
and observes actual updates without changing gradients, buffers or random flow.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ORIGINAL=REPO/'inter_hierarchy_MN40/hier_proxy_cycle_v9c/train.py'
AUDIT=REPO/'inter_hierarchy_MN40/hier_proxy_cycle_v9c/parameter_audit.py'
EXPECTED_SOURCE='4754913a64bb355080e55853e9a502d211d6153fbb6a23cd0c182dad437c7e56'
EXPECTED_AUDIT='a7473905e3183926705ad7789a5c773b8dbfc658441800797347e719c5f4cecc'
VERSION='b05-cb64-shell-eps-only-train-only-10e-1'
INITIAL_DIGEST='8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d'
EPSILONS={'small':1e-8,'large':1e-4}


def sha_text(source): return hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest()


def replace_once(source, old, new):
    if source.count(old)!=1: raise RuntimeError('Diagnostic edit boundary differs: '+old[:100])
    return source.replace(old,new,1)


def replace_region(source,start,end,new):
    if source.count(start)!=1 or source.count(end)!=1: raise RuntimeError('Diagnostic region differs')
    left,right=source.index(start),source.index(end)
    if right<=left: raise RuntimeError('Diagnostic region order differs')
    return source[:left]+new+source[right:]


def transformed_source(source,eps='small'):
    if eps not in EPSILONS: raise ValueError('Expected small or large epsilon')
    source=source.replace('\r\n','\n')
    if sha_text(source)!=EXPECTED_SOURCE: raise RuntimeError('Reviewed B05 parent SHA differs')
    changes=(
      ('CHECKPOINT_VERSION = "hycore-cycle-v9c-c01-cap23-gs1-m5002-testselect-1"',f'CHECKPOINT_VERSION = "{VERSION}"'),
      ('    args = arguments()\n','    args = diagnostic_arguments()\n'),
      ('    config = fixed_configuration(args)\n','    config = diagnostic_configuration(args, fixed_configuration(args))\n'),
      ('"main_training": not args.smoke, "diagnostic_only": args.smoke,','"main_training": False, "diagnostic_only": True,'),
      ('ClassCycleBatchSampler(labels[train_ids],','SourceClassBatchSampler(labels[train_ids],'),
      ('proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=.01)',f'proxy_optimizer = torch.optim.AdamW(proxy.parameters(), lr=.005, betas=(.9, .999), eps={EPSILONS[eps]!r}, weight_decay=.01)'),
      ('        generators = make_generators(args.seed, rank, device, enabled)\n','        generators = make_generators(args.seed, rank, device, enabled)\n        observer.bind(proxy, proxy_optimizer, args.run_dir, rank, manifest, config)\n'),
      ('        for epoch_index in range(start_epoch, args.epochs):','        for epoch_index in range(start_epoch, args.diagnostic_epochs):'),
      ('                whole_count, child_count = shared_crop_counts(generators["crop"], device)\n','                whole_count, child_count = shared_crop_counts(generators["crop"], device)\n                observer.begin(epoch, step + 1)\n'),
      ('whole_count, child_count, topk=20, seed=args.seed + epoch_index * 10000019, lambda_hier=.5)','whole_count, child_count, topk=20, seed=args.seed + (epoch_index * args.steps_per_epoch + step) * 10007, lambda_hier=.5, gumbel_state=generators["gumbel"].get_state().clone())'),
      ('                optimizer.step()\n','                optimizer.step()\n                proxy_observation = observer.before_step(proxy, proxy_optimizer)\n'),
      ('                    proxy_cap_stats = proxy.project_after_step(proxy_optimizer)\n','                    proxy_cap_stats = proxy.project_after_step(proxy_optimizer)\n                    observer.after_step(proxy_observation, proxy, proxy_optimizer)\n'),
      ('            rank_states = [None, None]\n','            observer.end_epoch(epoch)\n            rank_states = [None, None]\n'),
      ('                    print(json.dumps({"epoch": epoch, "selection_domain": selection_prefix,\n                        "selection_oa": selection[selection_prefix + "_oa"], "best": best}, allow_nan=False), flush=True)','                    print(json.dumps({"epoch": epoch, "diagnostic_train_only": True, "selection": None}, allow_nan=False), flush=True)'),
      ('manifest.update(status="completed", completed_epochs=args.epochs, optimizer_steps=args.epochs * args.steps_per_epoch, completed_optimizer_steps=args.epochs * args.steps_per_epoch,','manifest.update(status="completed", completed_epochs=args.diagnostic_epochs, optimizer_steps=args.diagnostic_epochs * args.steps_per_epoch, completed_optimizer_steps=args.diagnostic_epochs * args.steps_per_epoch,'),
      ('best=best, final_test=test_metrics, final_summary_skipped=args.smoke, test_used_for_selection=not args.smoke)','best=None, final_test=None, final_summary_skipped=True, test_used_for_selection=False)'),
      ('{"status": "completed", "epoch": args.epochs,','{"status": "completed", "epoch": args.diagnostic_epochs,'),
      ('print("V9C_CYCLE_B64_" + args.hier.upper() + "_COMPLETED", flush=True)','print("B05_EPS_ONLY_DIAGNOSTIC_COMPLETED", flush=True)'),
    )
    for old,new in changes: source=replace_once(source,old,new)
    source=replace_region(source,'        if args.smoke:\n            selection_ids = train_ids[:args.eval_batch_size * args.smoke_eval_batches]',
        '        split_sha = split_identity(train_ids, selection_ids)',
        '        selection_ids = np.empty(0, dtype=np.int64)\n        selection_points, selection_labels = points[:0], labels[:0]\n        selection_shards, selection_prefix = [], "none_diagnostic_train_only"\n')
    source=replace_region(source,'            selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)\n',
        '            observer.end_epoch(epoch)\n','            selection, clean_train, improved, best = None, None, False, None\n')
    source=replace_region(source,'        final_error, test_metrics = None, None\n',
        '        if rank == 0:\n            manifest.update(status="completed",', '        test_metrics = None\n        dist.barrier()\n')
    ast.parse(source)
    return source


def diagnostic_arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps',choices=tuple(EPSILONS),required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-epochs',type=int,default=10)
    parser.add_argument('--steps-per-epoch',type=int,default=200)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    if args.workers<0 or args.workers>8: parser.error('Expected0..8 workers')
    if args.smoke:
        if not (1<=args.diagnostic_epochs<=2 and 1<=args.steps_per_epoch<=2): parser.error('Smoke<=2epochs x2updates')
    elif (args.diagnostic_epochs,args.steps_per_epoch,args.workers)!=(10,200,8):
        parser.error('Approved diagnostic is10epochs x200steps, workers8')
    args.seed,args.epochs,args.hier,args.arm=4780,300,'on','b05'
    args.resume=None;args.eval_batch_size=16;args.train_eval_every=10
    args.audit_epochs=tuple(range(1,args.diagnostic_epochs+1));args.save_every=1
    args.skip_final_test=True;args.smoke_eval_batches=1
    return args


def diagnostic_configuration(args,original):
    config=dict(original)
    name=f'B05-C0.1-CAP2.3-GSbase1-M500_2-CB64-K20-Hshell4.6-LR0.005-EPS{EPSILONS[args.eps]}-10E-S4780'
    config.update(display_name=name,canonical_name=name,diagnostic_only=True,diagnostic_epochs=args.diagnostic_epochs,
        implementation_variant='b05_eps_10e_20261025',eps_variant=args.eps,arm='b05',
        sampler='uniform32 distinct classes x2 with replacement; SourceClassBatchSampler v5',sampler_epoch_reset=True,
        proxy_optimizer=dict(original['proxy_optimizer'],lr=.005,eps=EPSILONS[args.eps]),
        schedule='300-epoch cosine; stop after10 observed epochs; no scheduler restart',
        selection='none',selection_forward='none',validation='none',evaluation_batch_limit=None,
        test_read=False,test_used_for_selection=False,early_stopping=False,
        parameter_audit='first actual training batch each epoch; same relations and cloned current Gumbel state; restores BN/RNG',
        raw_proxy_gradient='synchronized optimizer-input .grad after lambda0.5 and cap/GS; not an eps-transformed gradient',
        proxy_update_observation='each actual optimizer step; fixed-moment epsilon counterfactual is algebraic only',
        source_checkpoint=None,teacher=None)
    return config


def transformed_audit_source(source):
    source=source.replace('\r\n','\n')
    if sha_text(source)!=EXPECTED_AUDIT: raise RuntimeError('Reviewed B05 audit SHA differs')
    source=replace_once(source,'lambda_hier=.5):','lambda_hier=.5, gumbel_state=None):')
    source=replace_once(source,'generator=torch.Generator(device=cloud.device).manual_seed(seed),','generator=torch.Generator(device=cloud.device).set_state(gumbel_state),')
    source=replace_once(source,'"lambda_hier": lambda_hier, "topk": topk,','"lambda_hier": lambda_hier, "topk": topk, "gumbel_reference": "clone of actual current private training generator",')
    ast.parse(source)
    return source


def verify_optimizer_payload(saved,completed_epochs,variant):
    group,=saved['proxy_optimizer']['param_groups']; schedule=saved['proxy_scheduler']
    lr=.0005+(.005-.0005)*(1+math.cos(math.pi*completed_epochs/300))/2
    if not (saved['format']==VERSION and group['initial_lr']==.005 and group['eps']==EPSILONS[variant]
        and tuple(group['betas'])==(.9,.999) and group['weight_decay']==.01
        and math.isclose(group['lr'],lr,rel_tol=1e-12,abs_tol=1e-15)
        and schedule['T_max']==300 and schedule['eta_min']==.0005 and schedule['last_epoch']==completed_epochs
        and saved['scheduler']['T_max']==300): raise RuntimeError('Actual saved diagnostic optimizer/schedule differs')
    return dict(lr=group['lr'],initial_lr=.005,eps=group['eps'],schedule_epochs=300,completed_epochs=completed_epochs)


def build_namespace(eps='small',source=None):
    if eps not in EPSILONS: raise ValueError('Unknown epsilon variant')
    for path in (HERE,REPO,REPO/'inter_hierarchy_MN40'):
        if str(path) not in sys.path: sys.path.insert(0,str(path))
    source=ORIGINAL.read_text(encoding='utf-8') if source is None else source
    namespace={'__name__':'b05_eps_only_diagnostic','__file__':str(ORIGINAL),'__package__':'inter_hierarchy_MN40.hier_proxy_cycle_v9c'}
    exec(compile(transformed_source(source,eps),str(ORIGINAL)+'[eps-only-10e]','exec'),namespace)
    from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
    from observer import Observer
    audit_ns={'__name__':'b05_eps_same_choice_audit','__file__':str(AUDIT),'__package__':'inter_hierarchy_MN40.hier_proxy_cycle_v9c'}
    exec(compile(transformed_audit_source(AUDIT.read_text(encoding='utf-8')),str(AUDIT)+'[same-choice]','exec'),audit_ns)
    def arguments(argv=None):
        args=diagnostic_arguments(argv)
        if args.eps!=eps: raise RuntimeError('Runner epsilon variant mismatch')
        return args
    namespace.update(diagnostic_arguments=arguments,diagnostic_configuration=diagnostic_configuration,
                     SourceClassBatchSampler=SourceClassBatchSampler,audit_parameter_gradients=audit_ns['audit_parameter_gradients'])
    namespace['observer']=Observer(namespace)
    return namespace


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--eps',choices=tuple(EPSILONS),required=True)
    known,_=parser.parse_known_args()
    build_namespace(known.eps)['main']()
