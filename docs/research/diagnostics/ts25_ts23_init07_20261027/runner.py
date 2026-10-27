"""TS25: TS23/B13 full recipe; only proxy Gaussian coefficient .9 -> .7."""
from __future__ import annotations
import ast,hashlib,importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
PARENT=HERE.parent/'b05_cb32_a1_followup_20261026/runner.py'
PARENT_SHA='b4c9b79d48890a6d0055f3efdffbe04f998c13bedcfd2f0c432e36e198ef367b'
VERSION='hycore-ts25-ts23-init07-full300-1'
VARIANT='k10p10w01'


def sha_text(x):return hashlib.sha256(x.replace('\r\n','\n').encode()).hexdigest()
if sha_text(PARENT.read_text(encoding='utf-8'))!=PARENT_SHA:raise RuntimeError('Pinned TS23 runner changed')
spec=importlib.util.spec_from_file_location('ts25_preserved_ts23',PARENT);parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ORIGINAL=parent.ORIGINAL;INITIAL_DIGEST=parent.INITIAL_DIGEST


def recipe(variant=VARIANT):
    if variant!=VARIANT:raise ValueError('Only the approved TS23 reference variant')
    return parent.recipe(variant)


def settings(variant=VARIANT):recipe(variant);return parent.settings(variant)


def transformed_source(raw,variant=VARIANT):
    recipe(variant);source=parent.transformed_source(raw,variant)
    source=parent.replace_once(source,f'CHECKPOINT_VERSION = "{parent.VERSION}"',f'CHECKPOINT_VERSION = "{VERSION}"')
    source=parent.replace_once(source,'proxy = HIERLoss(seed=args.seed).to(device)','proxy = INIT07_HIER_CLASS(seed=args.seed).to(device)')
    ast.parse(source);return source


def successor_configuration(args,original):
    config=parent.successor_configuration(args,original)
    name='HC-C01-Cap23-GS1-M500_2-CB32x200 / Hshell-A1-Ks10-Kp10-L0.1-W0-I07 (S4780)'
    config.update(canonical_name=name,display_name=name,implementation_variant='ts25_ts23_init07_20261027',run_alias='TS25',reference_alias='TS23',
        proxy_initialization='private seeded N(0,I)/sqrt(D)*2.3*.7; no per-row normalization',proxy_initialization_coefficient=.7,
        comparison_scope='TS23/B13 full300x200; only proxy Gaussian initialization coefficient .9 to .7',
        initialization_ablation_source=dict(parent_relative_path=PARENT.relative_to(REPO).as_posix(),parent_normalized_sha256=PARENT_SHA,
            wrapper_relative_path=HERE.relative_to(REPO).as_posix(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            transformed_normalized_sha256=sha_text(transformed_source(ORIGINAL.read_text(encoding='utf-8'),args.variant))))
    return config


def verify_optimizer_payload(saved,epochs,schedule_epochs,variant=VARIANT):
    if saved['format']!=VERSION:raise ValueError('Not a TS25 checkpoint')
    return parent.verify_optimizer_payload(dict(saved,format=parent.VERSION),epochs,schedule_epochs,variant)


def init_class():
    import torch,math
    from torch import nn
    from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
    class Init07(HIERLoss):
        def __init__(self,num_proxies=512,dim=256,seed=22):
            super().__init__(num_proxies,dim,seed)
            generator=torch.Generator(device='cpu').manual_seed(seed)
            values=torch.randn(num_proxies,dim,generator=generator)/math.sqrt(dim)*self.clip_r*.7
            self.tangent_proxies=nn.Parameter(values)
    return Init07


def build_namespace(variant=VARIANT,source=None):
    recipe(variant);sys.path[:0]=[str(HERE),str(REPO),str(REPO/'inter_hierarchy_MN40')]
    raw=ORIGINAL.read_text(encoding='utf-8') if source is None else source
    namespace={'__name__':'ts25_ts23_init07','__file__':str(ORIGINAL),'__package__':'inter_hierarchy_MN40.hier_proxy_cycle_v9c','INIT07_HIER_CLASS':init_class()}
    exec(compile(transformed_source(raw,variant),str(ORIGINAL)+'[TS25-init07]','exec'),namespace)
    from inter_hierarchy_MN40.hier_proxy_cycle_v9c import base_protocol
    def losses(outputs,labels):return base_protocol.global_base_losses(outputs,labels,expected_global_batch=32)
    audit={'__name__':'ts25_same_ts23_audit','__file__':str(parent.AUDIT_SOURCE),'__package__':'inter_hierarchy_MN40.hier_proxy_cycle_v9c'}
    exec(compile(parent.transformed_audit_source(parent.AUDIT_SOURCE.read_text(encoding='utf-8'),variant),str(parent.AUDIT_SOURCE)+'[TS25]','exec'),audit)
    audit['global_base_losses']=losses;original_arguments=namespace['arguments']
    def arguments(argv=None):
        args=original_arguments(argv)
        if args.variant!=VARIANT or args.resume is not None:raise RuntimeError('Fresh TS25 only')
        return args
    namespace.update(arguments=arguments,fixed_configuration=lambda args:successor_configuration(args,namespace['parent_configuration'](args)),
        global_base_losses=losses,audit_parameter_gradients=audit['audit_parameter_gradients'],EXPECTED_ORIGINAL_INITIAL_DIGEST=INITIAL_DIGEST)
    return namespace


if __name__=='__main__':build_namespace()['main']()
