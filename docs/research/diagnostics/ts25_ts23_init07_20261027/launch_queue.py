"""Reuse the published TS23 manager, validation and cleanup; bind TS25 to GPU1."""
from pathlib import Path
import ast,hashlib,sys,shutil,json
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];REL=HERE.relative_to(REPO)
sys.path.insert(0,str(HERE))
import runner
PARENT=HERE.parent/'b05_cb32_a1_followup_20261026/launch_queue.py'
PARENT_SHA='b5bc928252722337fe220fb85067b07fa497888d7aaa70e538de464420cee1ee'


def main():
    raw=PARENT.read_text(encoding='utf-8').replace('\r\n','\n')
    if hashlib.sha256(raw.encode()).hexdigest()!=PARENT_SHA:raise RuntimeError('Pinned TS23 manager changed')
    original='if args.poll_seconds < 10 or (args.variant, args.gpu) not in (("k10p10w01", 0), ("k10p20w005", 1)):'
    if raw.count(original)!=1:raise RuntimeError('GPU mapping boundary changed')
    raw=raw.replace(original,'if args.poll_seconds < 10 or args.variant != "k10p10w01" or args.gpu != 1:')
    raw=raw.replace('Approved mapping: Ks10/Kp10/lambda.1 GPU0, Ks10/Kp20/lambda.05 GPU1; poll >=10 seconds','Approved TS25 init0.7 uses Ks10/Kp10/lambda.1 GPU1; poll >=10 seconds')
    scope={'__name__':'ts25_manager_base','__file__':str(PARENT)};exec(compile(raw,str(PARENT)+'[TS25-GPU1]','exec'),scope)
    scope['runner']=runner;scope['RELATIVE']=REL
    old_freeze=scope['freeze_source'];old_accept=scope['accept_run'];old_write=scope['write']
    def freeze(repo,root,commit,variant):
        snapshot=old_freeze(repo,root,commit,variant)
        folder=Path('docs/research/diagnostics/b05_cb32_a1_followup_20261026');(snapshot/folder).mkdir(parents=True,exist_ok=True)
        identity=json.loads((root/'source_identity.json').read_text())
        for src in (repo/folder).glob('*.py'):
            shutil.copy2(src,snapshot/folder/src.name);identity['files'][(folder/src.name).as_posix()]=hashlib.sha256((snapshot/folder/src.name).read_bytes()).hexdigest()
        identity['successor'].update(run_alias='TS25',reference_alias='TS23',proxy_initialization_coefficient=.7)
        old_write(root/'source_identity.json',identity);return snapshot
    def accept(directory,variant,initial_hash,smoke):
        accepted=old_accept(directory,variant,initial_hash,smoke)
        import torch,math
        saved=torch.load(directory/'initialization_e0.pth',map_location='cpu',weights_only=False)
        generator=torch.Generator(device='cpu').manual_seed(4780)
        expected=torch.randn(512,256,generator=generator)/math.sqrt(256)*2.3*.7
        if saved['training_config']['proxy_initialization_coefficient']!=.7 or not torch.equal(saved['proxy']['tangent_proxies'],expected):raise RuntimeError('Actual initialization is not exact seed4780 coefficient0.7')
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball,depth
        h=depth(to_ball(expected,hook=False))
        accepted.update(run_alias='TS25',reference_alias='TS23',proxy_initialization_coefficient=.7,
            initial_proxy_tensor_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),initial_proxy_depth_median=float(h.median()),initial_proxy_depth_max=float(h.max()))
        return accepted
    def write(path,value):
        if Path(path).name=='queue_status.json':value.update(run_alias='TS25',reference_alias='TS23',proxy_initialization_coefficient=.7)
        old_write(path,value)
    scope.update(freeze_source=freeze,accept_run=accept,write=write)
    scope['main']()


if __name__=='__main__':main()
