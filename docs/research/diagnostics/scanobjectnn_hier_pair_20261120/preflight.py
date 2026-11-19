"""Fresh train-only smokes and source/off/input/RNG acceptance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from support import now,write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    with (args.root/'cpu_contracts.log').open('x') as log:
        subprocess.run([sys.executable,str(HERE/'test_contracts.py')],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=240)
    checks={}
    for name,mode in [('reference','reference'),('disabled','disabled'),('enabled','enabled')]:
        command=[sys.executable,'-u',str(HERE/'runner.py'),'--arm','SO-HIER' if mode=='enabled' else 'SO-H0',
            '--mode',mode,'--smoke','--data-dir',str(args.data_dir),'--output',str(args.root/name),
            '--source-commit',args.source_commit]
        with (args.root/(name+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        manifest=json.loads((args.root/name/'manifest.json').read_text())
        state=torch.load(args.root/name/'last.pth',map_location='cpu',weights_only=False)
        traces=[json.loads(line) for line in (args.root/name/'input_trace.jsonl').read_text().splitlines()]
        checks[name]=dict(manifest=manifest,state=state,traces=traces)
    reference=checks['reference'];disabled=checks['disabled'];enabled=checks['enabled']
    accepted={}
    for name,value in checks.items():
        mf,state=value['manifest'],value['state']
        accepted[name+'_complete4']=mf['status']=='completed' and mf['optimizer_steps']==4 and mf['completed_epochs']==2
        accepted[name+'_BN8']=state['metrics']['BN_counts']==[8]
        accepted[name+'_finite']=all(bool(torch.isfinite(x).all()) for x in state['net'].values())
        accepted[name+'_no_test']=state['metrics']['test'] is None
        accepted[name+'_initial_model']=mf['initialized_model_sha256']==reference['manifest']['initialized_model_sha256']
        accepted[name+'_inputs']=len(value['traces'])==4 and all(
            all(a[k]==b[k] for k in ('ids','cloud','labels','part','whole','part_points','whole_points'))
            for a,b in zip(reference['traces'],value['traces']))
        rng=state['rng'];rr=reference['state']['rng']
        accepted[name+'_main_rng']=rng['python']==rr['python'] and np.array_equal(rng['numpy'][1],rr['numpy'][1]) and rng['numpy'][2:]==rr['numpy'][2:] and torch.equal(rng['torch_cpu'],rr['torch_cpu']) and all(torch.equal(a,b) for a,b in zip(rng['torch_cuda'],rr['torch_cuda']))
    accepted['off_no_proxy']=reference['state']['proxy'] is None and disabled['state']['proxy'] is None
    accepted['enabled_proxy4']=enabled['state']['proxy'] is not None and all(int(v['step'])==4 for v in enabled['state']['proxy_optimizer']['state'].values())
    accepted['base_first_loss']=all(abs(a-b)<=1e-4+1e-5*abs(a) for a,b in zip(reference['traces'][0]['base_losses'],enabled['traces'][0]['base_losses']))
    view=torch.load(args.root/'enabled/first_hier_view.pth',map_location='cpu',weights_only=False)
    accepted['live_non_shell_branch']=view['graph_connected'] is True and view['fixed_shell'] is False
    differences={}
    for name in ['disabled','enabled']:
        differences[name+'_first_base_losses']=checks[name]['traces'][0]['base_losses']
    differences['reference_first_base_losses']=reference['traces'][0]['base_losses']
    differences['disabled_reference_parameter_max_abs']=max(float((disabled['state']['net'][k].to(torch.float64)-v.to(torch.float64)).abs().max()) for k,v in reference['state']['net'].items())
    result=dict(passed=all(accepted.values()),cpu_contracts_passed=True,source_commit=args.source_commit,observed_at=now(),
        checks=accepted,reference_repeatability=differences,bitwise_replay_claimed=False,
        dataset=reference['manifest']['dataset_identity'],production_initialization='fresh; no smoke weights loaded')
    write(args.root/'acceptance.json',result)
    if not result['passed']:
        raise RuntimeError('Preflight failed: '+str([k for k,v in accepted.items() if not v]))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
