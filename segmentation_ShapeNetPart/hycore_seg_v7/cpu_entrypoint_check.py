"""Full synthetic CPU entrypoint checks for all zero-regularizer arms."""
import ast
import contextlib
import json
from pathlib import Path
import torch
from . import train
from ..hycore_seg_v6.model import NativeTail2
from ..hycore_seg_v6.cpu_entrypoint_check import write_fixture
from ..hycore_seg_v1.utils import state_hash, write_json
from . import launch_experiment as launch

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def main():
    root=Path(__file__).resolve().parents[2]
    for suffix in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        out=root/'docs/research/tmp'/('pseg7_zero_cpu_20261030_'+suffix)
        if not out.exists():out.mkdir(parents=True);break
    write_fixture(out/'data');states={};results={}
    train.git_identity=lambda:{'commit':'SYNTHETIC_CPU_ONLY','branch':'synthetic','source_status':''}
    for arm in 'fgh':
        folder=out/arm;folder.mkdir()
        argv=['--data',str(out/'data'),'--out',str(folder),'--arm',arm,'--protocol','validation',
              '--epochs','2','--schedule-epochs','200','--batch-size','4','--eval-batch-size','4',
              '--num-points','32','--k','4','--workers','0','--cpu-threads','2','--device','cpu',
              '--part-min','4','--part-max','8','--max-steps','2','--monitor-every','1','--gradient-every','1',
              '--no-final-test','--allow-dirty-cpu']
        args=train.arguments(argv)
        assert args.alpha==args.beta==0.
        try:train.arguments(argv+['--alpha','.01'])
        except ValueError:pass
        else:raise AssertionError('Nonzero training coefficient accepted')
        with (out/(arm+'.log')).open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):train.train(args)
        initial=torch.load(folder/'e0.pth',map_location='cpu',weights_only=False)
        final=torch.load(folder/'last.pth',map_location='cpu',weights_only=False)
        manifest=json.loads((folder/'manifest.json').read_text());metrics=rows(folder/'metrics.jsonl')
        observations=rows(folder/'geometry.jsonl');steps=rows(folder/'steps.jsonl')
        assert final['epoch']==2 and final['updates']==4
        assert final['scheduler']['T_max']==200 and final['scheduler']['last_epoch']==2
        assert train.finite_tensor_state(final['model']) and train.finite_tensor_state(final['optimizer'])
        assert all(int(t)==4 for n,t in final['model'].items() if n.endswith('num_batches_tracked'))
        assert manifest['regularizers_active'] is False and manifest['regularizer_diagnostics_only'] is True
        assert manifest['dropout_active'] is (arm=='g') and manifest['trainable_parameters']==1460283
        assert all(s['reg']==0. and s['loss']==s['ce'] and s['clip_applied'] is False for s in steps)
        assert all(m['train_reg']==0. and m['train_loss']==m['train_ce'] for m in metrics)
        assert len(observations)==4 and all(len(o['geometry'])==11 for o in observations)
        assert launch._regularizers_disabled(observations)
        assert launch._ce_reaches_euclidean9_and_head(observations)
        assert all(o['diagnostics']['negative_legal_rate']==1. for o in observations)
        torch.manual_seed(4780);reference=NativeTail2(arm,k=4)
        assert state_hash(reference.state_dict())==state_hash(initial['model'])
        # Model and all optimizer tensors survive ordinary save/load exactly.
        reference.load_state_dict(final['model'],strict=True)
        assert state_hash(reference.state_dict())==state_hash(final['model'])
        optimizer=train.geoopt.optim.RiemannianSGD(reference.parameters(),lr=.1,momentum=.9,weight_decay=1e-4)
        optimizer.load_state_dict(final['optimizer']);assert train.finite_tensor_state(optimizer.state_dict())
        for key in ['train_generator','eval_generator','part_generator','negative_generator']:
            g=torch.Generator();g.set_state(final[key]);assert torch.equal(g.get_state(),final[key])
        assert json.loads((folder/'acceptance.json').read_text())['final_test_evaluations']==0
        states[arm]=(initial,final,metrics)
        results[arm]={'complete':True,'updates':4,'initial_model_sha256':state_hash(initial['model']),
                      'zero_R_gradients':True,'CE_only_exact':True,'BN_once':True,'finite':True}
    assert len({results[a]['initial_model_sha256'] for a in 'fgh'})==1
    for name in states['f'][1]['model']:
        torch.testing.assert_close(states['f'][1]['model'][name],states['h'][1]['model'][name],rtol=1e-4,atol=1e-4)
    assert torch.equal(states['f'][1]['rng']['torch'],states['h'][1]['rng']['torch'])
    for key in ['train_generator','eval_generator','part_generator','negative_generator']:
        assert all(torch.equal(states[a][1][key],states['f'][1][key]) for a in 'fgh')
    assert all([m['first_input_sha256'] for m in states[a][2]]==[m['first_input_sha256'] for m in states['f'][2]] for a in 'fgh')
    assert not torch.cuda.is_initialized()
    # Actual total-loss statement and future GPU authorization are explicit.
    tree=ast.parse((Path(__file__).parent/'train.py').read_text())
    assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)
               and isinstance(n.value,ast.Name) and n.value.id=='ce' for n in ast.walk(tree))
    assert launch.approved_coefficients('f')==(0.,0.) and launch.authorized_gpu('h',2)
    result={'complete':True,'CPU_only':True,'cuda_initialized':False,'arms':results,'F0_H0_final_state_FP32_close':True,
            'all_independent_RNG_paired':True,'full_entrypoint_updates_each':4}
    write_json(out/'checks.json',result);print(json.dumps({'output':str(out),**result}),flush=True)
if __name__=='__main__':main()
