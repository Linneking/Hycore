"""CPU contracts for epsilon-only training boundaries and detached observations."""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path[:0]=[str(HERE),str(REPO),str(REPO/'inter_hierarchy_MN40')]
spec=importlib.util.spec_from_file_location('eps_runner_contract',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from observer import Observer,capture_before,measure_step,mapped_depth


class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source=runner.ORIGINAL.read_text(encoding='utf-8')
        cls.transformed={key:runner.transformed_source(cls.source,key) for key in ('small','large')}

    def test_parent_and_audit_sha_fail_closed(self):
        with self.assertRaises(RuntimeError): runner.transformed_source(self.source+'\n# mutation')
        with self.assertRaises(RuntimeError): runner.transformed_audit_source(runner.AUDIT.read_text(encoding='utf-8')+'\n# mutation')

    def test_actual_only_optimizer_epsilon_differs_and_lr_is_fixed(self):
        normalized=[]
        for key,source in self.transformed.items():
            tree=ast.parse(source)
            nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.optim.AdamW']
            self.assertEqual(len(nodes),1)
            actual={kw.arg:ast.literal_eval(kw.value) for kw in nodes[0].keywords}
            self.assertEqual(actual,dict(lr=.005,betas=(.9,.999),eps=runner.EPSILONS[key],weight_decay=.01))
            next(kw for kw in nodes[0].keywords if kw.arg=='eps').value=ast.Constant(value=0.)
            normalized.append(ast.dump(tree))
        self.assertEqual(*normalized)

    def test_main_reads_only_train_and_has_no_eval_calls(self):
        for source in self.transformed.values():
            tree=ast.parse(source)
            main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
            names=[ast.unparse(n.func) for n in calls]
            self.assertNotIn('rank0_evaluation',names)
            self.assertNotIn('select_best',names)
            shards=[n for n in calls if ast.unparse(n.func)=='load_shards']
            self.assertEqual([ast.literal_eval(n.args[1]) for n in shards],['train'])
            for expression in ('range(start_epoch, args.diagnostic_epochs)','T_max=args.epochs',
                               'observer.before_step(proxy, proxy_optimizer)',
                               'sync_parameter_gradients(proxy.parameters())','SourceClassBatchSampler(labels[train_ids]'):
                self.assertIn(expression,source)

    def test_cli_exact_full_protocol_and_schedule_horizon(self):
        for variant in ('small','large'):
            args=runner.diagnostic_arguments(['--eps',variant,'--data-dir','unused','--run-dir','unused'])
            self.assertEqual((args.seed,args.epochs,args.diagnostic_epochs,args.steps_per_epoch,args.workers),(4780,300,10,200,8))
            self.assertEqual(args.audit_epochs,tuple(range(1,11)))
            self.assertEqual(args.save_every,1)
            self.assertIsNone(args.resume)
            smoke=runner.diagnostic_arguments(['--eps',variant,'--data-dir','unused','--run-dir','unused','--smoke','--diagnostic-epochs','2','--steps-per-epoch','2'])
            self.assertEqual((smoke.epochs,smoke.diagnostic_epochs,smoke.steps_per_epoch),(300,2,2))

    def test_audit_uses_same_relation_seed_and_cloned_actual_gumbel(self):
        self.assertIn('seed=args.seed + (epoch_index * args.steps_per_epoch + step) * 10007',self.transformed['small'])
        self.assertIn('gumbel_state=generators["gumbel"].get_state().clone()',self.transformed['small'])
        audit=runner.transformed_audit_source(runner.AUDIT.read_text(encoding='utf-8'))
        self.assertIn('generator=torch.Generator(device=cloud.device).set_state(gumbel_state)',audit)
        main=torch.Generator().manual_seed(47);state=main.get_state().clone()
        clone=torch.Generator().set_state(state);observed=torch.rand(12,generator=clone)
        self.assertTrue(torch.equal(main.get_state(),state))
        torch.testing.assert_close(observed,torch.rand(12,generator=main),rtol=0,atol=0)


class ActualOptimizerContracts(unittest.TestCase):
    def test_epsilon_changes_update_but_does_not_modify_input_gradient(self):
        parameters=[];reports=[];wanted=torch.tensor([[2e-6,-7e-5],[1e-5,2e-7]])
        for epsilon in (1e-8,1e-4):
            p=nn.Parameter(torch.tensor([[1.8,.1],[1.7,.2]]));p.grad=wanted.clone()
            optimizer=torch.optim.AdamW([p],lr=.005,eps=epsilon,weight_decay=.01)
            before=capture_before(p,optimizer);optimizer.step();report,_=measure_step(before,p,optimizer)
            torch.testing.assert_close(p.grad,wanted,rtol=0,atol=0)
            self.assertLess(report['prediction_error_max_coordinate'],2e-6)
            parameters.append(p.detach().clone());reports.append(report)
        self.assertFalse(torch.equal(*parameters))
        self.assertEqual(reports[0]['proxy_gradient_norm'],reports[1]['proxy_gradient_norm'])
        self.assertGreater(reports[0]['actual_update_norm'],reports[1]['actual_update_norm'])
        self.assertLess(reports[1]['update_attenuation_median'],.5)
        self.assertLess(reports[0]['fixed_moment_eps1e4_vs_eps1e8_update_cosine'],1.)

    def test_actual_multi_step_accounting_includes_moments_and_weight_decay(self):
        for eps in (1e-8,1e-4):
            p=nn.Parameter(torch.tensor([[2.1,.1],[2.,-.2]]));optimizer=torch.optim.AdamW([p],lr=.005,eps=eps,weight_decay=.01)
            for step in range(4):
                p.grad=torch.tensor([[1e-6*(-1)**step,2e-5],[3e-6,-1e-5]])
                before=capture_before(p,optimizer);optimizer.step();report,_=measure_step(before,p,optimizer)
                self.assertEqual(report['actual_optimizer_step'],step+1)
                self.assertLess(report['prediction_error_max_coordinate'],2e-6)

    def test_epoch_depth_displacements_and_cap_events_are_distinct(self):
        class Proxy(nn.Module):
            def __init__(self):
                super().__init__();self.tangent_proxies=nn.Parameter(torch.tensor([[2.29,0.],[2.31,0.],[1.5,0.]]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def atomic_json(path,payload): path.write_text(json.dumps(payload,allow_nan=False))
            def append_jsonl(path,payload):
                with path.open('a') as stream: stream.write(json.dumps(payload,allow_nan=False)+'\n')
            proxy=Proxy();optimizer=torch.optim.AdamW(proxy.parameters(),lr=.05,eps=1e-8,weight_decay=.01)
            observer=Observer(dict(atomic_json=atomic_json,append_jsonl=append_jsonl))
            observer.bind(proxy,optimizer,root,0,{},{});initial=mapped_depth(proxy.tangent_proxies)
            observer.begin(1,1);proxy.tangent_proxies.grad=torch.tensor([[-1e-4,0.],[1e-4,0.],[1e-4,0.]])
            before=observer.before_step(proxy,optimizer);optimizer.step();observer.after_step(before,proxy,optimizer);observer.end_epoch(1)
            saved=json.loads((root/'observer_epoch_001.json').read_text())
            self.assertEqual((saved['cap_count'],saved['ever_cap_count'],saved['returned_ever_count']),(1,2,1))
            self.assertEqual((saved['first_crossings_epoch'],saved['crossing_events_epoch'],saved['returns_epoch']),(1,1,1))
            torch.testing.assert_close(torch.tensor(saved['depth_delta_init'],dtype=torch.float64),mapped_depth(proxy.tangent_proxies)-initial)
            self.assertEqual(sum(saved[k] for k in ('inward_epoch_count','outward_epoch_count','stationary_epoch_count')),3)
            self.assertEqual(saved['metrics']['proxy_gradient_norm']['count'],1)
            self.assertEqual(saved['per_proxy_step_metrics']['gradient_norm']['count'],3)
            self.assertIsNotNone(json.loads((root/'observer_epoch_000.json').read_text())['initial_proxy_parameter_sha256'])


if __name__=='__main__': unittest.main(verbosity=2)
