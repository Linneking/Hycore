"""CPU contracts for faithful CB32 A0 early reruns with300-epoch schedules."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import geoopt

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path[:0]=[str(REPO),str(REPO/'inter_hierarchy_MN40')]
spec=importlib.util.spec_from_file_location('cb32_early_runner_contract',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EarlyRerunContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=runner.ORIGINAL.read_text(encoding='utf-8')
        cls.parent=runner.parent.transformed_source(cls.original,'a0')
        cls.source=runner.transformed_source(cls.original)

    def cli(self,extra=()):
        tree=ast.parse(self.source)
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('arguments','parent_configuration')]
        scope={'argparse':argparse,'Path':Path,'__doc__':'test','ORIGINAL_AUDIT_EPOCHS':runner.ORIGINAL_AUDIT_EPOCHS,
               'TRAIN_SHARD_ORDER':tuple('ply_data_train%d.h5'%i for i in (1,4,3,2,0))}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'<early-cli>','exec'),scope)
        args=scope['arguments'](['--variant','a0','--hier','on','--data-dir','unused','--run-dir','unused',*extra])
        return args,runner.successor_configuration(args,scope['parent_configuration'](args))

    def test_parent_and_audit_pins_reject_changed_source(self):
        self.assertEqual(runner.sha_text(runner.PARENT.read_text()),runner.EXPECTED_PARENT)
        with self.assertRaises(RuntimeError):runner.transformed_source(self.original+'\n# changed')
        with self.assertRaises(RuntimeError):runner.parent.transformed_audit_source(runner.AUDIT_SOURCE.read_text()+'\n# changed')

    def test_production_exact_recipe_observed10_horizon300_every_epoch_saved(self):
        args,cfg=self.cli()
        self.assertEqual((args.epochs,args.observed_epochs,args.steps_per_epoch,args.workers,args.eval_batch_size),(300,10,200,8,16))
        self.assertEqual((args.save_every,args.train_eval_every,args.audit_epochs),(1,10,runner.ORIGINAL_AUDIT_EPOCHS))
        self.assertEqual((cfg['world_size'],cfg['global_batch'],cfg['sample_K'],cfg['proxy_K'],cfg['seed']),(1,32,10,20,4780))
        self.assertEqual((cfg['base_c'],cfg['native_extra_cap'],cfg['native_GS'],cfg['hier_view_depth']),(.1,2.3,True,4.6))
        self.assertEqual((cfg['proxy_optimizer']['lr'],cfg['proxy_optimizer']['eps']),(.01,1e-8))
        self.assertEqual((cfg['observed_epochs'],cfg['schedule_epochs'],cfg['audit_epochs_executed']),(10,300,[1,5]))
        self.assertFalse(args.smoke)
        self.assertIsNone(args.resume)

    def test_smoke_train_only_still_uses_horizon300(self):
        args,cfg=self.cli(['--smoke','--observed-epochs','2','--steps-per-epoch','2','--skip-final-test'])
        self.assertEqual((args.epochs,args.observed_epochs,args.steps_per_epoch,args.save_every),(300,2,2,1))
        self.assertTrue(args.smoke and args.skip_final_test)
        self.assertEqual(cfg['audit_epochs_executed'],[1])
        self.assertIn('if args.smoke:\n            selection_ids = train_ids[:args.eval_batch_size * args.smoke_eval_batches]',self.source)

    def test_rejects_short_cosine_wrong_stop_optimizer_resume_or_cadence(self):
        for options in (['--epochs','10'],['--observed-epochs','5'],['--variant','a1'],['--resume','old.pth'],
                        ['--save-every','20'],['--train-eval-every','1'],['--audit-epochs','1'],
                        ['--smoke','--observed-epochs','2','--steps-per-epoch','2','--epochs','2','--skip-final-test']):
            with self.subTest(options=options),self.assertRaises(SystemExit):self.cli(options)

    def test_actual_training_step_ast_is_identical_to_original_singlecard(self):
        def inner_step(source):
            tree=ast.parse(source)
            main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            return next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                        and ast.unparse(n.target).startswith('(step,'))
        self.assertEqual(ast.dump(inner_step(self.parent)),ast.dump(inner_step(self.source)))
        self.assertIn('for epoch_index in range(start_epoch, args.observed_epochs):',self.source)
        self.assertNotIn('for epoch_index in range(start_epoch, args.epochs):',self.source)

    def test_formal_evaluation_and_historical_audit_path_unchanged(self):
        start='            selection = rank0_evaluation(backbone, selection_points, selection_labels, args, device, selection_prefix, rank)'
        end='            rank_states = [None]'
        self.assertEqual(self.parent[self.parent.index(start):self.parent.index(end)],
                         self.source[self.source.index(start):self.source.index(end)])
        start='        final_error, test_metrics = None, None'
        end='        if rank == 0:\n            manifest.update(status="completed",'
        self.assertEqual(self.parent[self.parent.index(start):self.parent.index(end)],
                         self.source[self.source.index(start):self.source.index(end)])
        self.assertIn('load_shards(args.data_dir, "test")',self.source)
        self.assertIn('if enabled and step == 0 and epoch in args.audit_epochs:',self.source)
        self.assertIn('topk=10, proxy_topk=20',self.source)

    def test_e0_contains_full_recovery_and_completion_reports_actual_stop(self):
        tree=ast.parse(self.source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='atomic_torch'
               and n.args and 'initialization_e0.pth' in ast.unparse(n.args[0])]
        self.assertEqual(len(calls),1)
        keys={k.value for k in calls[0].args[1].keys if isinstance(k,ast.Constant)}
        self.assertTrue({'net','proxy','optimizer','proxy_optimizer','scheduler','proxy_scheduler','rank_states',
                         'completed_epochs','completed_optimizer_steps','train_ids','selection_ids','training_config'}.issubset(keys))
        self.assertIn('completed_epochs=args.observed_epochs, optimizer_steps=args.observed_epochs * args.steps_per_epoch',self.source)
        self.assertIn('"main_training": False, "diagnostic_only": True, "early_rerun": True',self.source)

    def test_real_optimizers_after_ten_epochs_keep_original_cosines(self):
        model,proxy=nn.Parameter(torch.tensor([.1,.2])),nn.Parameter(torch.tensor([2.1,-.1]))
        opt=geoopt.optim.RiemannianSGD([model],lr=.1,momentum=.9,weight_decay=2e-4)
        popt=torch.optim.AdamW([proxy],lr=.01,eps=1e-8,betas=(.9,.999),weight_decay=.01)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300,eta_min=.005)
        psch=torch.optim.lr_scheduler.CosineAnnealingLR(popt,T_max=300,eta_min=.0005)
        for _ in range(10):
            for _ in range(2):
                opt.zero_grad();popt.zero_grad();(model.square().sum()+proxy.square().sum()).backward();opt.step();popt.step()
            sch.step();psch.step()
        payload={'format':runner.VERSION,'completed_epochs':10,'optimizer':opt.state_dict(),
                 'proxy_optimizer':popt.state_dict(),'scheduler':sch.state_dict(),'proxy_scheduler':psch.state_dict()}
        got=runner.verify_optimizer_payload(payload,10)
        self.assertEqual(got['schedule_epochs'],300)
        self.assertGreater(got['lr'],.0099)
        for field,value in [('T_max',10),('last_epoch',300)]:
            altered=copy.deepcopy(payload);altered['proxy_scheduler'][field]=value
            with self.assertRaises(RuntimeError):runner.verify_optimizer_payload(altered,10)
        altered=copy.deepcopy(payload);altered['proxy_optimizer']['param_groups'][0]['eps']=1e-4
        with self.assertRaises(RuntimeError):runner.verify_optimizer_payload(altered,10)


if __name__=='__main__':unittest.main(verbosity=2)
