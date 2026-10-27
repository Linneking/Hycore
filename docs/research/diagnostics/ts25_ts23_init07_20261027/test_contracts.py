import ast,argparse,importlib.util,sys,unittest,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path[:0]=[str(HERE),str(REPO),str(REPO/'inter_hierarchy_MN40')]
spec=importlib.util.spec_from_file_location('ts25_test_runner',HERE/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class Contracts(unittest.TestCase):
    def test_executable_change_only_initializer_and_version(self):
        source=r.ORIGINAL.read_text();a=r.parent.transformed_source(source,r.VARIANT);b=r.transformed_source(source)
        b=b.replace(r.VERSION,r.parent.VERSION).replace('INIT07_HIER_CLASS(seed=args.seed)','HIERLoss(seed=args.seed)')
        self.assertEqual(ast.dump(ast.parse(a)),ast.dump(ast.parse(b)))
    def cli(self,extra=()):
        tree=ast.parse(r.transformed_source(r.ORIGINAL.read_text()));f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('arguments','parent_configuration')]
        ns=dict(argparse=argparse,Path=Path,__doc__='test',TRAIN_SHARD_ORDER=())
        exec(compile(ast.Module(body=f,type_ignores=[]),'<cli>','exec'),ns)
        args=ns['arguments'](['--variant',r.VARIANT,'--hier','on','--data-dir','unused','--run-dir','unused',*extra])
        return args,r.successor_configuration(args,ns['parent_configuration'](args))
    def test_full_configuration_matches_TS23(self):
        a,c=self.cli();self.assertEqual((a.epochs,a.steps_per_epoch,a.seed,a.workers,a.eval_batch_size),(300,200,4780,8,16))
        self.assertEqual((c['global_batch'],c['sample_K'],c['proxy_K'],c['lambda_hier_after_warmup']),(32,10,10,.1))
        self.assertEqual((c['proxy_optimizer']['lr'],c['proxy_optimizer']['eps'],c['proxy_initialization_coefficient']),(.005,1e-4,.7))
        self.assertEqual((c['base_c'],c['native_extra_cap'],c['native_GS'],c['hier_view_depth']),(.1,2.3,True,4.6))
    def test_smoke_is_separate_and_training_only(self):
        a,c=self.cli(['--smoke','--epochs','2','--steps-per-epoch','2','--skip-final-test']);self.assertTrue(a.smoke and a.skip_final_test);self.assertEqual(a.epochs,2)
    def test_private_rng_and_exact_coefficient(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss
        state=torch.get_rng_state().clone();a=HIERLoss(seed=4780);b=r.init_class()(seed=4780)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        g=torch.Generator().manual_seed(4780);expected=torch.randn(512,256,generator=g)/16*2.3*.7
        self.assertTrue(torch.equal(b.tangent_proxies,expected));torch.testing.assert_close(b.tangent_proxies,a.tangent_proxies*(7/9),atol=5e-8,rtol=2e-7)
        self.assertEqual((a.margin,a.tau,a.c,a.clip_r),(b.margin,b.tau,b.c,b.clip_r))
    def test_saved_A1_schedule_still300(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));m=torch.nn.Parameter(torch.tensor([.1,.2]));import geoopt
        po=torch.optim.AdamW([p],lr=.005,eps=1e-4,weight_decay=.01);mo=geoopt.optim.RiemannianSGD([m],lr=.1,momentum=.9,weight_decay=2e-4)
        ps=torch.optim.lr_scheduler.CosineAnnealingLR(po,300,eta_min=.0005);ms=torch.optim.lr_scheduler.CosineAnnealingLR(mo,300,eta_min=.005)
        saved=dict(format=r.VERSION,completed_epochs=0,proxy_optimizer=po.state_dict(),proxy_scheduler=ps.state_dict(),optimizer=mo.state_dict(),scheduler=ms.state_dict())
        self.assertEqual(r.verify_optimizer_payload(saved,0,300)['initial_lr'],.005)
    def test_changed_parent_source_rejected(self):
        with self.assertRaises(RuntimeError):r.transformed_source(r.ORIGINAL.read_text()+'\n#changed')


if __name__=='__main__':unittest.main(verbosity=2)
