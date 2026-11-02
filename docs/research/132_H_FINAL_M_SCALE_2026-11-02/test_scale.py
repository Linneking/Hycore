import copy
import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('scale',Path(__file__).with_name('run_scale.py'))
scale=importlib.util.module_from_spec(spec);spec.loader.exec_module(scale)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Linear(3,4)
        self.mlr=torch.nn.Module()
        self.mlr._weight=torch.nn.Parameter(torch.randn(2,4))
        self.mlr._bias=torch.nn.Parameter(torch.randn(2,1))
        self.mlr.curvature=torch.nn.Parameter(torch.ones(()),requires_grad=False)

    def forward(self,x):
        return self.encoder(x)@self.mlr._weight.T+self.mlr._bias.T


class ScaleTests(unittest.TestCase):
    def test_scale_gradient_has_no_extra_factor_five(self):
        z=torch.tensor([[.3,-.2,.1],[.2,.5,-.1]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,2])
        for s in [1,5]:
            grad=torch.autograd.grad(scale.scale_loss(z,y,s),z)[0]
            expected=((z.detach()*s).softmax(-1)-torch.nn.functional.one_hot(y,3))/2
            torch.testing.assert_close(grad,expected,atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.equal(z.argmax(-1),(5*z).argmax(-1)))

    def test_restore_preserves_frozen_momentum_and_independent_arms(self):
        torch.manual_seed(4780)
        base=Tiny();opt=torch.optim.SGD(base.parameters(),lr=.003,momentum=.9,weight_decay=.01)
        base(torch.randn(8,3)).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        checkpoint={'config':{'lr':.003,'momentum':.9,'weight_decay':.01},'optimizer':copy.deepcopy(opt.state_dict())}
        a,b=copy.deepcopy(base),copy.deepcopy(base)
        oa,keys=scale.final_only_optimizer(a,checkpoint,torch.optim.SGD)
        ob,_=scale.final_only_optimizer(b,checkpoint,torch.optim.SGD)
        frozen=copy.deepcopy(a.encoder.state_dict());initial_b=copy.deepcopy(ob.state_dict())
        a(torch.randn(8,3)).square().mean().backward();oa.step()
        self.assertTrue(scale.tree_equal(a.encoder.state_dict(),frozen))
        self.assertTrue(scale.frozen_optimizer_equal(oa.state_dict(),checkpoint['optimizer'],keys))
        self.assertTrue(scale.tree_equal(ob.state_dict(),initial_b))
        self.assertTrue(scale.tree_equal(checkpoint['optimizer'],initial_b))
        self.assertFalse(scale.tree_equal(oa.state_dict(),ob.state_dict()))

    def test_only_expected_group_counter_may_change(self):
        initial={'param_groups':[{'params':[0,1],'lr':.003,'step':100}], 'state':{0:{'momentum_buffer':torch.ones(2)}}}
        current=copy.deepcopy(initial);current['param_groups'][0]['step']=110
        self.assertTrue(scale.frozen_optimizer_equal(current,initial,{1},10))
        self.assertFalse(scale.frozen_optimizer_equal(current,initial,{1},9))
        current['param_groups'][0]['lr']=.004
        self.assertFalse(scale.frozen_optimizer_equal(current,initial,{1},10))

    def test_real_rsgd_frozen_state_and_counter(self):
        try:import geoopt
        except ImportError:self.skipTest('geoopt not installed in local CPU environment')
        base=Tiny();opt=geoopt.optim.RiemannianSGD(base.parameters(),lr=.003,momentum=.9,weight_decay=.01)
        base(torch.randn(8,3)).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        checkpoint={'config':{'lr':.003,'momentum':.9,'weight_decay':.01},'optimizer':copy.deepcopy(opt.state_dict())}
        model=copy.deepcopy(base);frozen=copy.deepcopy(model.encoder.state_dict())
        restored,keys=scale.final_only_optimizer(model,checkpoint,geoopt.optim.RiemannianSGD)
        model(torch.randn(8,3)).square().mean().backward();restored.step()
        self.assertTrue(scale.tree_equal(model.encoder.state_dict(),frozen))
        self.assertTrue(scale.frozen_optimizer_equal(restored.state_dict(),checkpoint['optimizer'],keys,1))


if __name__=='__main__':unittest.main()
