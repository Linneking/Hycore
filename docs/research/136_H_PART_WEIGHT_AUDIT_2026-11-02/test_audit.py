import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('weighted_audit',Path(__file__).with_name('audit.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class AuditTests(unittest.TestCase):
    def test_class_mass_and_missing_support(self):
        y=torch.tensor([[0,0,0,1],[2,2,2,3]])
        w,n=mod.weights_from_fit(y,[0,2],[2,2])
        self.assertGreater(float(w[1]),float(w[0]));self.assertEqual(int(n[4]),0);self.assertEqual(float(w[4]),0)
        for lo in [0,2]:self.assertAlmostEqual(float((w[lo:lo+2]*n[lo:lo+2]).sum()/n[lo:lo+2].sum()),1.,places=6)

    def test_explicit_point_denominator(self):
        z=torch.tensor([[.1,.2],[.3,-.2]],requires_grad=True);y=torch.tensor([0,1]);w=torch.tensor([2.,4.])
        loss=mod.weighted_loss(z,y,w)
        expected=(torch.nn.functional.cross_entropy(5*z,y,reduction='none')*w[y]).sum()/2/5
        torch.testing.assert_close(loss,expected)
        g=torch.autograd.grad(loss,z)[0]
        torch.testing.assert_close(g,((5*z).softmax(-1)-torch.nn.functional.one_hot(y,2))*w[y,None]/2)
        torch.testing.assert_close(mod.weighted_loss(z,y,torch.ones(2)),torch.nn.functional.cross_entropy(5*z,y)/5)

    def test_paired_sampler_does_not_consume_global_rng(self):
        c=torch.arange(16).repeat_interleave(8);counts=list(range(1,17));state=torch.get_rng_state().clone()
        a=mod.draw(torch.Generator().manual_seed(4780),counts,c);b=mod.draw(torch.Generator().manual_seed(4780),counts,c)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]));self.assertEqual(a[key].numel(),8192)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
