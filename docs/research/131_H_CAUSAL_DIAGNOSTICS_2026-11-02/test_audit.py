import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit.py'))
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class DiagnosticTests(unittest.TestCase):
    def test_bn_uses_batch_without_mutating_buffers_or_dropout(self):
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Dropout(.9)).eval()
        x=torch.arange(24,dtype=torch.float32).reshape(8,3)
        state={k:v.clone() for k,v in model.state_dict().items()}
        original=model(x)
        with audit.batch_bn(model):
            self.assertFalse(model[1].training)
            batch=model(x)
            self.assertLess(float(batch.mean(0).abs().max()),1e-5)
        self.assertFalse(torch.allclose(original,batch))
        self.assertFalse(model[0].training)
        self.assertTrue(model[0].track_running_stats)
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,state[k]))

    def test_part_contributions_sum_to_batch_mean_gradient(self):
        torch.manual_seed(1)
        w=torch.randn(3,2,requires_grad=True); x=torch.randn(7,2)
        y=torch.tensor([0,0,1,1,1,2,2])
        losses=torch.nn.functional.cross_entropy(x@w.T,y,reduction='none')
        total=torch.autograd.grad(losses.mean(),w,retain_graph=True)[0]
        gs=[torch.autograd.grad((losses*(y==k)).sum()/y.numel(),w,retain_graph=True)[0] for k in range(3)]
        torch.testing.assert_close(sum(gs),total)

    def test_score_uses_global_argmax_but_legal_margin(self):
        z=torch.zeros(1,2,50); z[0,:,5]=2; z[0,:,40]=3
        y=torch.tensor([[4,5]]); c=torch.tensor([1])
        s=audit.scores(z,y,c)
        self.assertEqual(s['4']['recall'],0)
        self.assertEqual(s['5']['recall'],0)
        self.assertEqual(s['4']['margin'],-2)
        self.assertEqual(s['5']['margin'],2)


if __name__=='__main__': unittest.main()
