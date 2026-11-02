import sys
from pathlib import Path
import unittest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

class ProductionTests(unittest.TestCase):
    def test_metrics_preserve_original_and_count_missing(self):
        from segmentation_ShapeNetPart.hycore_seg_v9.metrics import SegMetrics
        from segmentation_ShapeNetPart.hycore_seg_v1.metrics import SegMetrics as Old
        p=np.array([[5,5,5,5],[6,6,7,7]]);y=np.array([[4,5,5,5],[6,6,7,7]]);c=np.array([1,2])
        new,old=SegMetrics(),Old();new.update(p,y,c);old.update(p,y,c)
        result=new.result()
        for k,v in old.result().items():self.assertEqual(result[k],v)
        self.assertEqual(result['parts']['4']['recall'],0.)
        self.assertEqual(result['parts']['4']['missing_objects'],1)
        self.assertIsNone(result['parts']['4']['precision'])

    def test_native_initialization_and_scaled_gradient(self):
        from segmentation_ShapeNetPart.hycore_seg_v6.model import NativeTail2 as Old
        from segmentation_ShapeNetPart.hycore_seg_v9.model import NativeTail2
        from segmentation_ShapeNetPart.hycore_seg_v1.utils import state_hash
        from segmentation_ShapeNetPart.hycore_seg_v9.train import plain_ce
        torch.manual_seed(4780);a=Old('h',k=4)
        torch.manual_seed(4780);b=NativeTail2('h',k=4)
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
        z=torch.randn(2,3,50,requires_grad=True);y=torch.tensor([[4,5,5],[41,42,43]])
        raw=plain_ce(z,y);torch.testing.assert_close(raw,plain_ce(z*1,y)/1)
        g=torch.autograd.grad(plain_ce(5*z,y)/5,z)[0]
        torch.testing.assert_close(g,((5*z).softmax(-1)-torch.nn.functional.one_hot(y,50))/6)

    def test_regularizer_zero_and_active_paths(self):
        from segmentation_ShapeNetPart.hycore_seg_v9.losses import HyCoReSegLoss
        x=torch.randn(4,12,256,requires_grad=True)*.02;c=torch.tensor([0,1,0,2]);state=torch.get_rng_state().clone()
        zero=HyCoReSegLoss(0,0,part_min=3,part_max=5);off,_=zero(x,c)
        self.assertFalse(off.requires_grad)
        on=HyCoReSegLoss(.01,.01,part_min=3,part_max=5);loss,_=on(x,c)
        self.assertTrue(loss.requires_grad);self.assertTrue(torch.isfinite(torch.autograd.grad(loss,x)[0]).all())
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
