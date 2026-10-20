"""Gradient-path and portable-data contracts for the authorized original GS arm."""
from pathlib import Path
import importlib.util
import os
import tempfile
import unittest
from unittest import mock
import json
import torch
import geoopt
import h5py
import numpy as np
from inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020 import train
from inter_hierarchy_MN40.hycore_c01_halfmargin_20261020 import train as baseline

REPO=Path(__file__).resolve().parents[3]

class HookContracts(unittest.TestCase):
    def test_forward_identical_encoder_scaled_classifier_unscaled(self):
        ball=geoopt.PoincareBall(c=.1)
        module=type('Geometry',(),{'manifold':ball})()
        tangent=torch.tensor([[.2,.1],[2.6,.4]],dtype=torch.float64)
        raw=ball.expmap0(tangent).detach().requires_grad_()
        raw0=raw.detach().clone().requires_grad_()
        w=torch.tensor([[.3,-.7],[.1,.4]],dtype=torch.float64,requires_grad=True)
        w0=w.detach().clone().requires_grad_()
        z=train.cap_final_mu(module,None,raw)
        z0=baseline.cap_final_mu(module,None,raw0)
        torch.testing.assert_close(z,z0,atol=0,rtol=0)
        # Include classifier and two intra-like branches at the shared embedding.
        loss=(z@w).square().sum()+.01*z.square().sum()+.01*z.sum()
        loss0=(z0@w0).square().sum()+.01*z0.square().sum()+.01*z0.sum()
        loss.backward();loss0.backward()
        scale=(1-.1*z.detach().square().sum(-1,keepdim=True)).square()/4
        torch.testing.assert_close(raw.grad,raw0.grad*scale,atol=1e-10,rtol=1e-8)
        torch.testing.assert_close(w.grad,w0.grad,atol=0,rtol=0)
        self.assertTrue(bool((scale<.25).all()))

    def test_initialization_unchanged_and_one_final_hook(self):
        original=train.import_original(REPO)
        torch.manual_seed(4780)
        model=train.configure_model(original.Hype_pointMLP(),True)
        self.assertEqual(len(model.emb._forward_hooks),1)
        self.assertEqual(train.state_digest(model),'8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d')
        train.verify_curvature(model)

    def test_cross_host_order_reproduces_requested_rows(self):
        module=type('Data',(),{})()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for name,value in [('ply_data_train0.h5',0),('ply_data_train1.h5',1)]:
                with h5py.File(directory/name,'w') as f:
                    f['data']=np.full((1,2,3),value,dtype='float32')
                    f['label']=np.array([[value]],dtype='int64')
            with mock.patch.dict(os.environ,HYCORE_TRAIN_SHARD_ORDER=json.dumps(['ply_data_train1.h5','ply_data_train0.h5'])):
                identity=train.install_data_root(module,directory)
                _,labels=module.load_data('train')
            self.assertEqual(labels.ravel().tolist(),[1,0])
            self.assertEqual(identity['train']['file_order'],['ply_data_train1.h5','ply_data_train0.h5'])

    def test_margin_function_unchanged_from_nohook_halfmargin(self):
        import inspect
        self.assertEqual(inspect.getsource(train.half_margin_criterion),inspect.getsource(baseline.half_margin_criterion))
        self.assertEqual(set(train.NAMES),{'c01-cap'})
        self.assertIn('-GS1-M500_2-TS-S4780',train.NAMES['c01-cap'])

if __name__=='__main__':unittest.main()