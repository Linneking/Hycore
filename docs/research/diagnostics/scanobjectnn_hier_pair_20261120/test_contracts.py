"""Functional contracts for the approved source-preserving two-arm design."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
import unittest
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from support import SEED, DATA_FILES, Hierarchy, hier_view, source_text, training_function
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import depth
from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss


class NativeBall:
    def logmap0(self, mu):
        radius=mu.norm(dim=-1,keepdim=True).clamp_min(1e-15)
        return torch.atanh(radius)*mu/radius


class Contracts(unittest.TestCase):
    def test_reference_is_literal_original_train(self):
        source=source_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train')
        self.assertEqual(training_function(source,False),ast.get_source_segment(source,node))

    def test_only_hier_loss_call_is_added(self):
        reference=training_function(source_text(),False)
        enabled=training_function(source_text(),True)
        added='\n        loss = branch.add_loss(loss, parent_mu, label, net.manifold2)'
        self.assertEqual(enabled.replace(added,'',1),reference)
        self.assertEqual(enabled.count(added),1)
        self.assertIn('0.01*t_loss + 0.001*h_loss',enabled)
        self.assertIn('kmin=600, kmax=1024',enabled)

    def test_branch_preserves_depth_until_cap_not_a_shell(self):
        tangent=torch.zeros(4,8,dtype=torch.float64)
        tangent[:,0]=torch.tensor([.1,.5,1.,3.],dtype=torch.float64)
        mu=torch.tanh(tangent.norm(dim=-1,keepdim=True))*torch.nn.functional.normalize(tangent,dim=-1)
        before=mu.clone()
        view=hier_view(mu,NativeBall())
        self.assertTrue(torch.equal(before,mu))
        expected=torch.tensor([.2,1.,2.,4.6],dtype=torch.float64)
        self.assertTrue(torch.allclose(depth(view),expected,atol=4e-5,rtol=1e-6))
        self.assertGreater(float(depth(view).std()),1.)
        self.assertTrue(torch.allclose(torch.nn.functional.normalize(mu,dim=-1),torch.nn.functional.normalize(view,dim=-1),atol=1e-12))

    def test_branch_backward_is_finite_and_connected(self):
        mu=torch.tensor([[.1,.2,.05],[.3,-.1,.2]],dtype=torch.float64,requires_grad=True)
        view=hier_view(mu,NativeBall())
        view.sum().backward()
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertGreater(float(mu.grad.abs().sum()),0.)

    def test_proxy_init_does_not_consume_main_rng(self):
        state=torch.get_rng_state().clone()
        proxy=HIERLoss(num_proxies=32,dim=8,seed=SEED)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(tuple(proxy.tangent_proxies.shape),(32,8))

    def test_hier_rngs_are_private(self):
        generator=torch.Generator(device='cpu').manual_seed(39)
        native=torch.zeros(32,8);native[:,0]=.2
        native[:,1]=torch.linspace(-.2,.2,32)
        view=hier_view(native,NativeBall())
        proxy=HIERLoss(num_proxies=32,dim=8,seed=SEED)
        labels=torch.arange(32)%15
        torch_state=torch.get_rng_state().clone()
        numpy_state=np.random.get_state()
        value,stats=proxy(view,labels,topk=20,proxy_topk=20,seed=33,generator=generator,monitor_structure=False)
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        after=np.random.get_state()
        self.assertTrue(np.array_equal(numpy_state[1],after[1]))
        self.assertEqual(numpy_state[2:],after[2:])
        self.assertEqual(stats['sample']['self_k_triplets'],0)
        self.assertEqual(stats['proxy']['self_k_triplets'],0)

    def test_dataset_identity_and_budget(self):
        self.assertEqual(DATA_FILES['training'][2]//32,356)
        self.assertEqual(300*(DATA_FILES['training'][2]//32),106800)
        self.assertEqual(set(DATA_FILES),{'training','test'})
        self.assertTrue(all(len(row[1])==64 for row in DATA_FILES.values()))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
