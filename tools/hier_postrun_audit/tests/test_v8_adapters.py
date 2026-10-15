"""Compare audit operators against the actual V8 production loss, on CPU."""
import unittest
import numpy as np
import torch
from tools.hier_postrun_audit.extract import resolve_proxy_mapping
from tools.hier_postrun_audit.backbone_probe import _loss_components
from tools.hier_postrun_audit.geometry import expmap0
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
from inter_hierarchy_MN40.hier_proxy_scratch_v8.hier_loss import HIERLoss


class V8AuditContracts(unittest.TestCase):
    def test_v8_map_is_explicit_and_does_not_relax_legacy_guards(self):
        saved=dict(format='hycore-hier-v8-c01-official-geometry-1',proxy={'tangent_proxies':object()},
            training_config=dict(c=.1,extra_HIER_tangent_cap=True,clip_r=2.3,HIER_backward_hook=True))
        mapping=resolve_proxy_mapping(saved)
        self.assertEqual(mapping,dict(numeric_radius_fraction=.999,source_tangent_cap=2.3,source_cap_epsilon=1e-5))
        saved['training_config']['clip_r']=3.
        with self.assertRaisesRegex(ValueError,'V8'):
            resolve_proxy_mapping(saved)
        saved['format']='hycore-hier-v7-w0-projected-proxy-1'
        with self.assertRaisesRegex(ValueError,'c=1'):
            resolve_proxy_mapping(saved)

    def test_saved_tangent_forward_cap_matches_production_numpy_geometry(self):
        torch.manual_seed(8)
        tangent=torch.randn(32,8,dtype=torch.float64)*3
        actual=to_ball(tangent,hook=False).numpy()
        declared=expmap0(tangent.numpy(),float(torch.as_tensor(.1)),numeric_radius_fraction=.999,
                         source_tangent_cap=2.3,source_cap_epsilon=1e-5)
        np.testing.assert_allclose(actual,declared,atol=2e-12,rtol=2e-12)

    def test_cached_feature_loss_and_proxy_gradients_match_actual_v8(self):
        torch.set_num_threads(2)
        torch.manual_seed(22)
        model=HIERLoss(num_proxies=32,dim=8)
        mu=to_ball(torch.randn(64,8),hook=False).detach().requires_grad_()
        nu=to_ball(torch.randn(64,8),hook=False).detach().requires_grad_()
        labels=torch.arange(32).repeat_interleave(2)
        logits=torch.randn(64,40,requires_grad=True)
        config=dict(c=.1,sample_K=6,proxy_K=6,t_per_anchor=3,margin=.1,tau=.1)
        tangent=model.tangent_proxies.detach().clone().requires_grad_()
        values=dict(mu=mu,nu=nu,logits=logits,gold=labels,child_count=256,sample_ids=np.arange(64))
        parts,stats=_loss_components(values,tangent,config,77)
        adapted=parts['sample']+parts['proxy']
        direct,_=model(mu,labels,topk=6,proxy_topk=6,seed=77,t_per_anchor=3,
            data_ids=torch.arange(64),generator=torch.Generator().manual_seed(77),monitor_structure=False)
        torch.testing.assert_close(adapted,direct,atol=1e-7,rtol=1e-6)
        left=torch.autograd.grad(adapted,(tangent,mu),retain_graph=True)
        right=torch.autograd.grad(direct,(model.tangent_proxies,mu))
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=2e-5)
        self.assertEqual(stats['production_operator'],'v8_c01_official_mobius_eps')


if __name__=='__main__': unittest.main()
