import unittest
import importlib.util
import numpy as np
import torch
from . import core as c
from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance


class Contracts(unittest.TestCase):
    def test_three_states_boundaries(self):
        np.testing.assert_array_equal(c.states(np.array([-.1,0,.1,.2,.3],np.float32),.2),[0,0,1,2,2])
        with self.assertRaises(ValueError): c.states([np.nan],.2)
    def test_midpoint_same_direction_not_origin(self):
        a=torch.tensor([[np.tanh(2.),0.]],dtype=torch.float64)
        b=torch.tensor([[np.tanh(2.5),0.]],dtype=torch.float64)
        m,h=c.midpoint(a,b)
        self.assertAlmostEqual(float(h),4.5,places=10)
        self.assertAlmostEqual(float(m[0,0]),np.tanh(2.25),places=10)
    def test_midpoint_angle_moves_inward(self):
        r=np.tanh(2.)
        a=torch.tensor([[r,0.]],dtype=torch.float64)
        phi=np.deg2rad(20.)
        b=torch.tensor([[r*np.cos(phi),r*np.sin(phi)]],dtype=torch.float64)
        _,h=c.midpoint(a,b)
        self.assertAlmostEqual(float(h),np.arctanh(np.tanh(4)*np.cos(phi/2)),places=10)
        _,h0=c.midpoint(a,-a)
        self.assertEqual(float(h0),0.)
    def test_aligned_literal_metric_matches_matrix(self):
        g=torch.Generator().manual_seed(7)
        a=torch.randn(5,4,generator=g,dtype=torch.float64)*.1
        b=torch.randn(5,4,generator=g,dtype=torch.float64)*.1
        torch.testing.assert_close(c.aligned_official_distance(a,b),official_distance(a,b,c=1.).diag())
    def test_undefined_direction_not_zero_angle(self):
        a=torch.tensor([[0.,0.],[.2,0.]])
        b=torch.tensor([[.2,0.],[0.,.2]])
        r=c.angles(a,b)
        self.assertTrue(torch.isnan(r[0]))
        self.assertAlmostEqual(float(r[1]),90.)
    def test_radial_velocity_has_zero_angle_effect(self):
        a=torch.tensor([[.2,.1]],dtype=torch.float64)
        b=torch.tensor([[.1,.2]],dtype=torch.float64)
        self.assertLess(abs(float(c.direction_rate(a,b,2*a,-3*b))),1e-12)
    def test_angle_rate_matches_joint_endpoint_finite_difference(self):
        w=torch.tensor([[.2,.1,.05]],dtype=torch.float64);p=torch.tensor([[.07,.19,.1]],dtype=torch.float64)
        gw=torch.tensor([[.03,-.02,.01]],dtype=torch.float64);gp=torch.tensor([[-.01,.04,.02]],dtype=torch.float64)
        predicted=c.direction_rate(w,p,gw,gp)
        eps=1e-6
        actual=(c.angles(w-eps*gw,p-eps*gp)-c.angles(w,p))/eps
        torch.testing.assert_close(predicted,actual,rtol=1e-5,atol=1e-5)
    def test_finite_pool_feasible_and_impossible(self):
        d=torch.tensor([[1.,2.,3.],[1.,2.,3.],[3.,2.,1.]])
        f=c.feasible_pair(d)
        self.assertTrue(f['exists']);self.assertNotEqual(f['a'],f['b'])
        self.assertTrue(all(x>=.1 for x in f['hier_gaps']))
        self.assertFalse(c.feasible_pair(torch.ones(3,4))['exists'])
    def test_hotspot_is_exposure_aware_and_zero_safe(self):
        h=c.hotspot([10,1],[10,1],[100,10])
        self.assertAlmostEqual(h['per_million_exposure']['mean'],100000.)
        self.assertIsNone(c.hotspot([0],[0],[10])['actual']['top1_fraction'])
    def test_columns_reject_unaligned_rows(self):
        x=c.Columns()
        with self.assertRaises(ValueError):x.add(a=[1],b=[1,2])

    @unittest.skipUnless(importlib.util.find_spec('geoopt'),'Geoopt required for native loss probe')
    def test_native_gradient_probe_is_detached_and_radial_angle_stationary(self):
        w=torch.tensor([[.2,.1,.05],[.1,.3,.08],[-.3,.15,.1],[.1,-.2,.2]],dtype=torch.float64)
        p=torch.tensor([[.08,.12,.04],[.1,.15,.03],[-.12,.08,.05],[.06,-.1,.09]],dtype=torch.float64)
        probe=c.gradient_probe(w,p,200)
        self.assertIsNone(w.grad);self.assertIsNone(p.grad)
        self.assertTrue(np.isfinite(probe['contrastive']['angle_rate']).all())
        self.assertLess(np.max(np.abs(probe['radial']['angle_rate'])),1e-8)
        self.assertLess(np.max(probe['radial']['part_tangent_norm']),1e-10)

    def test_synthetic_panel_preserves_pool_and_probability_denominators(self):
        from .analyze import SelectionAudit,relation_panel
        torch.set_num_threads(2)
        gen=torch.Generator().manual_seed(29)
        def points(n):
            z=torch.randn(n,4,generator=gen);return z/z.norm(dim=-1,keepdim=True)*(.15+.65*torch.rand(n,1,generator=gen))
        mu=points(64);parts=points(192).reshape(64,3,4)
        queue={'features':points(2048),'object_ids':torch.randint(0,64,(2048,),generator=gen),
               'scales':torch.arange(2048)%3,'steps':torch.zeros(2048,dtype=torch.long)}
        frozen=queue['features'].clone()
        audit=SelectionAudit(queue,1);store=c.Columns()
        keys=('hier_same_basic_wrong','hier_cross_basic_wrong','hier_only_short','ord_A_not_shallow','ord_B_not_shallow','ord_only_short')
        relation_panel(store,audit,{k:{} for k in keys},'TS18_e300',0,mu,parts,torch.arange(32).repeat_interleave(2),torch.arange(64),queue,123)
        records=store.finish();summary=audit.finish(records)
        self.assertGreater(len(records['valid']),0)
        self.assertEqual(sum(g['actual_all'] for g in summary['groups'] if g['role']=='A'),len(records['valid']))
        self.assertAlmostEqual(sum(g['expected_all'] for g in summary['groups'] if g['role']=='B'),len(records['valid']),delta=.01)
        torch.testing.assert_close(queue['features'],frozen)


if __name__=='__main__':unittest.main()
