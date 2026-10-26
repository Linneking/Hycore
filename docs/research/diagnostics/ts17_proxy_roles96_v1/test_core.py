"""CPU scientific contracts for role bookkeeping, geometry and controls."""
import unittest
import numpy as np
from core import *


class Contracts(unittest.TestCase):
    def test_overlapping_roles_and_correct_purity_numerators(self):
        counts=np.array([[3,0,2,0,0],[0,2,4,0,0],[0,0,1,5,0]])
        result=role_summary(counts,np.array([True,False,False,True,True]))
        self.assertEqual((result['selected'],result['unused']),(4,1))
        self.assertEqual(result['pure_all_count'],3);self.assertEqual(result['pure_selected_count'],2)
        self.assertEqual(result['pure_selected_fraction'],.5)
        self.assertEqual(sum(x['count'] for x in result['exclusive']),4)
        self.assertGreater(sum(g['of_selected'] for g in result['groups'].values()),1)

    def test_top4_coverage_budget_and_deduplicated_proxy_exposure(self):
        top=np.array([[0,1,2,3],[0,1,4,5]])
        a=coverage(top,[0,1],20)
        self.assertEqual((a['unique_objects'],a['max_exposure']),(6,2))
        self.assertEqual(a['coverage_fraction'],.3);self.assertEqual(a['capacity_normalized'],.75)
        with self.assertRaises(ValueError):coverage(np.array([[0,0,1,2]]),[0],20)

    def test_class_conditioned_null_keeps_four_distinct_objects(self):
        top=np.array([[0,1,2,3],[0,1,2,3]])
        values=null_maxima(top,[0,1],np.zeros(4,int),20,99)
        np.testing.assert_array_equal(values,np.full(20,2))

    def test_no_forced_bimodality(self):
        one=np.random.RandomState(3).normal(3.4,.1,512)
        self.assertFalse(depth_bands(one)['stable_two_bands'])
        two=np.r_[np.linspace(3.05,3.15,256),np.linspace(4.0,4.1,256)]
        b=depth_bands(two);self.assertTrue(b['stable_two_bands']);self.assertTrue(3.2<b['threshold']<3.9)

    def test_average_rank_ties(self):
        np.testing.assert_array_equal(average_ranks([1,1,3]),[.5,.5,2.])

    def test_expected_noncollision_mass_matches_exact_joint_enumeration(self):
        a=np.array([.1,.2,.7]);b=np.array([.4,.5,.1]);joint=a[:,None]*b[None,:];np.fill_diagonal(joint,0)
        np.testing.assert_allclose(joint.sum(1),a*(1-b))
        np.testing.assert_allclose(joint.sum(0),b*(1-a))
        self.assertAlmostEqual(joint.sum(),1-np.dot(a,b))

    def test_weighted_depth_quantiles_and_empty_role(self):
        self.assertEqual(stats([1,2,3],[0,1,4])['median'],3)
        self.assertIsNone(stats([1,2],[0,0])['median'])
        self.assertEqual(roles(np.zeros((3,5),int))[2].sum(),0)

    def test_examples_obey_8_4_4_and_dont_repeat_when_possible(self):
        c=np.arange(1,513)[None,:]*np.ones((3,1),int);h=np.linspace(3,4,512)
        cases=examples(c>0,h,c,np.arange(512)%2==0)
        self.assertEqual([sum(x['role']==r for x in cases) for r in ROLES],[8,4,4])
        self.assertEqual(len({x['proxy_id'] for x in cases}),16)

    def test_class_direction_centers_ignore_radial_scale(self):
        rng=np.random.RandomState(8);native=rng.randn(80,8);labels=np.repeat(np.arange(40),2);proxies=rng.randn(12,8)
        a,theta,_=nearest_centers(native,labels,proxies)
        b,other,_=nearest_centers(native*rng.uniform(.1,3,(80,1)),labels,proxies*3)
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(theta,other,atol=1e-10)

    def test_original_mapping_keeps_proxies_interior_and_shell_fixed(self):
        import torch
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import to_ball
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier
        torch.set_num_threads(2);torch.manual_seed(4)
        native=to_ball(torch.randn(32,8)*.2,hook=False)
        shell=whole_to_hier(native)
        np.testing.assert_allclose(depth(shell.numpy()),4.6,atol=2e-6)
        self.assertLess(float(depth(native.numpy()).max()),4.6)
        np.testing.assert_allclose(unit(native.numpy()),unit(shell.numpy()),atol=1e-7)

    def test_equal_shell_reverse_neighbor_rank_is_angular(self):
        import torch,math
        from inter_hierarchy_MN40.hier_proxy_scratch_v8.geometry import official_distance,to_ball
        from inter_hierarchy_MN40.hier_proxy_cycle_v9c.geometry import whole_to_hier
        angles=torch.tensor([.1,.5,1.1,2.,2.8])
        x=torch.stack([angles.cos(),angles.sin()],dim=1)
        shell=whole_to_hier(x);proxy=to_ball(torch.tensor([[1.4,0.]],dtype=torch.float32),hook=False)
        with torch.no_grad():d=official_distance(proxy,shell).numpy()[0]
        np.testing.assert_array_equal(np.argsort(d),np.arange(5))


if __name__=='__main__':unittest.main(verbosity=2)
