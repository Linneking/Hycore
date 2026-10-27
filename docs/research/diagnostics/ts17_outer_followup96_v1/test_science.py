import unittest
import numpy as np
from science import *


class Contracts(unittest.TestCase):
    def test_same_pairs_unique_even_when_k_and_order_repeat(self):
        raw=dict(i=np.array([0,1,0,2,2]),j=np.array([1,0,1,3,2]),same=np.ones(5,bool),distinct=np.array([1,1,1,1,0],bool))
        pairs,counts=sample_pairs(raw,np.array([0,0,1,1]),per_class=100)
        np.testing.assert_array_equal(pairs,[[0,1],[2,3]]);self.assertEqual(counts[:2],[1,1])
    def test_radial_swap_preserves_direction_and_requested_h(self):
        x=np.array([[1.,2.,3.],[-1.,4.,2.]])
        y=at_depth(x,[3.4,4.0]);np.testing.assert_allclose(unit(y),unit(x));np.testing.assert_allclose(depth(y),[3.4,4.0])
    def test_angle_swap_is_not_radial_swap(self):
        a=np.array([[1.,0.]]);b=np.array([[0.,1.]])
        self.assertAlmostEqual(angular_rows(a,b)[0],90)
        self.assertAlmostEqual(angular_rows(a,at_depth(a,[4]))[0],0)
    def test_class_mean_does_not_count_large_class_repeatedly(self):
        self.assertEqual(equal_class_mean(np.array([0.,0.,0.,1.]),np.array([0,0,0,1])),.5)
    def test_rank_correlation_ties_and_constant(self):
        self.assertAlmostEqual(rank_correlation([1,1,3],[3,3,1]),-1)
        self.assertIsNone(rank_correlation([1,1,1],[1,2,3]))
    def test_cases_diverse_and_distinct(self):
        ix=choose_cases(np.arange(30.),np.arange(30)%10)
        self.assertEqual(len(ix),6);self.assertEqual(len(set(ix)),6)


if __name__=='__main__':unittest.main(verbosity=2)
