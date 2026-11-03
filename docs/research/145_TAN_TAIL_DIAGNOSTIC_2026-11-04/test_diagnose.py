"""Synthetic CPU-only checks for read-only tail diagnostics."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

p=Path(__file__).with_name('diagnose.py')
spec=importlib.util.spec_from_file_location('diagnose145',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Contracts(unittest.TestCase):
    def test_absent_and_size_denominators(self):
        stats=m.FocusStatistics()
        y=np.array([47,47,48,48]);pred=y.copy()
        stats.update(pred,y,15,0,'synthetic0')
        y2=np.array([47,47,49,49]);pred2=np.array([47,47,47,47])
        stats.update(pred2,y2,15,1,'synthetic1')
        r=stats.summary()['table']['parts']['49']
        self.assertEqual(r['gt_present_objects'],1)
        self.assertEqual(r['absent_both_objects'],1)
        self.assertEqual(r['zero_recall_objects_given_present'],1)
        self.assertEqual(r['no_prediction_objects_given_present'],1)
        self.assertEqual(r['mean_iou_gt_present'],0.)
        self.assertEqual(r['mean_iou_all_objects_absent_both_is_one'],.5)
        self.assertEqual(r['gt_point_count_bins']['1_16']['gt_present_objects'],1)
        self.assertIsNone(r['gt_point_count_bins']['17_64']['mean_iou_gt_present'])
        self.assertEqual(r['precision'],None)

    def test_confusion_and_pairing(self):
        a=m.FocusStatistics();b=m.FocusStatistics()
        y=np.array([24,25,26,27]);pa=np.array([24,25,25,25]);pb=y.copy()
        ra=a.update(pa,y,8,0,'synthetic')
        rb=b.update(pb,y,8,0,'synthetic')
        matrix=a.summary()['lamp']['confusion_gt_rows_pred_columns_50x50']
        self.assertEqual(matrix[26][25],1)
        self.assertEqual(matrix[27][25],1)
        pair=m.paired_summary([ra],[rb])['lamp']
        self.assertGreater(pair['mean_iou_difference'],0)
        self.assertEqual(pair['improved_objects'],1)
        self.assertEqual(pair['parts']['26']['zero_recall_recovered_objects'],1)
        rb['id']='different'
        with self.assertRaises(ValueError):m.paired_summary([ra],[rb])

    def test_boundary_and_regions(self):
        points=np.array([[0,0,0],[.01,0,0],[5,0,0],[5.01,0,0]])
        y=np.array([47,47,48,48])
        self.assertFalse(m.boundary_mask(points,y,k=1).any())
        self.assertTrue(m.boundary_mask(points,y,k=3).all())
        self.assertFalse(m.boundary_mask(points[:1],y[:1]).any())
        s=m.FocusStatistics();s.update(y,y,15,0,'synthetic',points=points,tangent_norm=np.ones(4))
        r=s.summary()['table']
        self.assertEqual(r['regions']['boundary']['error_rate'],0.)
        self.assertEqual(r['tangent_norm_by_correctness']['correct']['mean'],1.)
        self.assertIsNone(r['tangent_norm_by_correctness']['incorrect']['mean'])

    def test_cosine_positive_row_factor_and_frozen_tensors(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(123)
        x=torch.randn(2,8,128);w=torch.randn(50,128)
        scale=torch.linspace(.008,1.,50)[:,None]
        before=(x.clone(),w.clone(),torch.get_rng_state().clone())
        q=F.linear(F.normalize(x,dim=-1),F.normalize(w,dim=-1))
        qeff=F.linear(F.normalize(x,dim=-1),F.normalize(w*scale,dim=-1))
        torch.testing.assert_close(q,qeff)
        self.assertTrue(torch.equal(q.argmax(-1),qeff.argmax(-1)))
        self.assertTrue(torch.equal(x,before[0]));self.assertTrue(torch.equal(w,before[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
