"""Contracts for replay identity, paired movement, and percentage denominators."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('cb32_early_postprocess',Path(__file__).with_name('postprocess.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def observations(epochs=(0,1),population=3):
    return [dict(run_id=module.REPLAY,stage=f'e{e:03d}',epoch=e,observation_id=f'{module.REPLAY}__e{e:03d}',
                 proxies={'available':True,'proxy_count':population,'usage':{'sample_usage':{'available':False}}}) for e in epochs]


class ReplayContracts(unittest.TestCase):
    def test_missing_duplicate_epochs_and_fabricated_initial_usage_fail(self):
        good=observations();self.assertEqual(module.replay_observations({'observations':good},(0,1)),good)
        for bad in [good[:1],good+[good[1]]]:
            with self.assertRaises(ValueError):module.replay_observations({'observations':bad},(0,1))
        good[0]['proxies']['usage']['sample_usage']['available']=True
        with self.assertRaises(ValueError):module.replay_observations({'observations':good},(0,1))
        value={'1':{'net':{'bitwise_equal':False,'max_abs_difference':.01,'mean_abs_difference':.001,'private_path':'not public'},
                    'proxy':{'bitwise_equal':True,'max_abs_difference':0.,'mean_abs_difference':0.}},'private_path':'not public'}
        clean=module.equivalence_scalars(value)
        self.assertNotIn('private_path',str(clean));self.assertEqual(clean['1']['net']['max_abs_difference'],.01)

    def test_raw_motion_hidden_by_cap_is_preserved_and_cross_run_ids_ignored(self):
        obs=observations();rows=[]
        for epoch in [0,1]:
            for p in range(3):rows.append({'observation_id':obs[epoch]['observation_id'],'proxy_id':p,'depth':4.6,'raw_tangent_norm':2.4+epoch})
        rows.append({'observation_id':'B05_CB32_A0_ORIGINAL__e001','proxy_id':0,'depth':0.,'raw_tangent_norm':999})
        scalars,summary=module.proxy_trajectories(obs,rows)
        self.assertEqual(len(scalars),6)
        self.assertEqual(summary[1]['depth_delta_init_stationary_percent'],100)
        self.assertEqual(summary[1]['raw_norm_delta_init_outward_percent'],100)
        self.assertAlmostEqual(summary[1]['raw_norm_delta_init_median'],1.)
        with self.assertRaises(ValueError):module.proxy_trajectories(obs,rows[:-2])

    def test_paired_delta_median_is_not_difference_of_population_medians(self):
        obs=observations();rows=[]
        for epoch,values in enumerate([[1.,2.,100.],[2.,100.,101.]]):
            rows.extend({'observation_id':obs[epoch]['observation_id'],'proxy_id':p,'depth':v,'raw_tangent_norm':v/2} for p,v in enumerate(values))
        _,summary=module.proxy_trajectories(obs,rows)
        self.assertEqual(summary[1]['depth_delta_init_median'],1.)
        self.assertNotEqual(summary[1]['depth_delta_init_median'],100.-2.)
        m=module.movement_stats([-1.,0.,1.])
        self.assertAlmostEqual(m['inward_percent']+m['outward_percent']+m['stationary_percent'],100.)

    def test_histograms_include_negative_motion_and_each_population_sums_to100(self):
        scalars=[]
        for epoch,values in [(0,[0.,0.,0.]),(1,[-.501,0.,.706])]:
            for value in values:scalars.append({'epoch':epoch,**{key:value for key in ['depth_delta_epoch','depth_delta_init','raw_norm_delta_epoch','raw_norm_delta_init']}})
        hist=module.movement_histograms(scalars)
        for epoch in [0,1]:
            for domain in ['depth_delta_epoch','depth_delta_init','raw_norm_delta_epoch','raw_norm_delta_init']:
                selected=[r for r in hist if r['epoch']==epoch and r['domain']==domain]
                self.assertEqual(sum(r['count'] for r in selected),3)
                self.assertAlmostEqual(sum(r['percent'] for r in selected),100.)
                self.assertTrue(all(r['population']==3 for r in selected))


if __name__=='__main__':unittest.main()
