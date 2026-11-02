"""Synthetic selection and numerical tests; no trained model or real subject."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage_evaluation import select_stage_windows, score_stage_draw, summarize_stage_draws


def prepared(stages, *, tail=0):
    labels=np.asarray(stages,dtype=np.int64)
    n=len(labels)*750+tail
    reference=np.tile(np.linspace(.8,1.2,750),len(labels))
    reference=np.concatenate((reference,np.ones(tail)))
    conditions=np.zeros((n,9),dtype=np.float32)
    conditions[:len(labels)*750,:4]=np.repeat(np.eye(4)[labels],750,axis=0)
    conditions[:,7]=1
    return dict(subject_id='SYNTHETIC',conditions=conditions,reference_seconds=reference,
        stage_epochs_four=labels.copy(),record=dict(stage_epochs=labels.copy(),
            valid_epochs=np.ones(len(labels),dtype=bool),adjacent=np.ones(max(0,len(labels)-1),dtype=bool),
            discarded_tail_samples=0))


class StageSelectionTests(unittest.TestCase):
    def test_fixed_grid_four_stages_seeded_unique_and_input_unchanged(self):
        source=prepared(([0]*4+[1]*4+[2]*4+[3]*4)*6)
        original=copy.deepcopy(source)
        a=select_stage_windows(source,window_epochs=4,windows_per_stage=2,seed=42)
        self.assertEqual(a,select_stage_windows(source,window_epochs=4,windows_per_stage=2,seed=42))
        b=select_stage_windows(source,window_epochs=4,windows_per_stage=2,seed=43)
        self.assertNotEqual(a['start_samples'],b['start_samples'])
        self.assertEqual(len(a['start_samples']),8)
        self.assertEqual(a['start_samples'],sorted(set(a['start_samples'])))
        self.assertTrue(all(start%3000==0 for start in a['start_samples']))
        for item in a['stages'].values():
            self.assertEqual(item['selected_windows'],2)
            self.assertEqual(item['selected_epochs'],8)
            self.assertTrue(item['target_met'])
        for key in ('conditions','reference_seconds','stage_epochs_four'):
            np.testing.assert_array_equal(source[key],original[key])
        for key in ('stage_epochs','valid_epochs','adjacent'):
            np.testing.assert_array_equal(source['record'][key],original['record'][key])

    def test_mixed_window_satisfies_multiple_stage_quotas_without_duplicate_epochs(self):
        source=prepared([0,1,2,3]*7)
        result=select_stage_windows(source,window_epochs=4,windows_per_stage=3)
        self.assertEqual(result['selected_windows'],3)
        self.assertEqual(sum(item['selected_epochs'] for item in result['stages'].values()),12)
        for item in result['stages'].values():
            self.assertEqual(item['selected_windows'],3)
            self.assertEqual(item['selected_epochs'],3)

    def test_missing_or_insufficient_stage_is_explicit_not_zero_quality(self):
        source=prepared([0,0,1,1])
        result=select_stage_windows(source,window_epochs=2,windows_per_stage=8)
        self.assertEqual(result['stages']['W']['status'],'insufficient_windows')
        self.assertEqual(result['stages']['W']['selected_windows'],1)
        self.assertEqual(result['stages']['REM']['status'],'missing')
        self.assertFalse(result['stages']['REM']['target_met'])

    def test_invalid_epoch_gap_and_one_observation_exclude_whole_windows_without_repacking(self):
        source=prepared([0,1]*6,tail=17)
        source['record']['valid_epochs'][0]=False
        source['record']['adjacent'][2]=False
        source['reference_seconds'][4*750:5*750]=0
        source['reference_seconds'][4*750]=1
        source['reference_seconds'][6*750]=-1
        source['conditions'][8*750,0:4]=[0,0,0,1]
        result=select_stage_windows(source,window_epochs=2,windows_per_stage=8)
        self.assertEqual(result['start_samples'],[10*750])
        self.assertEqual(len(result['excluded_windows']),5)
        reasons={reason for row in result['excluded_windows'] for reason in row['reasons']}
        self.assertIn('source_time_gap_inside_window',reasons)
        self.assertIn('fewer_than_two_observed_samples_in_epoch',reasons)
        self.assertIn('stage_conditions_disagree_with_labels',reasons)
        self.assertEqual(result['incomplete_window_tail_samples'],17)

    def test_grid_boundary_gap_is_not_crossed_and_incomplete_tail_is_reported(self):
        source=prepared([0,1,2,3,0],tail=23)
        source['record']['adjacent'][1]=False
        source['record']['discarded_tail_samples']=7
        result=select_stage_windows(source,window_epochs=2,windows_per_stage=1)
        self.assertEqual(result['start_samples'],[0,1500])
        self.assertEqual(result['incomplete_window_tail_samples'],773)
        self.assertEqual(result['incomplete_window_tail_complete_epochs'],1)
        self.assertEqual(result['source_discarded_tail_samples'],7)

    def test_ranking_invariant_to_valid_positive_ibi_magnitudes(self):
        source=prepared([0,1,2,3]*12)
        changed=copy.deepcopy(source)
        changed['reference_seconds']*=20
        self.assertEqual(select_stage_windows(source,window_epochs=4),
                         select_stage_windows(changed,window_epochs=4))

    def test_variable_model_lengths_and_invalid_arguments(self):
        source=prepared([0,1,2,3]*16)
        for k in (1,2,4,8):
            result=select_stage_windows(source,window_epochs=k,windows_per_stage=2)
            self.assertEqual(result['window_size'],k*750)
            self.assertTrue(all(start%(k*750)==0 for start in result['start_samples']))
        for value in (0,-1,True,2.0,'4',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                select_stage_windows(source,window_epochs=value)
        with self.assertRaises(ValueError):
            select_stage_windows(source,window_epochs=4,seed=-1)
        source['stage_epochs_four'][0]=2
        with self.assertRaisesRegex(ValueError,'disagree'):
            select_stage_windows(source,window_epochs=4)


class StageScoringTests(unittest.TestCase):
    def score(self,g,r,stages,starts=None,seed=42,**kwargs):
        return score_stage_draw(g,r,stages,start_samples=starts or [i*g.shape[1] for i in range(len(g))],
                                seed=seed,**kwargs)

    def test_reference_mask_moments_and_all_position_nonpositive_denominator(self):
        r=torch.tensor([[0.,1.,3.]*250]*2,dtype=torch.float64)
        g=torch.tensor([[-7.,2.,4.]*250,[9.,1.,3.]*250],dtype=torch.float64)
        stages=torch.tensor([[0],[2]],dtype=torch.int64)
        before=g.clone()
        result=self.score(g,r,stages,starts=[1500,4500])
        row=result['per_epoch'][0]
        self.assertEqual(row['observed_counts'],500)
        self.assertEqual(row['reference_mean_ms'],2000)
        self.assertEqual(row['generated_mean_ms'],3000)
        self.assertEqual(row['reference_variance_ms2'],1e6)
        self.assertEqual(row['generated_variance_ms2'],1e6)
        self.assertEqual(row['kl_mean_term'],.5)
        self.assertEqual(row['kl_variance_term'],0)
        self.assertEqual(row['generated_nonpositive_samples_all'],250)
        self.assertEqual(row['generated_nonpositive_fraction'],1/3)
        self.assertEqual(row['source_epoch_index'],2)
        self.assertEqual(row['start_second'],60)
        self.assertEqual(result['stages']['REM']['gaussian_kl_real_to_generated'],0)
        self.assertEqual(result['stages']['Light']['status'],'missing')
        self.assertIsNone(result['stages']['Light']['gaussian_kl_real_to_generated'])
        torch.testing.assert_close(g,before,rtol=0,atol=0)

    def test_stage_epochs_equal_weight_and_floor_independent_closed_form(self):
        r=torch.ones((2,1500),dtype=torch.float64)
        shifts=torch.tensor([1.,3.,5.,9.],dtype=torch.float64).repeat_interleave(750).reshape(2,1500)/1000
        g=r+shifts
        stages=torch.tensor([[0,0],[0,2]],dtype=torch.int64)
        result=self.score(g,r,stages)
        self.assertAlmostEqual(result['stages']['W']['mean_absolute_error_ms'],3,places=9)
        self.assertAlmostEqual(result['stages']['W']['gaussian_kl_real_to_generated'],35/6,places=7)
        self.assertAlmostEqual(result['stages']['REM']['gaussian_kl_real_to_generated'],81/2,places=7)
        self.assertAlmostEqual(result['overall']['gaussian_kl_real_to_generated'],14.5,places=7)
        self.assertEqual(result['stages']['W']['windows'],2)
        self.assertEqual(result['stages']['W']['epochs'],3)
        self.assertEqual(result['stages']['W']['observed_samples'],2250)
        self.assertEqual([row['source_epoch_index'] for row in result['per_epoch']],list(range(4)))

    def test_nonfinite_reference_masked_but_generated_never_excused(self):
        r=torch.ones((1,750),dtype=torch.float64)
        r[0,:2]=float('nan')
        g=torch.ones_like(r)
        stages=torch.tensor([[3]],dtype=torch.int64)
        result=self.score(g,r,stages)
        self.assertEqual(result['stages']['Deep']['observed_samples'],748)
        self.assertEqual(result['stages']['Deep']['gaussian_kl_real_to_generated'],0)
        g[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            self.score(g,r,stages)
        r.zero_();r[0,0]=1;g.fill_(1)
        with self.assertRaisesRegex(ValueError,'at least two'):
            self.score(g,r,stages)

    def test_overlap_alignment_and_invalid_stage_are_rejected(self):
        r=torch.ones((2,1500),dtype=torch.float64)
        stages=torch.tensor([[0,1],[2,3]],dtype=torch.int64)
        for starts in ([0,750],[0,0],[1,3000],[-750,3000]):
            with self.subTest(starts=starts),self.assertRaises(ValueError):
                self.score(r,r,stages,starts=starts)
        for bad in (stages.float(),torch.tensor([[0,1],[2,4]])):
            with self.assertRaises(ValueError):
                self.score(r,r,bad)

    def test_across_draw_population_std_missing_and_protocol_identity(self):
        r=torch.ones((1,750),dtype=torch.float64)
        stage=torch.tensor([[0]],dtype=torch.int64)
        a=self.score(r+.001,r,stage,seed=42)
        b=self.score(r+.003,r,stage,seed=43)
        summary=summarize_stage_draws([a,b])
        metric=summary['stages']['W']['metrics']['mean_error_ms']
        self.assertAlmostEqual(metric['mean'],2,places=9)
        self.assertAlmostEqual(metric['std'],1,places=9)
        self.assertEqual(summary['stages']['REM']['metrics']['mean_error_ms'],dict(mean=None,std=None))
        self.assertIn('not a confidence interval',summary['uncertainty'])
        self.assertEqual(summarize_stage_draws([a])['stages']['W']['metrics']['mean_error_ms']['std'],0)
        with self.assertRaisesRegex(ValueError,'Repeated seeds'):
            summarize_stage_draws([a,a])
        b['per_epoch'][0]['source_epoch_index']+=1
        with self.assertRaisesRegex(ValueError,'same source'):
            summarize_stage_draws([a,b])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable; numerical CPU tests only')
    def test_cuda_same_statistics_and_inputs_remain_on_device(self):
        r=torch.tensor([[0.,1.,3.]*250],dtype=torch.float64)
        g=r+.1
        stages=torch.tensor([[1]],dtype=torch.int64)
        cpu=self.score(g,r,stages)
        gpu_g,gpu_r,gpu_stages=g.cuda(),r.cuda(),stages.cuda()
        gpu=self.score(gpu_g,gpu_r,gpu_stages)
        for key,value in cpu['stages']['Light'].items():
            if isinstance(value,float):
                self.assertAlmostEqual(value,gpu['stages']['Light'][key],places=8)
        self.assertEqual(gpu_g.device.type,'cuda')
        torch.testing.assert_close(gpu_g,g.cuda(),rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
