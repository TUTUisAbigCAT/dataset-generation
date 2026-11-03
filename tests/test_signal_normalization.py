"""Fixed-bound clean IBI normalization, affine metadata, and legacy compatibility."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_utils import Preprocessor
from signal_normalization import (normalization_spec, normalization_affine_parameters,
                                  signal_affine_parameters, validate_signal_bounds)

MINMAX=dict(method='minmax',min_seconds=0.,max_seconds=2.)
MINMAX01=dict(MINMAX,output_range=[0.,1.])


class AffineHelperTests(unittest.TestCase):
    def test_default_and_explicit_zscore_use_actual_training_mean_and_std(self):
        scaler=dict(signal_mean_seconds=.8,signal_std_seconds=.2)
        self.assertIsNone(normalization_spec({}))
        self.assertIsNone(normalization_spec({'signal_normalization':{'method':'zscore'}}))
        self.assertEqual(signal_affine_parameters(scaler),(.8,.2))
        self.assertEqual(signal_affine_parameters(dict(scaler,signal_normalization={'method':'zscore'})),(.8,.2))

    def test_minmax_uses_separate_parameters_not_mean_std_and_requires_consistent_metadata(self):
        scaler=dict(signal_mean_seconds=.8,signal_std_seconds=.2,signal_normalization=MINMAX.copy(),
                    signal_center_seconds=1.,signal_scale_seconds=1.)
        self.assertEqual(signal_affine_parameters(scaler),(1.,1.))
        for key in ('signal_center_seconds','signal_scale_seconds','signal_normalization'):
            changed=copy.deepcopy(scaler);del changed[key]
            with self.subTest(key=key),self.assertRaises(ValueError):
                signal_affine_parameters(changed)
        for key,value in (('signal_center_seconds',.8),('signal_scale_seconds',.2),
                          ('signal_center_seconds',True),('signal_scale_seconds',float('inf'))):
            changed=copy.deepcopy(scaler);changed[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                signal_affine_parameters(changed)
        changed=copy.deepcopy(scaler);changed['signal_normalization']['max_seconds']=3.
        with self.assertRaisesRegex(ValueError,'disagree'):
            signal_affine_parameters(changed)

    def test_strict_normalization_schema_rejects_null_typos_invalid_bounds(self):
        invalid=[None,True,'minmax',{}, {'method':'MINMAX'}, {'method':'zscore','min_seconds':0},
                 {'method':'minmax','min_seconds':0},dict(MINMAX,clip=True)]
        invalid += [dict(MINMAX,min_seconds=value) for value in (True,'0',float('nan'),float('inf'))]
        invalid += [dict(MINMAX,max_seconds=value) for value in (0,-1,False,float('inf'))]
        invalid += [dict(method='minmax',min_seconds=-1e308,max_seconds=1e308)]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalization_spec({'signal_normalization':value})
        original={'signal_normalization':dict(method='minmax',min_seconds=0,max_seconds=2)}
        normalized=normalization_spec(original)
        self.assertEqual(normalized,MINMAX)
        normalized['max_seconds']=5.
        self.assertEqual(original['signal_normalization']['max_seconds'],2)

    def test_clean_bounds_are_strict_and_never_modify_input(self):
        valid=np.array([0,.5,1,2.],dtype=np.float64)
        before=valid.copy()
        validate_signal_bounds(valid,MINMAX)
        np.testing.assert_array_equal(valid,before)
        for invalid in (np.array([-1e-12,1.]),np.array([1.,2.+1e-12]),np.array([np.nan]),np.array([np.inf])):
            before=invalid.copy()
            with self.assertRaises(ValueError):
                validate_signal_bounds(invalid,MINMAX,context='fixture')
            np.testing.assert_array_equal(invalid,before)

    def test_output_range_canonicalization_preserves_legacy_and_does_not_mutate_config(self):
        explicit=dict(MINMAX,output_range=[-1,1])
        self.assertEqual(normalization_spec({'signal_normalization':explicit}),MINMAX)
        self.assertEqual(normalization_affine_parameters(explicit),(1.,1.))
        source={'signal_normalization':dict(MINMAX,output_range=[0,1])}
        canonical=normalization_spec(source)
        self.assertEqual(canonical,MINMAX01)
        self.assertEqual(normalization_affine_parameters(canonical),(0.,2.))
        canonical['output_range'][0]=.5
        self.assertEqual(source['signal_normalization']['output_range'],[0,1])
        self.assertEqual(explicit['output_range'],[-1,1])
        # The affine formula also applies to a nonzero physical lower bound.
        self.assertEqual(normalization_affine_parameters(dict(MINMAX01,min_seconds=.5,max_seconds=2.5)),(.5,2.))
        with self.assertRaises(ValueError):
            normalization_affine_parameters({'method':'zscore'})

    def test_output_range_strict_validation_and_frozen_affine_crosscheck(self):
        for value in (None,True,'[0,1]',(0,1),[],[0],[0,1,2],[False,1],
                      [0,'1'],[0,float('nan')],[0,float('inf')],[1,0],[0,2],[-1,0]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalization_spec({'signal_normalization':dict(MINMAX,output_range=value)})
        scaler=dict(signal_mean_seconds=.8,signal_std_seconds=.2,
                    signal_normalization=copy.deepcopy(MINMAX01),
                    signal_center_seconds=0.,signal_scale_seconds=2.)
        self.assertEqual(signal_affine_parameters(scaler),(0.,2.))
        for center,scale in ((1.,1.),(0.,1.),(1.,2.)):
            changed=dict(scaler,signal_center_seconds=center,signal_scale_seconds=scale)
            with self.subTest(center=center,scale=scale),self.assertRaisesRegex(ValueError,'disagree'):
                signal_affine_parameters(changed)
        changed=copy.deepcopy(scaler);del changed['signal_normalization']['output_range']
        with self.assertRaisesRegex(ValueError,'disagree'):
            signal_affine_parameters(changed)

    def test_output_range_zero_one_leaves_clean_bounds_and_unbounded_inverse_semantics(self):
        values=np.array([0.,1.,2.])
        center,scale=normalization_affine_parameters(MINMAX01)
        validate_signal_bounds(values,MINMAX01)
        np.testing.assert_array_equal((values-center)/scale,[0.,.5,1.])
        # Production downstream code applies this same affine to device tensors.
        normalized=torch.tensor([-1.,0.,.5,1.,2.],dtype=torch.float64)
        np.testing.assert_array_equal((normalized*scale+center).numpy(),[-2.,0.,1.,2.,4.])
        for invalid in (np.array([-1e-12]),np.array([2.+1e-12])):
            with self.assertRaises(ValueError):
                validate_signal_bounds(invalid,MINMAX01)


class SignalNormalizationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_root=(ROOT/'work'/'signal_normalization_tests').resolve()
        self.temp_root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.temp_root)
        self.directory=Path(self.temp.name)
        self.signals={sid:np.tile(np.array([0,.5,1,1.5],dtype=np.float32),1500) for sid in ('A','B','V','T')}
        self.ahi=dict(A=5.,B=15.,V=25.,T=35.)
        for sid in self.signals:
            self.write_record(sid)
        (self.directory/'participant_info.csv').write_text(
            'SID,AHI,BMI,AGE,GENDER\nA,5,20,20,F\nB,15,30,40,M\nV,25,40,60,F\nT,35,50,80,M\n',encoding='utf-8')
        (self.directory/'splits.json').write_text(json.dumps(
            dict(train=['A.npz','B.npz'],val=['V.npz'],test=['T.npz'])),encoding='utf-8')

    def tearDown(self):
        resolved=self.directory.resolve()
        self.assertNotEqual(resolved,self.temp_root)
        self.assertTrue(resolved.is_relative_to(self.temp_root))
        self.temp.cleanup()

    def write_record(self,sid,stages=None):
        np.savez(self.directory/(sid+'.npz'),data=self.signals[sid][:,None],fs=25,ahi=self.ahi[sid],
                 stages=np.array([0,1,2,3,4,1,0,3] if stages is None else stages),synthetic=True)

    def processor(self,*,policy='preserve',spec='absent'):
        config=dict(data_dir=str(self.directory),window_epochs=4,missing_value_policy=policy)
        if spec!='absent':
            config['signal_normalization']=copy.deepcopy(spec)
        return Preprocessor('DREAMT',data_config=config)

    def test_default_explicit_zscore_and_all_missing_policies_keep_legacy_identity(self):
        for policy in ('preserve','masked','zero_fill_one'):
            default=self.processor(policy=policy)
            explicit=self.processor(policy=policy,spec={'method':'zscore'})
            self.assertIsNone(default.signal_normalization)
            self.assertIsNone(explicit.signal_normalization)
            self.assertEqual(default.data_identity(),explicit.data_identity())
            self.assertNotIn('signal_normalization',default.data_identity())
            self.assertNotIn('signal_normalization',default.scaler)
            self.assertNotIn('signal_center_seconds',default.scaler)
            self.assertNotIn('signal_scale_seconds',default.scaler)
            np.testing.assert_array_equal(default.window_array('A.npz',0,3000),explicit.window_array('A.npz',0,3000))

    def test_preserve_minmax_keeps_raw_zeros_and_training_statistics_but_maps_clean_values(self):
        original=(self.directory/'A.npz').read_bytes()
        baseline=self.processor()
        processor=self.processor(spec=MINMAX)
        scaler=processor.scaler
        self.assertEqual(scaler['signal_mean_seconds'],.75)
        self.assertAlmostEqual(scaler['signal_std_seconds'],np.sqrt(.3125))
        self.assertEqual(signal_affine_parameters(scaler),(1.,1.))
        for key in baseline.scaler:
            self.assertEqual(scaler[key],baseline.scaler[key])
        window=processor.window_array('A.npz',0,3000)
        expected=np.tile(np.array([-1.,-.5,0.,.5],dtype=np.float32),750)
        np.testing.assert_array_equal(window[:,0],expected)
        np.testing.assert_array_equal(processor.inverse_signal(window[:,0]),self.signals['A'][:3000])
        np.testing.assert_array_equal(window[:,1:],baseline.window_array('A.npz',0,3000)[:,1:])
        np.testing.assert_array_equal(processor.record('A.npz')['ibi'],self.signals['A'])
        self.assertEqual((self.directory/'A.npz').read_bytes(),original)
        np.testing.assert_array_equal(processor.window_observed_mask('A.npz',0,3000),self.signals['A'][:3000]>0)
        self.assertTrue(torch.is_tensor(processor.make_training_dataset(3000,750)[0]))
        self.assertEqual(processor.provenance()['signal_normalization'],MINMAX)

    def test_explicit_legacy_output_range_keeps_identity_and_arrays_for_all_policies(self):
        for policy in ('preserve','masked','zero_fill_one'):
            implicit=self.processor(policy=policy,spec=MINMAX)
            explicit=self.processor(policy=policy,spec=dict(MINMAX,output_range=[-1.,1.]))
            with self.subTest(policy=policy):
                self.assertEqual(explicit.data_identity(),implicit.data_identity())
                self.assertEqual(explicit.signal_normalization,MINMAX)
                self.assertNotIn('output_range',explicit.scaler['signal_normalization'])
                np.testing.assert_array_equal(explicit.window_array('A.npz',0,3000),
                                              implicit.window_array('A.npz',0,3000))

    def test_minmax_zero_one_endpoints_true_statistics_conditions_and_masks_are_unchanged(self):
        self.signals['V'][:3]=[0.,1.,2.]
        self.write_record('V')
        before=(self.directory/'V.npz').read_bytes()
        baseline=self.processor(spec=MINMAX)
        processor=self.processor(spec=MINMAX01)
        for key in baseline.scaler:
            if key not in ('signal_normalization','signal_center_seconds','signal_scale_seconds'):
                self.assertEqual(processor.scaler[key],baseline.scaler[key])
        self.assertEqual(processor.scaler['signal_mean_seconds'],.75)
        self.assertAlmostEqual(processor.scaler['signal_std_seconds'],np.sqrt(.3125))
        self.assertEqual(signal_affine_parameters(processor.scaler),(0.,2.))
        data=processor.window_array('V.npz',0,3000)
        np.testing.assert_array_equal(data[:3,0],[0.,.5,1.])
        np.testing.assert_array_equal(processor.inverse_signal(data[:,0]),self.signals['V'][:3000])
        np.testing.assert_array_equal(data[:,1:],baseline.window_array('V.npz',0,3000)[:,1:])
        np.testing.assert_array_equal(processor.record('V.npz')['ibi'],self.signals['V'])
        np.testing.assert_array_equal(processor.window_observed_mask('V.npz',0,3000),self.signals['V'][:3000]>0)
        self.assertEqual((self.directory/'V.npz').read_bytes(),before)
        np.testing.assert_array_equal(processor.inverse_signal(np.array([-1.,0.,.5,1.,2.])),[-2.,0.,1.,2.,4.])
        self.assertTrue(torch.is_tensor(processor.make_training_dataset(3000,750)[0]))

    def test_minmax_zero_one_identity_restores_frozen_range_and_rejects_changed_range(self):
        processor=self.processor(spec=MINMAX01)
        identity=processor.data_identity()
        self.assertEqual(identity['signal_normalization'],MINMAX01)
        self.assertEqual(identity['scaler']['signal_normalization'],MINMAX01)
        live=self.directory/'live.json'
        live.write_text(json.dumps(dict(data_dir='invalid',signal_normalization=MINMAX)),encoding='utf-8')
        with mock.patch.dict(os.environ,{'DREAMT_CONFIG_PATH':str(live),'DREAMT_DATA_DIR':'invalid'}):
            restored=Preprocessor.from_identity(identity)
        self.assertEqual(restored.data_identity(),identity)
        np.testing.assert_array_equal(restored.window_array('A.npz',0,3000),processor.window_array('A.npz',0,3000))
        self.assertEqual(restored.provenance()['signal_normalization'],MINMAX01)
        for owner in ('outer','scaler'):
            changed=copy.deepcopy(identity)
            target=changed if owner=='outer' else changed['scaler']
            target['signal_normalization']['output_range']=[-1.,1.]
            with self.subTest(owner=owner),self.assertRaisesRegex(ValueError,'identity mismatch'):
                Preprocessor.from_identity(changed)

    def test_endpoint_two_seconds_maps_to_one_without_clipping_inverse_outputs(self):
        self.signals['V'][:4]=[0,2,.5,1]
        self.write_record('V')
        processor=self.processor(spec=MINMAX)
        data=processor.window_array('V.npz',0,3000)
        np.testing.assert_array_equal(data[:4,0],[-1.,1.,-.5,0.])
        # No promise that noisy xt or generated samples remain in [-1,1].
        np.testing.assert_array_equal(processor.inverse_signal(np.array([-2.,-1.,0.,1.,2.])),[-1.,0.,1.,2.,3.])

    def test_zero_replacement_and_minmax_are_independent_options_in_correct_order(self):
        a=self.processor(policy='zero_fill_one')
        b=self.processor(spec=MINMAX)
        both=self.processor(policy='zero_fill_one',spec=MINMAX)
        self.assertEqual(a.window_array('A.npz',0,3000)[0,0],0.)
        self.assertEqual(b.window_array('A.npz',0,3000)[0,0],-1.)
        self.assertEqual(both.window_array('A.npz',0,3000)[0,0],0.)
        self.assertEqual(a.scaler['signal_mean_seconds'],1.)
        self.assertEqual(b.scaler['signal_mean_seconds'],.75)
        self.assertEqual(both.scaler['signal_mean_seconds'],1.)
        np.testing.assert_array_equal(both.inverse_signal(both.window_array('A.npz',0,3000)[:4,0]),[1.,.5,1.,1.5])

    def test_outside_bounds_training_fails_but_structurally_invalid_epochs_remain_excluded(self):
        self.signals['A'][0]=2.1
        self.write_record('A')
        for policy in ('preserve','masked','zero_fill_one'):
            with self.subTest(policy=policy),self.assertRaisesRegex(ValueError,'Training A.npz.*outside fixed minmax'):
                self.processor(policy=policy,spec=MINMAX)
        # Unknown-stage epochs cannot enter the original training scaler.
        self.write_record('A',stages=[5,1,2,3,4,1,0,3])
        processor=self.processor(spec=MINMAX)
        self.assertEqual(processor.scaler['signal_fit_samples'],11250)
        with self.assertRaisesRegex(ValueError,'invalid epoch'):
            processor.window_array('A.npz',0,3000)

    def test_heldout_bounds_fail_when_that_window_is_loaded_never_refit_or_clip(self):
        self.signals['V'][3000]=2.1
        self.signals['T'][:]=999.
        self.write_record('V');self.write_record('T')
        processor=self.processor(spec=MINMAX)
        before=processor.data_identity()
        valid=processor.window_array('V.npz',0,3000)
        self.assertTrue(np.all((valid[:,0]>=-1)&(valid[:,0]<=1)))
        with self.assertRaisesRegex(ValueError,'Window V.npz:3000.*outside fixed minmax'):
            processor.window_array('V.npz',3000,3000)
        self.assertEqual(processor.data_identity(),before)
        self.assertEqual(float(processor.record('V.npz')['ibi'][3000]),float(np.float32(2.1)))
        self.assertNotIn('T.npz',processor.accessed_files)

    def test_identity_frozen_spec_roundtrip_and_bounds_affine_statistics_tampering(self):
        processor=self.processor(spec=MINMAX)
        identity=processor.data_identity()
        self.assertEqual(identity['signal_normalization'],MINMAX)
        self.assertEqual(identity['scaler']['signal_normalization'],MINMAX)
        bad=self.directory/'live.json'
        bad.write_text(json.dumps(dict(data_dir='invalid',signal_normalization={'method':'zscore'})),encoding='utf-8')
        with mock.patch.dict(os.environ,{'DREAMT_CONFIG_PATH':str(bad),'DREAMT_DATA_DIR':'invalid'}):
            restored=Preprocessor.from_identity(identity)
        self.assertEqual(restored.data_identity(),identity)
        np.testing.assert_array_equal(restored.window_array('A.npz',0,3000),processor.window_array('A.npz',0,3000))
        variants=[]
        changed=copy.deepcopy(identity);del changed['signal_normalization'];variants.append(changed)
        changed=copy.deepcopy(identity);changed['signal_normalization']['max_seconds']=3.;variants.append(changed)
        for key in ('signal_center_seconds','signal_scale_seconds','signal_mean_seconds','signal_std_seconds'):
            changed=copy.deepcopy(identity);changed['scaler'][key]+=.01;variants.append(changed)
        changed=copy.deepcopy(identity);del changed['scaler']['signal_normalization'];variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed['scaler']),self.assertRaisesRegex(ValueError,'identity mismatch'):
                Preprocessor.from_identity(changed)

    def test_fixed_bounds_do_not_change_existing_constant_training_signal_rejection(self):
        for sid in ('A','B'):
            self.signals[sid][:]=1.
            self.write_record(sid)
        with self.assertRaisesRegex(ValueError,'nonzero variance'):
            self.processor(spec=MINMAX)


if __name__=='__main__':
    unittest.main()
