"""Synthetic regression for replacing only stored IBI zeros by one second."""
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
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_utils import Preprocessor, ZERO_FILL_ONE_VALUE_POLICY, ZERO_VALUE_POLICY


class ZeroFillDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_root=(ROOT/'work'/'zero_fill_data_tests').resolve()
        self.temp_root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.temp_root)
        self.directory=Path(self.temp.name)
        self.signals={
            'A':np.tile(np.array([0,.5,1,1.5],dtype=np.float32),1500),
            'B':np.tile(np.array([0,.5,1,1.5],dtype=np.float32),1500),
            'V':np.tile(np.array([0,100,200,300],dtype=np.float32),1500),
            'T':np.full(6000,999.,dtype=np.float32),
        }
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

    def write_record(self,sid,*,stages=None,epoch_index=None):
        payload=dict(data=self.signals[sid][:,None],fs=25,ahi=self.ahi[sid],synthetic=True,
                     stages=np.array([0,1,2,3,4,1,0,3] if stages is None else stages))
        if epoch_index is not None:
            payload['epoch_index']=np.asarray(epoch_index)
        np.savez(self.directory/(sid+'.npz'),**payload)

    def processor(self,policy='zero_fill_one',**config):
        return Preprocessor('DREAMT',data_config=dict(data_dir=str(self.directory),window_epochs=4,
                                                     missing_value_policy=policy,**config))

    def test_training_scaler_fills_zeros_in_seconds_and_never_uses_validation_or_test(self):
        processor=self.processor()
        self.assertEqual(processor.scaler['signal_mean_seconds'],1.)
        self.assertAlmostEqual(processor.scaler['signal_std_seconds'],np.sqrt(.125))
        self.assertEqual(processor.scaler['signal_fit_samples'],12000)
        self.assertEqual(processor.scaler['fit_split'],'train')
        self.assertIn('1.0 second',processor.scaler['signal_fit_policy'])
        np.testing.assert_array_equal(processor.scaler['continuous_mean'],[10,25,30])
        np.testing.assert_array_equal(processor.scaler['continuous_std'],[5,5,10])
        self.assertEqual(processor.accessed_files,['A.npz','B.npz'])
        identity=processor.data_identity()
        processor.window_array('V.npz',0,3000)
        self.assertEqual(processor.data_identity(),identity)
        self.assertNotIn('T.npz',processor.accessed_files)
        with self.assertRaisesRegex(ValueError,'test is sealed'):
            processor.record('T.npz')

    def test_zero_replaced_before_zscore_raw_records_files_masks_and_positive_values_unchanged(self):
        original_files={sid:(self.directory/(sid+'.npz')).read_bytes() for sid in self.signals}
        processor=self.processor()
        records={sid:processor.record(sid+'.npz')['ibi'].copy() for sid in ('A','B','V')}
        for sid in ('A','V'):
            data=processor.window_array(sid+'.npz',0,3000)
            mask=processor.window_observed_mask(sid+'.npz',0,3000)
            raw=self.signals[sid][:3000]
            expected=np.where(raw==0,1.,raw)
            self.assertEqual(data.shape,(3000,10))
            self.assertEqual(data.dtype,np.float32)
            self.assertEqual(mask.dtype,np.bool_)
            np.testing.assert_allclose(processor.inverse_signal(data[:,0]),expected,rtol=1e-7,atol=1e-7)
            np.testing.assert_array_equal(mask,raw>0)
            # Filling raw zero gives standardized zero only because train mean is 1.
            self.assertEqual(data[0,0],0.)
            self.assertFalse(mask[0])
            if sid=='A':
                self.assertEqual(data[2,0],0.)
                self.assertTrue(mask[2])
            mask[:]=False
        for sid,raw in records.items():
            np.testing.assert_array_equal(processor.record(sid+'.npz')['ibi'],raw)
        for sid,contents in original_files.items():
            self.assertEqual((self.directory/(sid+'.npz')).read_bytes(),contents)

    def test_fill_is_one_second_not_standardized_zero_when_training_mean_differs(self):
        self.signals['A']*=2
        self.signals['B']*=2
        for sid in ('A','B'):
            self.write_record(sid)
        processor=self.processor()
        self.assertEqual(processor.scaler['signal_mean_seconds'],1.75)
        data=processor.window_array('V.npz',0,3000)
        self.assertNotEqual(float(data[0,0]),0.)
        self.assertAlmostEqual(float(processor.inverse_signal(data[0,0])),1.,places=7)
        self.assertAlmostEqual(float(processor.inverse_signal(data[3,0])),300.,places=4)

    def test_training_returns_plain_tensor_and_all_zero_windows_are_valid(self):
        for sid in ('A','V'):
            self.signals[sid][:3000]=0
            self.write_record(sid)
        processor=self.processor()
        dataset=processor.make_training_dataset(3000,750)
        self.assertIn(('A.npz',0),dataset.window_index)
        tensor=dataset[dataset.window_index.index(('A.npz',0))]
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(tuple(tensor.shape),(3000,10))
        np.testing.assert_allclose(processor.inverse_signal(tensor[:,0].numpy()),np.ones(3000),atol=1e-7)
        self.assertFalse(processor.window_observed_mask('A.npz',0,3000).any())
        batch=next(iter(DataLoader(dataset,batch_size=2)))
        self.assertTrue(torch.is_tensor(batch))
        self.assertEqual(tuple(batch.shape),(2,3000,10))
        self.assertIn(('V.npz',0),processor.make_validation_dataset(stride=750).window_index)
        self.assertIn(('V.npz',0),processor.select_validation_windows(99))
        self.assertFalse(processor.window_observed_mask('V.npz',0,3000).any())

    def test_scaler_counts_source_epochs_once_not_overlapping_training_windows(self):
        self.signals['A']=np.repeat(np.arange(8,dtype=np.float32)/2,750)
        self.write_record('A')
        processor=self.processor()
        original=np.concatenate((self.signals['A'],self.signals['B'])).astype(np.float64)
        expected=np.where(original==0,1.,original)
        self.assertEqual(processor.scaler['signal_fit_samples'],12000)
        self.assertAlmostEqual(processor.scaler['signal_mean_seconds'],expected.mean())
        self.assertAlmostEqual(processor.scaler['signal_std_seconds'],expected.std(ddof=0))
        overlapping=processor.make_training_dataset(3000,750)
        self.assertEqual(len(overlapping),10)
        self.assertNotEqual(len(overlapping)*3000,processor.scaler['signal_fit_samples'])

    def test_structural_invalid_epochs_and_gaps_still_reject_without_high_positive_clipping(self):
        self.signals['A'][750]=np.nan
        self.signals['A'][2250]=-.5
        self.signals['A'][3000]=1000.
        self.write_record('A',stages=[0,1,5,3,4,1,0,3],epoch_index=[0,1,2,3,4,5,7,8])
        processor=Preprocessor('DREAMT',data_config=dict(data_dir=str(self.directory),window_epochs=2,
                                                        missing_value_policy='zero_fill_one'))
        record=processor.record('A.npz')
        np.testing.assert_array_equal(record['valid_epochs'],[True,False,False,False,True,True,True,True])
        actual=[start for name,start in processor.make_training_dataset(1500,750).window_index if name=='A.npz']
        self.assertEqual(actual,[3000,4500])
        for start in (0,750,1500,2250,3750):
            with self.subTest(start=start),self.assertRaisesRegex(ValueError,'invalid epoch or known time gap'):
                processor.window_array('A.npz',start,1500)
        filled=[]
        for filename in processor.splits['train']:
            row=processor.record(filename)
            signal=row['ibi'].reshape(-1,750)[row['valid_epochs']].astype(np.float64)
            filled.append(np.where(signal==0,1.,signal).reshape(-1))
        expected=np.concatenate(filled)
        self.assertEqual(processor.scaler['signal_fit_samples'],len(expected))
        self.assertAlmostEqual(processor.scaler['signal_mean_seconds'],expected.mean())
        self.assertAlmostEqual(processor.scaler['signal_std_seconds'],expected.std(ddof=0))
        restored=processor.inverse_signal(processor.window_array('A.npz',3000,1500)[:,0])
        self.assertAlmostEqual(float(restored[0]),1000.,places=3)

    def test_metadata_subject_weights_unchanged_for_all_zero_training_subject(self):
        self.signals['A'][:]=0
        self.write_record('A')
        filled=self.processor()
        preserved=self.processor('preserve')
        self.assertEqual(filled.scaler['metadata_fit_subjects'],2)
        self.assertEqual(filled.scaler['continuous_mean'],preserved.scaler['continuous_mean'])
        self.assertEqual(filled.scaler['continuous_std'],preserved.scaler['continuous_std'])
        self.assertEqual(filled.scaler['signal_fit_samples'],12000)
        self.assertEqual(filled.scaler_subjects,['A','B'])

    def test_identity_roundtrip_is_explicit_ignores_live_configuration_and_rejects_tampering(self):
        processor=self.processor()
        identity=processor.data_identity()
        self.assertEqual(identity['missing_value_policy'],'zero_fill_one')
        self.assertEqual(identity['zero_value_policy'],ZERO_FILL_ONE_VALUE_POLICY)
        unrelated=self.directory/'unrelated.json'
        unrelated.write_text(json.dumps(dict(data_dir='missing',missing_value_policy='masked')),encoding='utf-8')
        with mock.patch.dict(os.environ,{'DREAMT_CONFIG_PATH':str(unrelated),'DREAMT_DATA_DIR':'missing'}):
            restored=Preprocessor.from_identity(identity)
        self.assertEqual(restored.data_identity(),identity)
        np.testing.assert_array_equal(restored.window_array('A.npz',0,3000),processor.window_array('A.npz',0,3000))
        variants=[]
        for policy in ('preserve','masked'):
            changed=copy.deepcopy(identity);changed['missing_value_policy']=policy;variants.append(changed)
        changed=copy.deepcopy(identity);del changed['missing_value_policy'];variants.append(changed)
        changed=copy.deepcopy(identity);changed['zero_value_policy']=ZERO_VALUE_POLICY;variants.append(changed)
        changed=copy.deepcopy(identity);changed['scaler']['signal_mean_seconds']+=.01;variants.append(changed)
        changed=copy.deepcopy(identity);changed['scaler']['signal_fit_policy']='wrong';variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed.get('missing_value_policy')),self.assertRaisesRegex(ValueError,'identity mismatch'):
                Preprocessor.from_identity(changed)
        provenance=restored.provenance()
        self.assertEqual(provenance['missing_value_policy'],'zero_fill_one')
        self.assertEqual(provenance['zero_value_policy'],ZERO_FILL_ONE_VALUE_POLICY)
        self.assertEqual(len(provenance['columns']),10)

    def test_fill_making_all_training_points_constant_uses_existing_zero_variance_rejection(self):
        for sid in ('A','B'):
            self.signals[sid][:]=np.tile([0.,1.],3000)
            self.write_record(sid)
        # Stored data has variance, but the requested 0->1 transform does not.
        self.assertGreater(self.processor('preserve').scaler['signal_std_seconds'],0)
        with self.assertRaisesRegex(ValueError,'nonzero variance'):
            self.processor()
        for sid in ('A','B'):
            self.signals[sid][:]=0
            self.write_record(sid)
        with self.assertRaisesRegex(ValueError,'nonzero variance'):
            self.processor()

    def test_preserve_and_masked_legacy_behavior_and_identity_are_unchanged(self):
        preserved=self.processor('preserve')
        masked=self.processor('masked')
        self.assertNotIn('missing_value_policy',preserved.data_identity())
        self.assertEqual(preserved.scaler['signal_mean_seconds'],.75)
        self.assertAlmostEqual(preserved.scaler['signal_std_seconds'],np.sqrt(.3125))
        self.assertEqual(masked.data_identity()['missing_value_policy'],'masked')
        self.assertEqual(masked.scaler['signal_fit_samples'],9000)
        self.assertAlmostEqual(masked.scaler['signal_std_seconds'],np.sqrt(1/6))
        self.assertTrue(torch.is_tensor(preserved.make_training_dataset(3000,750)[0]))
        self.assertEqual(set(masked.make_training_dataset(3000,750)[0]),{'data','observed_mask'})
        for processor in (preserved,masked):
            self.assertEqual(Preprocessor.from_identity(processor.data_identity()).data_identity(),processor.data_identity())


if __name__=='__main__':
    unittest.main()
