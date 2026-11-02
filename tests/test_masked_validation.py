"""Masked validation math and integration; CPU is explicitly mocked for tests."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import epoch_validation as coordinator
import validation_metrics as metrics
from diffusion_schedule import build_diffusion_config


def kl(generated, reference, mask=None, **options):
    return metrics.summarize_distribution(generated, reference, observed_mask=mask, **options)["aggregate"]["gaussian_kl_real_to_generated"]


def numpy_kl(generated, reference, mask, floor=1.):
    scores = []
    for g, r, selected in zip(generated, reference, mask):
        g, r = np.asarray(g)[selected]*1000, np.asarray(r)[selected]*1000
        vg, vr = max(float(g.var()), floor), max(float(r.var()), floor)
        scores.append(.5*(math.log(vg/vr)+(vr+(r.mean()-g.mean())**2)/vg-1))
    return float(np.mean(scores))


class MaskedMomentTests(unittest.TestCase):
    def test_analytic_same_variance_mean_shift(self):
        real = torch.tensor([[1., 3., 0., 0.]], dtype=torch.float64)
        generated = torch.tensor([[2., 4., -99., 10000.]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False, False]])
        self.assertEqual(kl(generated, real, mask), .5)
        self.assertNotAlmostEqual(kl(generated, real), .5)

    def test_excluded_values_do_not_change_either_fit(self):
        mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
        real = torch.tensor([[.7, 0., 1.1, 0.], [0., .8, 1., 1.4]], dtype=torch.float64)
        generated = torch.tensor([[.6, 15., 1.2, -1.], [99., .9, 1.1, 1.5]], dtype=torch.float64)
        before_real, before_gen = real.clone(), generated.clone()
        expected = kl(generated, real, mask)
        changed_real = real.clone()
        changed_real[~mask] = torch.tensor([float("nan"), float("inf"), -1e300], dtype=torch.float64)
        changed_gen = generated.clone()
        changed_gen[~mask] = torch.tensor([1e100, -1e100, -500.], dtype=torch.float64)
        self.assertEqual(kl(changed_gen, changed_real, mask), expected)
        self.assertTrue(torch.equal(real, before_real))
        self.assertTrue(torch.equal(generated, before_gen))
        self.assertAlmostEqual(expected, numpy_kl(generated.numpy(), real.numpy(), mask.numpy()), places=12)

    def test_negative_generated_values_remain_selected(self):
        real = torch.tensor([[1., 3., 0.]], dtype=torch.float64)
        generated = torch.tensor([[-3., -1., 10.]], dtype=torch.float64)
        self.assertEqual(kl(generated, real, torch.tensor([[True, True, False]])), 8.)

    def test_variance_floor_is_applied_after_masked_moments(self):
        real = torch.tensor([[1., 1., 0.]])
        generated = torch.tensor([[2., 2., 100.]])
        self.assertEqual(kl(generated, real, torch.tensor([[True, True, False]])), 500000.)

    def test_every_generated_position_must_be_finite(self):
        real = torch.tensor([[1., 2., 0.]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        for value in (float("nan"), float("inf"), float("-inf"), 1e308):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Generated IBI must be finite"):
                kl(torch.tensor([[1., 2., value]], dtype=torch.float64), real, mask)
        with self.assertRaisesRegex(ValueError, "Reference IBI must be finite"):
            kl(real, torch.tensor([[float("nan"), 2., 0.]], dtype=torch.float64), mask)

    def test_mask_must_be_boolean_same_shape_same_device_with_two_points(self):
        values = torch.ones((2, 4))
        for bad in (np.ones((2,4), dtype=bool), torch.ones((2,4)), torch.ones((2,4),dtype=torch.int64)):
            with self.subTest(mask=type(bad)), self.assertRaises(TypeError):
                kl(values,values,bad)
        with self.assertRaisesRegex(ValueError, "shape"):
            kl(values,values,torch.ones((2,3),dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "same device"):
            kl(values,values,torch.ones((2,4),dtype=torch.bool,device="meta"))
        for selected in ((False,False,False,False),(True,False,False,False)):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, "at least two"):
                kl(values,values,torch.tensor([selected,(True,True,True,True)]))

    def test_uneven_batches_use_equal_window_not_observed_point_weights(self):
        real = torch.tensor([[1.,3.,0.,0.],[1.,2.,3.,0.],[1.,2.,3.,4.]],dtype=torch.float64)
        mask = real > 0
        generated = real + torch.tensor([[1.],[2.],[4.]])
        expected = numpy_kl(generated.numpy(),real.numpy(),mask.numpy())
        scores = [numpy_kl(generated[i:i+1].numpy(),real[i:i+1].numpy(),mask[i:i+1].numpy()) for i in range(3)]
        self.assertNotAlmostEqual(expected,float(np.average(scores,weights=mask.sum(1).numpy())))
        accumulator=metrics.DistributionAccumulator()
        accumulator.update(generated[:2],real[:2],observed_mask=mask[:2])
        accumulator.update(generated[2:],real[2:],observed_mask=mask[2:])
        self.assertAlmostEqual(accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"],expected,places=12)
        self.assertAlmostEqual(kl(generated,real,mask),expected,places=12)

    def test_none_keeps_legacy_var_mean_reduction_and_zeros(self):
        real=torch.tensor([[0.,1.,2.,3.]],dtype=torch.float64)
        generated=real+.5
        with mock.patch.object(metrics,"_masked_var_mean",side_effect=AssertionError("legacy must not use masked moments")), \
             mock.patch.object(torch,"var_mean",wraps=torch.var_mean) as reduction:
            value=kl(generated,real)
        self.assertEqual(reduction.call_count,2)
        self.assertAlmostEqual(value,.1, places=14)
        self.assertEqual(value,kl(generated,real,None))
        self.assertNotEqual(value,kl(generated,real,real>0))


class MaskedPreprocessor:
    def __init__(self, policy="masked", windows=3):
        self.window_epochs,self.window_size=1,750
        self.missing_value_policy=policy
        self.splits={"train":["train.npz"],"val":[f"v{i}.npz" for i in range(windows)],"test":["sealed.npz"]}
        self.scaler={"signal_mean_seconds":1.,"signal_std_seconds":.5,"fit_split":"train"}
        self.split_fingerprint="synthetic-split"
        self.columns=np.array(["IBI", "W", "Light", "REM", "Deep", "AHI", "BMI", "AGE", "F", "M"])
        self.records={}
        for i,name in enumerate(self.splits["val"]):
            ibi=np.zeros(750,dtype=np.float64)
            ibi[:2+i]=np.linspace(.5,1.5,2+i)
            self.records[name]=dict(ibi=ibi,stages=np.zeros(750,dtype=np.int8),file_sha256=str(i)*64)
        self.mask_calls=[]
    def record(self,name):
        if name not in self.splits["val"]:
            raise AssertionError("Validation touched non-validation data")
        return self.records[name]
    def window_array(self,name,start,length):
        result=np.zeros((length,10),dtype=np.float32)
        result[:,0]=(self.record(name)["ibi"][start:start+length]-1)/.5
        if self.missing_value_policy=="masked":
            result[self.record(name)["ibi"][start:start+length]==0,0]=0
        result[:,1]=1
        result[:,5:]=37
        return result
    def window_observed_mask(self,name,start,length):
        self.mask_calls.append((name,start,length))
        return self.record(name)["ibi"][start:start+length]>0
    def make_validation_dataset(self,*,window_size,stride):
        return SimpleNamespace(window_index=[(name,0) for name in self.splits["val"]])
    def provenance(self):
        return {"data_synthetic":True}


class MaskedEpochValidationTests(unittest.TestCase):
    def make_validator(self,processor,options=None):
        temporary=tempfile.TemporaryDirectory(prefix="masked_validation_",dir=ROOT)
        self.addCleanup(temporary.cleanup)
        model=torch.nn.Linear(1,1).train()
        for parameter in model.parameters():
            parameter.grad=torch.ones_like(parameter)*.25
        args=SimpleNamespace(window_size=750,s4_lmax=750,timesteps=3,beta_0=.0001,beta_T=.02,
                             diffusion_schedule="linear",terminal_alpha_bar=1e-8)
        config={"window_epochs":1,"validation":{"batch_size":2,**(options or {})}}
        with mock.patch.object(coordinator,"_require_device",return_value=torch.device("cpu")):
            validator=coordinator.EpochValidator(processor,args,model,build_diffusion_config(args),"cpu",Path(temporary.name)/"validation",config)
        return validator,model

    def test_mask_protocol_records_frozen_pairing_and_per_window_coverage(self):
        processor=MaskedPreprocessor()
        validator,_=self.make_validator(processor)
        self.assertEqual(validator.reference_mask.dtype,torch.bool)
        self.assertEqual(validator.reference_mask.sum(1).tolist(),[2,3,4])
        self.assertEqual(validator.protocol["schema"],"ibi_epoch_masked_window_gaussian_kl_v5")
        self.assertIn("same fixed real-data mask",validator.protocol["reference_mask"]["pairing"])
        self.assertIn("never supplied",validator.protocol["reference_mask"]["generation_dependency"])
        self.assertEqual([entry["reference_mask"]["observed_samples"] for entry in validator.windows_metadata],[2,3,4])
        self.assertEqual([entry["reference_mask"]["missing_samples"] for entry in validator.windows_metadata],[748,747,746])
        frozen=validator.reference_mask.clone()
        processor.records["v0.npz"]["ibi"][:]=1
        self.assertTrue(torch.equal(frozen,validator.reference_mask))
        json.dumps(validator.protocol,allow_nan=False)

    def test_preserve_default_retains_original_protocol_and_does_not_request_masks(self):
        processor=MaskedPreprocessor("preserve")
        validator,_=self.make_validator(processor)
        self.assertIsNone(validator.reference_mask)
        self.assertEqual(processor.mask_calls,[])
        self.assertEqual(validator.protocol["schema"],"ibi_epoch_window_gaussian_kl_v4")
        self.assertNotIn("reference_mask",validator.protocol)
        self.assertIn("stored zeros included",validator.protocol["aggregation"])

    def test_observed_only_gives_both_training_policies_identical_reference_masks(self):
        processors=[MaskedPreprocessor(policy) for policy in ("preserve","masked")]
        validators=[self.make_validator(p,{"reference_mask_policy":"observed_only"})[0] for p in processors]
        self.assertTrue(torch.equal(validators[0].reference_mask,validators[1].reference_mask))
        self.assertTrue(torch.equal(validators[0].reference,validators[1].reference))
        self.assertEqual([(r["filename"],r["start_sample"]) for r in validators[0].windows_metadata],
                         [(r["filename"],r["start_sample"]) for r in validators[1].windows_metadata])
        for validator in validators:
            self.assertEqual(validator.protocol["reference_mask_policy"],"observed_only")

    def test_filter_happens_before_greedy_nonoverlap_and_count_limit(self):
        processor=MaskedPreprocessor("preserve",windows=1)
        processor.window_epochs,processor.window_size=4,3000
        data=np.zeros(6000)
        data[3000:]=1
        processor.records["v0.npz"]["ibi"]=data
        processor.make_validation_dataset=lambda **_:SimpleNamespace(window_index=[("v0.npz",n) for n in (0,750,1500,2250,3000)])
        options={**coordinator.DEFAULT_VALIDATION,"max_windows":1,"reference_mask_policy":"observed_only"}
        selected,metadata=coordinator.select_fixed_windows(processor,options)
        self.assertEqual(selected,[("v0.npz",750)])
        self.assertEqual(metadata["excluded_insufficient_observation_candidates"],1)
        # Legacy selection is unchanged and retains the first all-zero window.
        legacy,_=coordinator.select_fixed_windows(processor,{**coordinator.DEFAULT_VALIDATION,"max_windows":1})
        self.assertEqual(legacy,[("v0.npz",0)])

    def test_generation_uses_fixed_mask_only_for_scoring_and_restores_training_state(self):
        validator,training=self.make_validator(MaskedPreprocessor())
        masks=validator.reference_mask.clone()
        state_before={k:v.clone() for k,v in training.state_dict().items()}
        grads_before=[p.grad.clone() for p in training.parameters()]
        random_before=random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone()
        generated=validator.reference+torch.tensor([[.1],[.2],[.3]],dtype=torch.float64)
        generated[~masks]=-99.
        generated[0,-1]=0.  # Zero counts as nonpositive, just like a negative value.
        sizes=[]
        def sample(model,clean,diffusion_config,args,device):
            self.assertIsNot(model,training)
            self.assertFalse(model.training)
            start=sum(sizes)
            self.assertTrue(torch.equal(clean,validator.windows[start:start+len(clean)]))
            sizes.append(len(clean))
            random.random(); np.random.rand(); torch.rand(1)
            return (generated[start:start+len(clean)]-validator.center)/validator.scale,{"unit_test_mock":True}
        # Mock only model construction and sampling; actual accumulation, RNG
        # restoration, paired masks, report history and early stopping run.
        with mock.patch("training_utils.fetchModel",side_effect=lambda *_:torch.nn.Linear(1,1)), \
             mock.patch.object(coordinator,"sample_validation_windows",side_effect=sample), \
             contextlib.redirect_stdout(io.StringIO()):
            result=validator.run(1,training,.2)
        self.assertEqual(sizes,[2,1])
        self.assertAlmostEqual(result["aggregate"]["gaussian_kl_real_to_generated"],
                               numpy_kl(generated.numpy(),validator.reference.numpy(),masks.numpy()),places=12)
        self.assertEqual(result["diagnostics"],dict(
            generated_samples=2250,generated_nonpositive_samples=2241,
            generated_nonpositive_fraction=2241/2250))
        saved=json.loads((validator.output/"epoch_0001"/"metrics.json").read_text())
        self.assertEqual(saved["diagnostics"],result["diagnostics"])
        self.assertEqual((validator.training_output/"epoch_metrics.csv").read_text().splitlines()[0].split(","),
                         list(coordinator.EPOCH_METRIC_FIELDS))
        self.assertTrue(torch.equal(validator.reference_mask,masks))
        self.assertTrue(training.training)
        self.assertTrue(all(torch.equal(v,training.state_dict()[k]) for k,v in state_before.items()))
        self.assertTrue(all(torch.equal(a,p.grad) for a,p in zip(grads_before,training.parameters())))
        self.assertEqual(random_before[0],random.getstate())
        self.assertTrue(np.array_equal(random_before[1][1],np.random.get_state()[1]))
        self.assertEqual(random_before[1][2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(random_before[2],torch.random.get_rng_state()))

    def test_nonfinite_unscored_generation_still_fails_before_saving_diagnostics(self):
        for invalid in (float("nan"),float("inf"),-float("inf")):
            with self.subTest(invalid=invalid):
                validator,training=self.make_validator(MaskedPreprocessor())
                generated=validator.reference[:2].clone()
                generated[0,-1]=invalid
                with mock.patch("training_utils.fetchModel",side_effect=lambda *_:torch.nn.Linear(1,1)), \
                     mock.patch.object(coordinator,"sample_validation_windows",return_value=(
                         (generated-validator.center)/validator.scale,{"unit_test_mock":True})), \
                     contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):
                    validator.run(1,training,.2)
                self.assertFalse((validator.output/"epoch_0001"/"metrics.json").exists())
                self.assertEqual(validator.reports,[])

    def test_reference_mask_config_is_strict_and_default_payload_unchanged(self):
        self.assertEqual(coordinator.validation_options({}),coordinator.DEFAULT_VALIDATION)
        self.assertEqual(coordinator.validation_options({"validation":{"reference_mask_policy":"training"}}),coordinator.DEFAULT_VALIDATION)
        for policy in (None,True,3,"generated_positive","unknown"):
            with self.subTest(policy=policy),self.assertRaises(ValueError):
                coordinator.validation_options({"validation":{"reference_mask_policy":policy}})


class WeightedLossTests(unittest.TestCase):
    def test_observed_point_weights_override_batch_sizes_without_changing_legacy(self):
        rows=[dict(epsilon_mse=1.,batch_samples=2,loss_weight=100),dict(epsilon_mse=3.,batch_samples=1,loss_weight=300)]
        self.assertEqual(coordinator.weighted_loss(rows),2.5)
        self.assertEqual(coordinator.weighted_loss([{k:v for k,v in row.items() if k!="loss_weight"} for row in rows]),5/3)
    def test_invalid_loss_weights_are_rejected(self):
        for weight in (True,np.bool_(False),0,-1,float("nan"),float("inf"),"2",None,1+2j):
            with self.subTest(weight=weight),self.assertRaises(ValueError):
                coordinator.weighted_loss([dict(epsilon_mse=1.,batch_samples=1,loss_weight=weight)])
        with self.assertRaises(ValueError):
            coordinator.weighted_loss([])


if __name__=="__main__":
    unittest.main()
