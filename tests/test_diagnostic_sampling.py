"""Small sensitive denoisers test diagnostic isolation and common random numbers."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnostic_sampling as diagnostics
from diffusion_schedule import build_diffusion_config


class SensitiveDenoiser(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1, device=device))
        self.register_buffer("weights", torch.arange(1, 10, dtype=torch.float32, device=device)/20)

    def forward(self, x, times):
        return (self.scale*x[:, :, 0]+(x[:, :, 1:]*self.weights).sum(-1)).unsqueeze(1)


def fixture(device="cpu", *, length=8, total=18):
    device = torch.device(device)
    args = SimpleNamespace(window_size=length, s4_lmax=length, stride=2, timesteps=3,
                           beta_0=.001, beta_T=.02, diffusion_schedule="linear",
                           terminal_alpha_bar=1e-8)
    schedule = {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in build_diffusion_config(args).items()}
    conditions = torch.zeros(total, 9, dtype=torch.float32, device=device)
    stages = (torch.arange(total, device=device)//max(1,length//2))%4
    conditions[torch.arange(total, device=device), stages] = 1
    conditions[:, 7] = 1
    reference = .8+.04*torch.sin(torch.arange(total, dtype=torch.float64, device=device))
    scaler = dict(signal_mean_seconds=.9, signal_std_seconds=.3,
                  continuous_mean=[10.,25.,40.], continuous_std=[5.,4.,10.])
    return SensitiveDenoiser(device).eval(), conditions, reference, scaler, args, schedule, device


class DiagnosticSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def run_fixture(self, items, **kwargs):
        options = dict(starts=[0], repeats=3, comparison_repeats=2, seed=18, stride=2, windows_per_group=2)
        options.update(kwargs)
        return diagnostics._run_sampling_diagnostics(*items, **options)

    def test_full_protocol_sensitive_conditions_crn_and_model_rng_isolation(self):
        items = fixture()
        model, conditions, reference, *_ = items
        state_before = {key: value.detach().clone() for key,value in model.state_dict().items()}
        condition_before, reference_before = conditions.clone(), reference.clone()
        torch.manual_seed(7151)
        rng_before = torch.get_rng_state().clone()
        progress = []
        report, arrays = self.run_fixture(items, progress=progress.append)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(conditions, condition_before))
        self.assertTrue(torch.equal(reference, reference_before))
        self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in state_before.items()))
        self.assertTrue(report["model_state"]["unchanged"])
        self.assertEqual(report["model_state"]["before_sha256"],report["model_state"]["after_sha256"])
        self.assertEqual(arrays["repeat_generated_seconds"].shape,(1,3,8))
        self.assertEqual(arrays["condition_generated_seconds"].shape,(1,12,8))
        self.assertEqual(arrays["independent_night_seconds"].shape,(2,18))
        self.assertEqual(arrays["continued_night_seconds"].shape,(2,18))
        self.assertEqual(arrays["repeat_generated_seconds"].dtype,np.float64)
        self.assertTrue(np.array_equal(arrays["repeat_generated_seconds"][0,0],arrays["condition_generated_seconds"][0,0]))
        same = report["stages"]["same_conditions"]
        self.assertGreater(same["between_draw_rows"][0]["std_of_draw_means_ms"],0)
        changes = report["stages"]["changed_conditions"]
        self.assertTrue(changes["determinism_rows"][0]["baseline_repeat_bitwise_equal"])
        self.assertTrue(all(row["end_rng_matches_baseline"] for row in changes["rows"]))
        for name in ("stage_REM","AHI_plus1std","gender_swap"):
            row = next(row for row in changes["rows"] if row["variant"]==name)
            self.assertGreater(row["generation_rmse_vs_baseline_ms"],1e-4)
            response = [row for row in changes["epsilon_rows"] if row["variant"]==name]
            self.assertTrue(all(row["epsilon_rmse_vs_baseline"]>0 for row in response))
        summaries = [event for event in progress if event["event"]=="stage_summary"]
        self.assertEqual([event["stage"] for event in summaries],list(report["stages"]))
        # Tail window overlaps only for complete coverage, and only its new
        # suffix is emitted. Both branches retain the same N and eval windows.
        coverage = report["stages"]["independent_vs_continued"]["coverage"]
        self.assertEqual(coverage["independent_window_starts"],[0,8,10])
        self.assertEqual(coverage["final_independent_window_overlap_samples"],6)
        self.assertTrue(coverage["identical_evaluation_windows"])

    def test_reference_is_posthoc_only_and_metrics_match_hand_formula(self):
        items = fixture()
        report, arrays = self.run_fixture(items, repeats=1, comparison_repeats=1, kl_variance_floor_ms2=4.)
        changed = list(items)
        changed[2] = items[2]+3
        report_changed, arrays_changed = self.run_fixture(changed, repeats=1, comparison_repeats=1, kl_variance_floor_ms2=4.)
        for key in ("repeat_generated_seconds","condition_generated_seconds","independent_night_seconds","continued_night_seconds"):
            self.assertTrue(np.array_equal(arrays[key],arrays_changed[key],equal_nan=True),key)
        row = report["stages"]["same_conditions"]["rows"][0]
        generated = arrays["repeat_generated_seconds"][0,0]*1000
        reference = arrays["reference_seconds"][:8]*1000
        vg,vr = max(float(generated.var()),4.),max(float(reference.var()),4.)
        expected = .5*(math.log(vg/vr)+(vr+(reference.mean()-generated.mean())**2)/vg-1)
        self.assertAlmostEqual(row["gaussian_kl_real_to_generated"],expected,places=11)
        self.assertNotEqual(row["reference_mean_ms"],report_changed["stages"]["same_conditions"]["rows"][0]["reference_mean_ms"])

    def test_invalid_counterfactuals_are_skipped_with_explicit_nan_slots(self):
        items = list(fixture())
        items[3]["continuous_mean"] = [.2,.5,5.]
        report, arrays = self.run_fixture(items,repeats=1,comparison_repeats=1)
        rows = report["stages"]["changed_conditions"]["rows"]
        for name in ("AHI_minus1std","BMI_minus1std","AGE_minus1std"):
            index = diagnostics.VARIANTS.index(name)
            row = next(row for row in rows if row["variant"]==name)
            self.assertEqual(row["status"],"skipped")
            self.assertIn("no clipping",row["reason"])
            self.assertFalse(arrays["condition_variant_valid"][0,index])
            self.assertTrue(np.isnan(arrays["condition_generated_seconds"][0,index]).all())

    def test_public_entry_never_falls_back_to_cpu(self):
        with self.assertRaisesRegex(RuntimeError,"CUDA"):
            diagnostics.run_sampling_diagnostics(*fixture(),starts=[0])

    def test_negative_samples_are_retained_without_clipping(self):
        items = list(fixture())
        items[3]["signal_mean_seconds"] = -.5
        report, arrays = self.run_fixture(items,repeats=1,comparison_repeats=1)
        values = arrays["repeat_generated_seconds"][0,0]
        self.assertTrue(bool((values<0).any()))
        self.assertEqual(report["stages"]["same_conditions"]["rows"][0]["generated_negative_fraction"],float((values<0).mean()))

    def test_actual_small_s4_runs_all_stages_without_state_mutation(self):
        from training_utils import fetchModel
        items = list(fixture())
        args = items[4]
        for key,value in dict(backbone="S4",res_channels=4,skip_channels=4,num_res_layers=1,
                              diff_step_embed_in=4,diff_step_embed_mid=8,diff_step_embed_out=8,
                              s4_dstate=4,s4_dropout=0.,s4_bidirectional=True,s4_layernorm=True).items():
            setattr(args,key,value)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            items[0] = fetchModel(10,1,args).eval()
        report, arrays = self.run_fixture(items,repeats=1,comparison_repeats=1)
        self.assertTrue(report["model_state"]["unchanged"])
        self.assertEqual(len(report["stages"]),3)
        self.assertTrue(report["stages"]["changed_conditions"]["determinism_rows"][0]["baseline_repeat_bitwise_equal"])
        self.assertTrue(np.isfinite(arrays["continued_night_seconds"]).all())

    def test_rejects_malformed_inputs_before_sampling(self):
        for options in (dict(repeats=0),dict(starts=[11]),dict(starts=[0,0]),
                        dict(kl_variance_floor_ms2=float("nan")),dict(seed=True)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.run_fixture(fixture(),**options)
        items = list(fixture())
        items[1][:,:4] = .25
        with self.assertRaisesRegex(ValueError,"one-hot"):
            self.run_fixture(items)

    def test_detects_registered_model_state_mutation(self):
        class Mutating(SensitiveDenoiser):
            def forward(self,x,t):
                self.weights.add_(.001)
                return super().forward(x,t)
        items = list(fixture())
        items[0] = Mutating().eval()
        with self.assertRaisesRegex(RuntimeError,"Model parameters or buffers changed"):
            self.run_fixture(items,repeats=1,comparison_repeats=1)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA hardware unavailable; actual diagnostic GPU test skipped")
    def test_public_cuda_protocol_restores_selected_rng(self):
        device = torch.device("cuda",torch.cuda.current_device())
        items = fixture(device,length=750,total=1500)
        rng_before = torch.cuda.get_rng_state(device).clone()
        report,arrays = diagnostics.run_sampling_diagnostics(*items,starts=[0],repeats=1,
            comparison_repeats=1,stride=750,windows_per_group=2)
        self.assertTrue(torch.equal(rng_before,torch.cuda.get_rng_state(device)))
        self.assertEqual(arrays["continued_night_seconds"].shape,(1,1500))
        self.assertTrue(report["model_state"]["unchanged"])


if __name__ == "__main__":
    unittest.main()
