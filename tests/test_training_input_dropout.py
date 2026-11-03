"""CPU numerical/unit tests plus explicitly skipped real CUDA counterparts.

Actual training-loop AST is executed on tiny CPU models. This is not a claim
of production CUDA training or improved generation quality.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_input_dropout import input_dropout_options, create_training_input_dropout, TrainingInputDropout
from scripts import run_dreamt
from scripts.verify_upstream_boundary import (
    expected_training_source, remove_training_input_dropout_edits, TRAINING_INPUT_DROPOUT_EDITS)


def training_loop(source):
    tree = ast.parse(source)
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    return compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                   "actual_training_loop::CPU_unit_test", "exec")


class InputDropoutTests(unittest.TestCase):
    def test_configuration_is_explicit_strict_and_defaults_to_disabled(self):
        self.assertEqual(input_dropout_options({}), dict(probability=0., seed=42))
        self.assertEqual(input_dropout_options({"training_input_dropout": {"probability": .1}}), dict(probability=.1, seed=42))
        self.assertEqual(input_dropout_options({"training_input_dropout": {"probability": 1, "seed": 2**63-1}}),
                         dict(probability=1., seed=2**63-1))
        for invalid in (None, [], False, {"probability": -.1}, {"probability": 1.01},
                        {"probability": float("nan")}, {"probability": float("inf")},
                        {"probability": True}, {"probability": "0.1"}, {"seed": True},
                        {"seed": 1.5}, {"seed": -1}, {"seed": 2**63}, {"p": .1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                input_dropout_options({"training_input_dropout": invalid})

    def test_probability_zero_is_same_tensor_no_random_generator_or_counter_allocation(self):
        value = torch.randn(2, 20, 10)
        before = torch.random.get_rng_state().clone()
        with mock.patch.object(torch, "Generator", side_effect=AssertionError("p0 must not create a generator")), \
             mock.patch.object(torch, "rand", side_effect=AssertionError("p0 must not draw")), \
             mock.patch.object(torch, "zeros", side_effect=AssertionError("p0 must not allocate counters")):
            adapter = TrainingInputDropout(dict(probability=0., seed=42), "cuda:0")
            self.assertIs(adapter(value), value)
            report = adapter.report()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIsNone(adapter._generator)
        self.assertFalse(report["active"])
        self.assertEqual([report[k] for k in ("application_calls", "eligible_values", "dropped_values", "observed_fraction")], [0, 0, 0, 0.])

    def test_all_drop_erases_only_ibi_and_keeps_original_input_and_target(self):
        value = torch.arange(2*20*10, dtype=torch.float32).reshape(2,20,10)+1
        target = torch.randn(2,1,20)
        before, target_before = value.clone(), target.clone()
        adapter = TrainingInputDropout(dict(probability=1., seed=42), "cpu")
        actual = adapter(value)
        self.assertTrue(torch.all(actual[:, :, 0] == 0))
        self.assertTrue(torch.equal(actual[:, :, 1:], before[:, :, 1:]))
        self.assertTrue(torch.equal(value, before))
        self.assertTrue(torch.equal(target, target_before))
        self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        report = adapter.report()
        self.assertEqual((report["application_calls"], report["eligible_values"], report["dropped_values"]), (1,40,40))
        self.assertEqual(report["observed_fraction"], 1.)

    def check_random_stream(self, device):
        values = torch.full((3, 2000, 10), 2., device=device)
        global_cpu = torch.random.get_rng_state().clone()
        global_device = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        first = TrainingInputDropout(dict(probability=.1, seed=42), device)
        second = TrainingInputDropout(dict(probability=.1, seed=42), device)
        result1 = first(values)
        result2 = first(values)
        self.assertTrue(torch.equal(result1, second(values)))
        self.assertTrue(torch.equal(result2, second(values)))
        self.assertFalse(torch.equal(result1[:, :, 0], result2[:, :, 0]))
        self.assertTrue(torch.equal(global_cpu, torch.random.get_rng_state()))
        if global_device is not None:
            self.assertTrue(torch.equal(global_device, torch.cuda.get_rng_state(device)))
        self.assertTrue(torch.equal(values, torch.full_like(values, 2.)))
        self.assertTrue(torch.equal(result1[:, :, 1:], values[:, :, 1:]))
        self.assertTrue(torch.all((result1[:, :, 0] == 0) | (result1[:, :, 0] == 2)))
        report = first.report()
        self.assertEqual(report["eligible_values"], 12000)
        self.assertEqual(report["dropped_values"], int((result1[:, :, 0] == 0).sum()+(result2[:, :, 0] == 0).sum()))
        self.assertGreater(report["observed_fraction"], .075)
        self.assertLess(report["observed_fraction"], .125)

    def test_dedicated_cpu_stream_reproducible_without_global_rng_consumption(self):
        self.check_random_stream(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA input-dropout test needs a GPU")
    def test_dedicated_cuda_stream_and_condition_integrity(self):
        self.check_random_stream(torch.device("cuda:0"))

    def test_device_counts_are_int64_beyond_uint32(self):
        adapter = TrainingInputDropout(dict(probability=1.), "cpu")
        self.assertEqual(adapter._eligible_values.dtype, torch.int64)
        self.assertEqual(adapter._dropped_values.dtype, torch.int64)
        adapter._eligible_values.fill_(2**32-3)
        adapter._dropped_values.fill_(2**32-3)
        adapter(torch.ones(1,10,10))
        report = adapter.report()
        self.assertEqual(report["eligible_values"], 2**32+7)
        self.assertEqual(report["dropped_values"], 2**32+7)

    def test_wrong_shape_or_device_does_not_fall_back(self):
        adapter = TrainingInputDropout(dict(probability=.1), "cpu")
        for value in (torch.ones(3,10), torch.ones(0,10,10), torch.ones(1,5,10,dtype=torch.int64)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                adapter(value)
        with self.assertRaisesRegex(ValueError, "input device"):
            adapter(torch.ones(1,3,10,device="meta"))

    def test_factory_reads_only_the_selected_dataset_configuration(self):
        temp_root = ROOT/"work/input_dropout_tests"
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp)/"config.json"
            path.write_text(json.dumps(dict(training_input_dropout=dict(probability=.1,seed=9))),encoding="utf-8")
            with mock.patch.dict(os.environ,{"DREAMT_CONFIG_PATH":str(path)}):
                adapter = create_training_input_dropout("cpu")
            self.assertEqual(adapter.options,dict(probability=.1,seed=9))
            with mock.patch.dict(os.environ,{"DREAMT_CONFIG_PATH":str(path.with_name("absent.json"))}), self.assertRaises(FileNotFoundError):
                create_training_input_dropout("cpu")

    def run_actual_loop(self, source, probability):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.7))
                self.bias = torch.nn.Parameter(torch.tensor(.1))
                self.inputs = []
            def forward(self, values, timesteps):
                self.inputs.append(values.detach().clone())
                return (values[:, :, 0]*self.weight+self.bias).unsqueeze(1)
        with torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42)
            model = Model()
            optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
            dataset = torch.arange(5*12*10,dtype=torch.float32).reshape(5,12,10)/100+1
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
            adapter = TrainingInputDropout(dict(probability=probability,seed=42),"cpu")
            noise, times, losses, targets, gradient_rows = [], [], [], [], []
            def normal(*args, **kwargs):
                result = torch.normal(*args, **kwargs);noise.append(result.clone());return result
            def randint(*args, **kwargs):
                result = torch.randint(*args, **kwargs);times.append(result.clone());return result
            def criterion(prediction, target):
                prediction.retain_grad()
                loss = torch.nn.functional.mse_loss(prediction,target)
                losses.append(float(loss.detach()));targets.append(target.clone());gradient_rows.append(prediction)
                return loss
            namespace=dict(torch=torch,np=np,model=model,optimizer=optimizer,criterion=criterion,
                device=torch.device("cpu"),args=SimpleNamespace(epochs=2),dataloader=dataloader,
                diffusion_config={"T":3,"alpha_bars":torch.tensor([[.9],[.5],[.1]])},
                non_hier_cols=torch.tensor([0]),randint=randint,normal=normal,sqrt=torch.sqrt,
                training_input_dropout=adapter,validator=SimpleNamespace(run=lambda *args:None,should_stop=False))
            exec(training_loop(source),namespace)
            return dict(parameters={key:value.detach().clone() for key,value in model.state_dict().items()},
                noise=noise,times=times,inputs=model.inputs,losses=losses,targets=targets,
                predictions=gradient_rows,rng=torch.random.get_rng_state().clone(),dropout=adapter.report())

    def test_inactive_actual_training_loop_is_bitwise_equal_to_prior_loop(self):
        current=(ROOT/"training_wavestitch.py").read_bytes()
        before=self.run_actual_loop(remove_training_input_dropout_edits(current),0.)
        after=self.run_actual_loop(current,0.)
        self.assertEqual(before["losses"],after["losses"])
        self.assertTrue(torch.equal(before["rng"],after["rng"]))
        for key in before["parameters"]:
            self.assertTrue(torch.equal(before["parameters"][key],after["parameters"][key]))
        for key in ("noise","times","inputs","targets"):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[key],after[key])))
        self.assertEqual(after["dropout"]["application_calls"],0)

    def test_active_actual_loop_keeps_noise_timesteps_shuffle_targets_and_full_mse(self):
        current=(ROOT/"training_wavestitch.py").read_bytes()
        before=self.run_actual_loop(current,0.)
        after=self.run_actual_loop(current,1.)
        self.assertTrue(torch.equal(before["rng"],after["rng"]))
        for key in ("noise","times","targets"):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[key],after[key])))
        for a,b in zip(before["inputs"],after["inputs"]):
            self.assertTrue(torch.equal(a[:,:,1:],b[:,:,1:]))
            self.assertTrue(torch.all(b[:,:,0] == 0))
        for prediction,target,loss in zip(after["predictions"],after["targets"],after["losses"]):
            self.assertEqual(loss,float(torch.nn.functional.mse_loss(prediction.detach(),target)))
            torch.testing.assert_close(prediction.grad,2*(prediction.detach()-target)/target.numel())
            self.assertTrue(torch.any(prediction.grad != 0))
        report=after["dropout"]
        self.assertEqual((report["application_calls"],report["eligible_values"],report["dropped_values"]),(6,120,120))

    def test_exact_source_reversal_accepts_only_old_or_three_known_additions(self):
        actual=(ROOT/"training_wavestitch.py").read_bytes()
        original=(ROOT/"third_party/WaveStitch/training_wavestitch.py").read_bytes()
        prior=expected_training_source(original,include_input_dropout=False)
        self.assertEqual(remove_training_input_dropout_edits(actual),prior)
        self.assertIs(remove_training_input_dropout_edits(prior),prior)
        self.assertEqual(expected_training_source(original),actual)
        bad_sources=[actual.replace(b"nn.MSELoss()",b"nn.L1Loss()"),prior+b"\n# unrelated\n",
                     actual.replace(b"batch_noised = training_input_dropout(batch_noised)",b"batch_noised = batch_noised * 0.9")]
        for before,after in TRAINING_INPUT_DROPOUT_EDITS:
            bad_sources.append(actual.replace(after.encode(),before.encode(),1))
        for source in bad_sources:
            with self.subTest(source_sha=hash(source)),self.assertRaises(ValueError):
                remove_training_input_dropout_edits(source)

    def test_two_real_s4_updates_for_both_clean_input_ranges(self):
        # Real preprocessing + S4 + actual optimizer loop, reduced model on CPU.
        # This is an implementation smoke test, not a production training run.
        from data_utils import Preprocessor
        from diffusion_schedule import build_diffusion_config
        from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
        from training_utils import fetchModel
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for output_range, probability in (([0,1],0.),([-1,1],.1)):
                with self.subTest(output_range=output_range,probability=probability), \
                     torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
                    processor = Preprocessor("DREAMT",data_config=dict(data_dir=str(ROOT/"example_data"),
                        window_epochs=1,missing_value_policy="preserve",signal_normalization=dict(
                            method="minmax",min_seconds=0,max_seconds=2,output_range=output_range)))
                    dataset = processor.make_training_dataset(750,750)
                    batches = [torch.stack([dataset[0],dataset[1]]),dataset[2].unsqueeze(0)]
                    self.assertTrue(all(torch.all((batch[:,:,0] >= output_range[0]) &
                                                 (batch[:,:,0] <= output_range[1])) for batch in batches))
                    args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS,window_size=750,s4_lmax=750,
                        epochs=1,res_channels=8,skip_channels=8,num_res_layers=1,s4_dstate=8,
                        diff_step_embed_in=8,diff_step_embed_mid=8,diff_step_embed_out=8,
                        timesteps=4,diffusion_schedule="linear_power",terminal_alpha_bar=1e-8))
                    torch.manual_seed(42)
                    model = fetchModel(10,1,args)
                    before = {name:value.detach().clone() for name,value in model.named_parameters()}
                    inputs = []
                    hook = model.register_forward_pre_hook(lambda module, values: inputs.append(values[0].detach().clone()))
                    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
                    adapter = TrainingInputDropout(dict(probability=probability,seed=42),"cpu")
                    losses = []
                    def criterion(predicted,target):
                        value = torch.nn.functional.mse_loss(predicted,target)
                        losses.append(float(value.detach()))
                        return value
                    scope = dict(torch=torch,np=np,model=model,optimizer=optimizer,criterion=criterion,
                        device=torch.device("cpu"),args=args,dataloader=batches,
                        diffusion_config=build_diffusion_config(args),non_hier_cols=torch.tensor([0]),
                        randint=torch.randint,normal=torch.normal,sqrt=torch.sqrt,training_input_dropout=adapter,
                        validator=SimpleNamespace(run=lambda *args:None,should_stop=False))
                    exec(training_loop((ROOT/"training_wavestitch.py").read_bytes()),scope)
                    hook.remove()
                    self.assertEqual(len(losses),2)
                    self.assertTrue(np.isfinite(losses).all())
                    self.assertTrue(all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
                                        for p in model.parameters()))
                    self.assertTrue(any(not torch.equal(before[name],p) for name,p in model.named_parameters()))
                    for original,noisy in zip(batches,inputs):
                        self.assertTrue(torch.equal(original[:,:,1:],noisy[:,:,1:]))
                    report = adapter.report()
                    if probability:
                        self.assertEqual((report["application_calls"],report["eligible_values"]),(2,2250))
                        self.assertGreater(report["dropped_values"],0)
                        self.assertLess(report["dropped_values"],2250)
                    else:
                        self.assertEqual(report["application_calls"],0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_launcher_freezes_policy_and_helper_without_gpu_execution(self):
        temp_root=ROOT/"work/input_dropout_tests";temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            folder=Path(tmp);config=folder/"config.json";output=folder/"run"
            config.write_text(json.dumps(dict(data_dir=str(ROOT/"example_data"),
                training_input_dropout=dict(probability=.1,seed=42))),encoding="utf-8")
            with mock.patch.object(sys,"argv",["run_dreamt.py","--config",str(config),"--output-dir",str(output),"--skip-generation"]), \
                 mock.patch.dict(os.environ,{}), mock.patch.object(run_dreamt,"require_cuda",return_value=torch.device("cuda:0")), \
                 mock.patch.object(run_dreamt,"cuda_environment",return_value={"mocked_unit_boundary":True}), \
                 mock.patch.object(run_dreamt.subprocess,"run",return_value=subprocess.CompletedProcess([],0)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_dreamt.main()
            preflight=json.loads((output/"preflight.json").read_text(encoding="utf-8"))
            frozen=json.loads((output/"workspace/dataset_config.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["training_input_dropout"],dict(probability=.1,seed=42))
            self.assertEqual(frozen["training_input_dropout"],preflight["training_input_dropout"])
            self.assertEqual((output/"workspace/training_input_dropout.py").read_bytes(),(ROOT/"training_input_dropout.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
