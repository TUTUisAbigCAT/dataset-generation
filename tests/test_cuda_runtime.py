"""CUDA guards are tested without a GPU; a real GPU smoke test skips explicitly."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cuda_runtime import require_cuda, assert_cuda_model, cuda_environment


class CudaPolicyTests(unittest.TestCase):
    def test_no_cuda_fails_without_allocating_or_falling_back(self):
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.empty") as allocate:
            with self.assertRaisesRegex(RuntimeError, "CUDA is required.*CPU fallback is disabled"):
                require_cuda()
            allocate.assert_not_called()

    def test_cuda_initialization_error_is_fatal(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.empty", side_effect=RuntimeError("driver unavailable")):
            with self.assertRaisesRegex(RuntimeError, "device initialization failed") as result:
                require_cuda()
            self.assertIn("driver unavailable", str(result.exception.__cause__))

    def test_assigned_device_is_used_for_cuda_allocation(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=2), \
             mock.patch("torch.empty") as allocate:
            selected = require_cuda()
            self.assertEqual(selected, torch.device("cuda:2"))
            allocate.assert_called_once_with(0, device=torch.device("cuda:2"))

    def test_host_parameters_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "parameter weight: cpu"):
            assert_cuda_model(torch.nn.Linear(2, 1))

    def test_host_buffers_are_rejected_even_without_parameters(self):
        model = torch.nn.Module()
        model.register_buffer("running", torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "buffer running: cpu"):
            assert_cuda_model(model)

    def test_cuda_parameters_and_buffers_are_accepted(self):
        tensor = SimpleNamespace(device=torch.device("cuda:0"))
        model = SimpleNamespace(named_parameters=lambda: [("weight", tensor)],
                                named_buffers=lambda: [("running", tensor)])
        assert_cuda_model(model)

    def test_environment_rejects_host_device_before_gpu_query(self):
        with mock.patch("torch.cuda.get_device_name") as name:
            with self.assertRaisesRegex(RuntimeError, "non-CUDA"):
                cuda_environment(torch.device("cpu"))
            name.assert_not_called()

    def test_environment_records_selected_device_without_changing_runtime(self):
        with mock.patch("torch.cuda.get_device_name", return_value="test GPU") as name, \
             mock.patch("torch.backends.cudnn.version", return_value=90100):
            record = cuda_environment(torch.device("cuda:3"))
        name.assert_called_once_with(torch.device("cuda:3"))
        self.assertEqual(record["device"], "cuda:3")
        self.assertEqual(record["device_name"], "test GPU")
        self.assertEqual(record["torch_version"], torch.__version__)
        self.assertEqual(record["cuda_runtime_version"], torch.version.cuda)
        self.assertFalse(record["cpu_fallback_enabled"])


@unittest.skipUnless(torch.cuda.is_available(), "Real CUDA runtime unavailable; no GPU execution was performed")
class RealCudaSmokeTests(unittest.TestCase):
    def test_baseline_forward_backward_and_adam_update_remain_on_cuda(self):
        from training_utils import fetchModel, fetchDiffusionConfig
        device = require_cuda()
        args = SimpleNamespace(backbone="S4", res_channels=8, skip_channels=8, num_res_layers=1,
                               diff_step_embed_in=8, diff_step_embed_mid=16, diff_step_embed_out=16,
                               s4_lmax=32, s4_dstate=8, s4_dropout=0., s4_bidirectional=True,
                               s4_layernorm=True, beta_0=.0001, beta_T=.02, timesteps=4)
        model = fetchModel(10, 1, args).to(device)
        assert_cuda_model(model)
        config = {key: value.to(device) if torch.is_tensor(value) else value
                  for key, value in fetchDiffusionConfig(args).items()}
        self.assertTrue(all(value.device == device for value in config.values() if torch.is_tensor(value)))
        batch = torch.randn(2, 32, 10, device=device)
        times = torch.randint(config["T"], (2,), device=device)
        noise = torch.randn(batch.shape, device=device)
        alpha = config["alpha_bars"][times].reshape(2, 1, 1)
        mask = torch.ones(batch.shape, device=device)
        mask[:, :, 0] = 0
        noised = (1 - mask) * (alpha.sqrt() * batch + (1 - alpha).sqrt() * noise) + mask * batch
        before = [value.detach().clone() for value in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with torch.device(device):
            prediction = model(noised, times[:, None])
        self.assertEqual(prediction.device, device)
        loss = torch.nn.functional.mse_loss(prediction, noise[:, :, :1].permute(0, 2, 1))
        self.assertEqual(loss.device, device)
        self.assertTrue(torch.isfinite(loss).item())
        optimizer.zero_grad()
        loss.backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(value.device == device and torch.isfinite(value).all().item() for value in gradients))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        assert_cuda_model(model)
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    unittest.main()
