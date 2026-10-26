"""Actual S4 complex-state, recurrence and backward checks; no optimization."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TSImputers.S4Model import HippoSSKernel, S4


class S4ComplexStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_parameter_gradients(self, model, device):
        count = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(parameter.grad.device, device, name)
                self.assertEqual(parameter.grad.dtype, parameter.dtype, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                count += 1
        self.assertGreater(count, 0)

    def check_complex_state_and_layer(self, device):
        features, states, length = 2, 8, 64
        for precision, complex_dtype, real_dtype in (
            (1, torch.complex64, torch.float32),
            (2, torch.complex128, torch.float64),
        ):
            with self.subTest(device=str(device), precision=precision):
                torch.manual_seed(2631)
                kernel = HippoSSKernel(
                    H=features, N=states, L=length, channels=2,
                    precision=precision, trainable=True,
                ).to(device).kernel
                state = torch.randn(
                    2, features, states // 2, dtype=complex_dtype,
                    device=device, requires_grad=True,
                )
                impulse, state_response = kernel(state=state, L=length)
                self.assertEqual(kernel._w().dtype, complex_dtype)
                for result in (impulse, state_response):
                    self.assertEqual(result.device, device)
                    self.assertEqual(result.dtype, real_dtype)
                    self.assertTrue(bool(torch.isfinite(result).all()))
                (impulse.square().mean() + state_response.square().mean()).backward()
                self.assertIsNotNone(state.grad)
                self.assertEqual(state.grad.dtype, complex_dtype)
                self.assertEqual(state.grad.device, device)
                self.assertTrue(bool(torch.isfinite(state.grad).all()))
                self.assertGreater(float(state.grad.norm()), 0)
                self.assert_parameter_gradients(kernel, device)

                # Compare the FFT kernel with the full-state recurrence. This
                # detects a wrong complex resolvent even if gradients are finite.
                with torch.no_grad():
                    impulse, _ = kernel(L=length)
                    kernel.setup_step(mode="dense")
                    recurrent_state = kernel.default_state(1)
                    outputs = []
                    for step in range(length):
                        u = torch.full(
                            (1, features), 1.0 if step == 0 else 0.0,
                            dtype=real_dtype, device=device,
                        )
                        y, recurrent_state = kernel.step(u, recurrent_state)
                        outputs.append(y.squeeze(0))
                    dense = torch.stack(outputs, dim=-1)
                    self.assertEqual(recurrent_state.dtype, complex_dtype)
                    self.assertEqual(recurrent_state.device, device)
                    relative_error = (impulse-dense.real).norm()/dense.real.norm()
                    tolerance = 5e-4 if precision == 1 else 1e-10
                    self.assertLess(float(relative_error), tolerance)
                    self.assertTrue(bool(torch.isfinite(recurrent_state).all()))

                # Exercise the real bidirectional layer and its output mixer.
                # precision=2 controls the internal complex state; the existing
                # Hippo wrapper intentionally exports a float32 kernel.
                torch.manual_seed(2642)
                layer = S4(
                    d_model=features, d_state=states, l_max=length,
                    bidirectional=True, precision=precision, trainable=True,
                ).to(device)
                signal = torch.randn(
                    2, features, length, device=device, requires_grad=True,
                )
                output, _ = layer(signal)
                self.assertEqual(output.shape, signal.shape)
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.device, device)
                self.assertTrue(bool(torch.isfinite(output).all()))
                output.square().mean().backward()
                self.assertTrue(bool(torch.isfinite(signal.grad).all()))
                self.assertGreater(float(signal.grad.norm()), 0)
                self.assert_parameter_gradients(layer, device)

    def test_cpu_complex_state_and_full_layer(self):
        self.check_complex_state_and_layer(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; actual GPU check skipped")
    def test_cuda_complex_state_and_full_layer(self):
        self.check_complex_state_and_layer(torch.device("cuda", torch.cuda.current_device()))


if __name__ == "__main__":
    unittest.main()
