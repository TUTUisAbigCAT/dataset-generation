"""Observed-sample reduction and gradient checks; no model training."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from missing_data import masked_noise_mse


class MaskedDiffusionLossTests(unittest.TestCase):
    def example(self, device="cpu", dtype=torch.float64):
        predicted = torch.tensor([[[1., -2., 100.]], [[-3., 200., 300.]]],
            device=device, dtype=dtype, requires_grad=True)
        target = torch.zeros_like(predicted)
        observed = torch.tensor([[True, True, False], [True, False, False]], device=device)
        return predicted, target, observed

    def test_observed_sample_reduction_not_average_of_window_means(self):
        pred, target, mask = self.example()
        loss, squared_sum, count = masked_noise_mse(pred, target, mask)
        self.assertEqual(count, 3)
        self.assertIs(type(count), int)
        self.assertEqual(squared_sum.item(), 14.)
        self.assertEqual(loss.item(), 14/3)
        self.assertFalse(squared_sum.requires_grad)
        self.assertIsNone(squared_sum.grad_fn)
        self.assertEqual(loss.device, pred.device)
        self.assertEqual(squared_sum.device, pred.device)

    def test_backward_is_zero_at_unobserved_positions(self):
        pred, target, mask = self.example()
        loss, _, _ = masked_noise_mse(pred, target, mask)
        loss.backward()
        expected = torch.tensor([[[2/3, -4/3, 0.]], [[-2., 0., 0.]]], dtype=pred.dtype)
        torch.testing.assert_close(pred.grad, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(pred.grad[:, 0, :][~mask], torch.zeros(3, dtype=pred.dtype)))

    def test_all_observed_matches_ordinary_mse(self):
        pred = torch.arange(10, dtype=torch.float64).reshape(2, 1, 5)/3
        target = pred.flip(-1)
        loss, squared_sum, count = masked_noise_mse(pred, target, torch.ones(2, 5, dtype=torch.bool))
        self.assertEqual(count, 10)
        torch.testing.assert_close(loss, torch.nn.functional.mse_loss(pred, target), rtol=0, atol=1e-15)
        torch.testing.assert_close(squared_sum, loss*count, rtol=0, atol=1e-15)

    def test_unobserved_extreme_finite_values_never_enter_arithmetic(self):
        maximum = torch.finfo(torch.float32).max
        pred = torch.tensor([[[1., maximum]]], requires_grad=True)
        target = torch.tensor([[[0., -maximum]]])
        loss, squared_sum, count = masked_noise_mse(pred, target, torch.tensor([[True, False]]))
        self.assertEqual((loss.item(), squared_sum.item(), count), (1., 1., 1))
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.tensor([[[2., 0.]]])))

    def test_half_precision_accumulation_does_not_overflow_valid_count(self):
        pred = torch.ones(1, 1, 65536, dtype=torch.float16)
        target = torch.zeros_like(pred)
        loss, squared_sum, count = masked_noise_mse(pred, target, torch.ones(1, 65536, dtype=torch.bool))
        self.assertEqual(count, 65536)
        self.assertEqual(squared_sum.dtype, torch.float32)
        self.assertEqual(squared_sum.item(), 65536.)
        self.assertEqual(loss.item(), 1.)

    def test_nonfinite_values_are_rejected_even_where_mask_is_false(self):
        for field in ("predicted", "target"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    pred, target, mask = self.example()
                    pred = pred.detach().clone()
                    (pred if field == "predicted" else target)[0, 0, 2] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        masked_noise_mse(pred, target, mask)

    def test_all_missing_and_empty_inputs_are_rejected(self):
        pred, target, mask = self.example()
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_noise_mse(pred, target, torch.zeros_like(mask))
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_noise_mse(torch.empty(1, 1, 0), torch.empty(1, 1, 0), torch.empty(1, 0, dtype=torch.bool))

    def test_nonboolean_and_negative_masks_are_rejected(self):
        pred, target, mask = self.example()
        for bad in (mask.float(), mask.to(torch.uint8), mask.to(torch.int64)-1):
            with self.subTest(dtype=bad.dtype):
                with self.assertRaisesRegex(TypeError, "Boolean"):
                    masked_noise_mse(pred, target, bad)

    def test_shape_dtype_and_tensor_contracts_are_strict(self):
        pred, target, mask = self.example()
        cases = [
            (pred[:, 0, :], target[:, 0, :], mask, ValueError),
            (pred.expand(-1, 2, -1), target.expand(-1, 2, -1), mask, ValueError),
            (pred, target[:1], mask, ValueError),
            (pred, target, mask[:, None, :], ValueError),
            (pred, target, mask[:1], ValueError),
            (pred, target.float(), mask, TypeError),
            (pred.long(), target.long(), mask, TypeError),
            (pred.to(torch.complex128), target.to(torch.complex128), mask, TypeError),
            (pred.tolist(), target, mask, TypeError),
            (pred, target, mask.tolist(), TypeError),
        ]
        for args in cases:
            with self.subTest(error=args[-1], shapes=[getattr(v, "shape", None) for v in args[:3]]):
                with self.assertRaises(args[-1]):
                    masked_noise_mse(*args[:3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware/runtime unavailable; no CPU substitution")
    def test_cuda_forward_backward_remains_on_cuda(self):
        pred, target, mask = self.example("cuda", torch.float32)
        loss, squared_sum, count = masked_noise_mse(pred, target, mask)
        self.assertEqual(count, 3)
        self.assertEqual(loss.device.type, "cuda")
        self.assertEqual(squared_sum.device.type, "cuda")
        loss.backward()
        self.assertEqual(pred.grad.device.type, "cuda")
        expected = torch.tensor([[[2/3, -4/3, 0.]], [[-2., 0., 0.]]], device=pred.device)
        torch.testing.assert_close(pred.grad, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware/runtime unavailable; no CPU substitution")
    def test_cross_device_inputs_are_rejected_without_automatic_transfer(self):
        pred, target, mask = self.example("cuda", torch.float32)
        with self.assertRaisesRegex(ValueError, "same device"):
            masked_noise_mse(pred, target.cpu(), mask)
        with self.assertRaisesRegex(ValueError, "same device"):
            masked_noise_mse(pred, target, mask.cpu())


if __name__ == "__main__":
    unittest.main()
