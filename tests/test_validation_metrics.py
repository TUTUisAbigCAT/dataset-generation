"""Independent numerical cases for the physical-unit validation diagnostics."""
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_metrics import preserve_random_state, summarize_distribution


class ValidationMetricsTests(unittest.TestCase):
    def test_identity_and_population_variance(self):
        values = torch.tensor([[0.5, 1.5], [0.5, 1.5]], dtype=torch.float64)
        result = summarize_distribution(values, values)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["generated_mean_ms"], 1000.0)
        self.assertEqual(aggregate["generated_variance_ms2"], 250000.0)
        self.assertEqual(aggregate["mean_absolute_error_ms"], 0.0)
        self.assertEqual(aggregate["variance_absolute_error_ms2"], 0.0)
        self.assertEqual(aggregate["variance_ratio"], 1.0)
        self.assertEqual(len(result["per_window"]), 2)
        json.dumps(result, allow_nan=False)

    def test_shift_changes_mean_without_changing_variance(self):
        reference = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        result = summarize_distribution(reference - 0.25, reference)["aggregate"]
        self.assertEqual(result["generated_mean_ms"], 750.0)
        self.assertEqual(result["mean_error_ms"], -250.0)
        self.assertEqual(result["mean_absolute_error_ms"], 250.0)
        self.assertEqual(result["variance_error_ms2"], 0.0)

    def test_scale_squares_variance(self):
        reference = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
        result = summarize_distribution(reference * 2, reference)["aggregate"]
        self.assertEqual(result["mean_error_ms"], 1000.0)
        self.assertEqual(result["generated_variance_ms2"], 1000000.0)
        self.assertEqual(result["variance_error_ms2"], 750000.0)
        self.assertEqual(result["variance_ratio"], 4.0)

    def test_global_and_per_window_errors_are_distinct(self):
        reference = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        generated = torch.tensor([[0.5, 0.5], [1.5, 1.5]], dtype=torch.float64)
        result = summarize_distribution(generated, reference)
        self.assertEqual(result["aggregate"]["mean_absolute_error_ms"], 0.0)
        self.assertEqual(result["aggregate"]["mean_window_absolute_error_ms"], 500.0)
        self.assertAlmostEqual(result["aggregate"]["variance_absolute_error_ms2"], 250000.0)
        self.assertEqual(result["aggregate"]["variance_window_absolute_error_ms2"], 0.0)
        self.assertIsNone(result["aggregate"]["variance_ratio"])
        self.assertEqual([row["mean_error_ms"] for row in result["per_window"]], [-500.0, 500.0])
        self.assertTrue(all(row["variance_ratio"] is None for row in result["per_window"]))

    def test_zero_and_negative_values_are_not_clipped(self):
        generated = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
        reference = torch.zeros_like(generated)
        result = summarize_distribution(generated, reference)["aggregate"]
        self.assertEqual(result["generated_mean_ms"], -500.0)
        self.assertEqual(result["generated_variance_ms2"], 250000.0)
        self.assertIsNone(result["variance_ratio"])
        json.dumps(result, allow_nan=False)

    def test_rejects_nonfinite_and_overflow(self):
        for value in (float("nan"), float("inf"), float("-inf"), 1e308):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_distribution(torch.tensor([[0.0, value]], dtype=torch.float64),
                                       torch.zeros((1, 2), dtype=torch.float64))
        with self.assertRaises(FloatingPointError):
            summarize_distribution(torch.tensor([[-1e200, 1e200]], dtype=torch.float64),
                                   torch.zeros((1, 2), dtype=torch.float64))

    def test_rejects_shape_and_device_mismatch(self):
        for shape in ((2,), (0, 2), (1, 1), (1, 2, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                summarize_distribution(torch.zeros(shape), torch.zeros(shape))
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            summarize_distribution(torch.zeros((1, 2)), torch.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "same device"):
            summarize_distribution(torch.zeros((1, 2)), torch.empty((1, 2), device="meta"))
        with self.assertRaises(TypeError):
            summarize_distribution([[0.0, 1.0]], torch.zeros((1, 2)))
        with self.assertRaises(TypeError):
            summarize_distribution(torch.zeros((1, 2), dtype=torch.complex64), torch.zeros((1, 2)))

    def test_does_not_change_input_or_attach_gradients(self):
        generated = torch.tensor([[0.5, 1.5]], requires_grad=True)
        reference = torch.tensor([[0.25, 1.25]])
        saved_generated, saved_reference = generated.detach().clone(), reference.clone()
        result = summarize_distribution(generated, reference)
        self.assertTrue(torch.equal(saved_generated, generated))
        self.assertTrue(torch.equal(saved_reference, reference))
        self.assertIsNone(generated.grad)
        self.assertIsInstance(result["aggregate"]["generated_mean_ms"], float)

    def test_random_state_restores_on_success_and_exception(self):
        def draws():
            return random.random(), np.random.rand(), torch.rand(3)

        with preserve_random_state("cpu"):
            random.seed(123)
            np.random.seed(123)
            torch.random.default_generator.manual_seed(123)
            with preserve_random_state("cpu"):
                expected = draws()
            for fail in (False, True):
                with self.subTest(fail=fail):
                    try:
                        with preserve_random_state("cpu"):
                            random.seed(456)
                            np.random.seed(456)
                            torch.random.default_generator.manual_seed(456)
                            draws()
                            if fail:
                                raise RuntimeError("validation failed")
                    except RuntimeError:
                        pass
                    with preserve_random_state("cpu"):
                        actual = draws()
                    self.assertEqual(expected[:2], actual[:2])
                    self.assertTrue(torch.equal(expected[2], actual[2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device RNG test")
    def test_cuda_arithmetic_and_rng_restoration(self):
        device = torch.device("cuda", torch.cuda.current_device())
        with preserve_random_state(device):
            with preserve_random_state(device):
                expected = torch.rand(4, device=device)
            expected_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
            try:
                with preserve_random_state(device):
                    # Model construction uses manual_seed, which seeds every
                    # visible GPU even when validation runs on just one GPU.
                    torch.manual_seed(98765)
                    for index in range(torch.cuda.device_count()):
                        torch.rand(20, device=torch.device("cuda", index))
                    reference = torch.tensor([[0.5, 1.5]], device=device)
                    result = summarize_distribution(reference * 2, reference)
                    self.assertEqual(result["aggregate"]["variance_ratio"], 4.0)
                    raise RuntimeError("validation failed")
            except RuntimeError:
                pass
            actual_cuda_states = torch.cuda.get_rng_state_all()
            self.assertEqual(len(expected_cuda_states), len(actual_cuda_states))
            for index, (expected_state, actual_state) in enumerate(zip(expected_cuda_states, actual_cuda_states)):
                with self.subTest(cuda_index=index):
                    self.assertTrue(torch.equal(expected_state, actual_state))
            self.assertTrue(torch.equal(expected, torch.rand(4, device=device)))


if __name__ == "__main__":
    unittest.main()
