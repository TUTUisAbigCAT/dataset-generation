"""Independent numerical cases for all-window diagonal Gaussian KL."""
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_metrics import DistributionAccumulator, preserve_random_state, summarize_distribution


class ValidationMetricsTests(unittest.TestCase):
    def kl(self, generated, reference, **options):
        result = summarize_distribution(generated, reference, **options)
        self.assertEqual(set(result), {"aggregate"})
        self.assertEqual(set(result["aggregate"]), {"gaussian_kl_real_to_generated"})
        json.dumps(result, allow_nan=False)
        return result["aggregate"]["gaussian_kl_real_to_generated"]

    def test_identity_is_zero_for_nonconstant_and_constant_coordinates(self):
        values = torch.tensor([[0.5, 1.5, 0.0], [1.0, 0.5, 0.0],
                               [1.5, 1.0, 0.0]], dtype=torch.float64)
        self.assertEqual(self.kl(values, values), 0.0)
        self.assertEqual(self.kl(torch.zeros_like(values), torch.zeros_like(values)), 0.0)

    def test_hand_calculated_mean_variance_and_kl_direction(self):
        # In ms: reference means [1,3], population variances [1,4];
        # generated means [3,2], population variances [4,1]. Log terms cancel.
        reference = torch.tensor([[0.0, 0.001], [0.002, 0.005]], dtype=torch.float64)
        generated = torch.tensor([[0.001, 0.001], [0.005, 0.003]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(generated, reference), 2.125, places=12)
        self.assertAlmostEqual(self.kl(reference, generated), 3.25, places=12)

    def test_constant_mean_shift_and_dimension_sum(self):
        for dimension in (2, 750, 1500, 3000, 6000):
            with self.subTest(dimension=dimension):
                reference = torch.tensor([[0.5], [1.5]], dtype=torch.float64).repeat(1, dimension)
                generated = reference + 0.125
                # Per coordinate: 0.5 * 125^2 / 250000 = 1/32 nats.
                self.assertAlmostEqual(self.kl(generated, reference), dimension / 32.0, places=12)

    def test_all_window_distribution_does_not_average_paired_window_kl(self):
        reference = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64)
        generated = reference.flip(0)
        self.assertEqual(self.kl(generated, reference), 0.0)
        paired_average = sum(self.kl(generated[index:index + 1], reference[index:index + 1])
                             for index in range(2)) / 2
        self.assertEqual(paired_average, 1000000.0)

    def test_independent_row_permutations_preserve_kl(self):
        reference = torch.tensor([[0.2, 0.8, 1.4], [0.3, 1.1, 1.2],
                                  [1.3, 0.4, 0.9]], dtype=torch.float64)
        generated = torch.tensor([[0.5, 0.8, 1.0], [0.4, 1.5, 1.1],
                                  [0.9, 0.2, 0.7]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(generated, reference),
                               self.kl(generated[[2, 0, 1]], reference[[1, 2, 0]]), places=12)

    def test_diagonal_approximation_does_not_measure_cross_coordinate_covariance(self):
        reference = torch.tensor([[0.5, 0.5], [1.5, 1.5]], dtype=torch.float64)
        generated = torch.tensor([[0.5, 1.5], [1.5, 0.5]], dtype=torch.float64)
        # The correlation changes from +1 to -1; coordinate moments still match.
        self.assertEqual(self.kl(generated, reference), 0.0)

    def test_chunk_merges_match_numpy_population_moments_with_short_last_batch(self):
        generated_ms = np.array([[100, 200, 300, 400], [150, 250, 350, 450],
                                 [800, 500, 200, 100], [200, 400, 600, 800],
                                 [1000, 900, 800, 700], [0, -100, 300, 600],
                                 [3000, 4000, 5000, 6000]], dtype=np.float64)
        reference_ms = np.array([[200, 100, 500, 300], [500, 250, 0, 900],
                                 [900, 700, 600, 500], [100, 200, 300, 400],
                                 [600, 800, 1000, 1200], [700, 400, 100, -200],
                                 [1000, 1500, 2000, 2500]], dtype=np.float64)
        gm, rm = generated_ms.mean(axis=0), reference_ms.mean(axis=0)
        gv, rv = generated_ms.var(axis=0, ddof=0), reference_ms.var(axis=0, ddof=0)
        expected = 0.5 * np.sum(np.log(gv / rv) + (rv + (rm - gm) ** 2) / gv - 1)
        generated, reference = torch.from_numpy(generated_ms / 1000), torch.from_numpy(reference_ms / 1000)
        for boundaries in ((0, 3, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7), (0, 7)):
            with self.subTest(boundaries=boundaries):
                accumulator = DistributionAccumulator()
                for start, stop in zip(boundaries, boundaries[1:]):
                    accumulator.update(generated[start:stop], reference[start:stop])
                actual = accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"]
                self.assertEqual(accumulator.count, 7)
                self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(self.kl(generated, reference), expected, places=12)

    def test_variance_floor_is_in_ms_squared_and_does_not_clip_inputs(self):
        generated = torch.tensor([[0.002, -0.002]], dtype=torch.float64, requires_grad=True)
        reference = torch.zeros_like(generated)
        saved_generated, saved_reference = generated.detach().clone(), reference.detach().clone()
        self.assertEqual(self.kl(generated, reference, kl_variance_floor_ms2=4.0), 1.0)
        self.assertEqual(self.kl(generated, reference), 4.0)
        self.assertTrue(torch.equal(generated, saved_generated))
        self.assertTrue(torch.equal(reference, saved_reference))
        self.assertIsNone(generated.grad)

    def test_floor_applies_to_both_variances_after_all_window_accumulation(self):
        reference = torch.tensor([[0.0, 0.0], [0.001, 0.001]], dtype=torch.float64)
        generated = reference + 0.001
        accumulator = DistributionAccumulator(kl_variance_floor_ms2=4.0)
        for index in range(2):
            accumulator.update(generated[index:index + 1], reference[index:index + 1])
        self.assertEqual(accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"], 0.25)
        # The floor must not corrupt stored moments, which still have v=0.25 ms^2.
        np.testing.assert_allclose((accumulator.reference_m2 / 2).numpy(), [0.25, 0.25])
        np.testing.assert_allclose((accumulator.generated_m2 / 2).numpy(), [0.25, 0.25])
        zero = torch.zeros((2, 2), dtype=torch.float64)
        nonzero = torch.tensor([[-0.002, -0.002], [0.002, 0.002]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(nonzero, zero), math.log(4) - 0.75, places=12)
        self.assertAlmostEqual(self.kl(zero, nonzero), 3 - math.log(4), places=12)

    def test_nearly_equal_variances_keep_positive_sub_epsilon_kl(self):
        reference = torch.tensor([[-0.001, -0.001], [0.001, 0.001]], dtype=torch.float64)
        generated = reference * (1.0 + 1e-8)
        # Independent high-precision evaluation avoids the cancellation in
        # log(vg/vr) + vr/vg - 1 when the two variances almost coincide.
        gv = float((generated.numpy() * 1000).var(axis=0, ddof=0)[0])
        rv = float((reference.numpy() * 1000).var(axis=0, ddof=0)[0])
        with localcontext() as context:
            context.prec = 70
            ratio = Decimal.from_float(rv) / Decimal.from_float(gv)
            expected = float(ratio - 1 - ratio.ln())
        actual = self.kl(generated, reference, kl_variance_floor_ms2=1e-6)
        self.assertGreater(actual, 0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0)

    def test_tiny_variance_ratio_uses_log_difference_without_underflow(self):
        reference = torch.zeros((2, 2), dtype=torch.float64)
        generated = torch.tensor([[-1e47, -1e47], [1e47, 1e47]], dtype=torch.float64)
        actual = self.kl(generated, reference, kl_variance_floor_ms2=1e-300)
        self.assertAlmostEqual(actual, 400 * math.log(10) - 1, places=10)
        with self.assertRaises(FloatingPointError):
            self.kl(reference, generated, kl_variance_floor_ms2=1e-300)

    def test_rejects_invalid_variance_floor(self):
        values = torch.ones((2, 2))
        for floor in (True, False, None, "1", 0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    DistributionAccumulator(kl_variance_floor_ms2=floor)
                with self.assertRaises(ValueError):
                    summarize_distribution(values, values, kl_variance_floor_ms2=floor)

    def test_rejects_nonfinite_and_overflow(self):
        for value in (float("nan"), float("inf"), float("-inf"), 1e308):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_distribution(torch.tensor([[0.0, value]], dtype=torch.float64),
                                       torch.zeros((1, 2), dtype=torch.float64))
        with self.assertRaises(FloatingPointError):
            summarize_distribution(torch.tensor([[-1e200, 1e200]], dtype=torch.float64),
                                   torch.zeros((1, 2), dtype=torch.float64))

    def test_rejects_shape_type_device_and_cross_batch_feature_mismatch(self):
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
        with self.assertRaises(ValueError):
            DistributionAccumulator().finalize()
        accumulator = DistributionAccumulator()
        accumulator.update(torch.ones((2, 4)), torch.zeros((2, 4)))
        with self.assertRaises(ValueError):
            accumulator.update(torch.ones((1, 3)), torch.zeros((1, 3)))

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for streaming KL test")
    def test_cuda_streaming_kl_remains_on_device_and_matches_analytic_value(self):
        reference = torch.tensor([[0.0, 0.001], [0.002, 0.005]], dtype=torch.float64, device="cuda")
        generated = torch.tensor([[0.001, 0.001], [0.005, 0.003]], dtype=torch.float64, device="cuda")
        accumulator = DistributionAccumulator()
        accumulator.update(generated[:1], reference[:1])
        accumulator.update(generated[1:], reference[1:])
        for value in (accumulator.generated_mean, accumulator.reference_mean,
                      accumulator.generated_m2, accumulator.reference_m2):
            self.assertEqual(value.device, generated.device)
            self.assertEqual(value.dtype, torch.float64)
        self.assertAlmostEqual(accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"], 2.125, places=12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device RNG test")
    def test_cuda_arithmetic_and_rng_restoration(self):
        device = torch.device("cuda", torch.cuda.current_device())
        with preserve_random_state(device):
            with preserve_random_state(device):
                expected = torch.rand(4, device=device)
            expected_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
            try:
                with preserve_random_state(device):
                    torch.manual_seed(98765)
                    for index in range(torch.cuda.device_count()):
                        torch.rand(20, device=torch.device("cuda", index))
                    reference = torch.tensor([[0.5, 1.5]], device=device)
                    self.assertEqual(self.kl(reference, reference), 0.0)
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
