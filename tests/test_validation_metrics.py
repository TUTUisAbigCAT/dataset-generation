"""Independent numerical cases for the mean of paired within-window Gaussian KLs."""
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

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

    def test_identity_is_zero_for_nonconstant_and_constant_windows(self):
        values = torch.tensor([[0.5, 1.5, 0.0], [1.0, 0.5, 0.0],
                               [1.5, 1.0, 0.0]], dtype=torch.float64)
        self.assertEqual(self.kl(values, values), 0.0)
        self.assertEqual(self.kl(torch.zeros_like(values), torch.zeros_like(values)), 0.0)

    def test_hand_calculated_mean_variance_and_kl_direction(self):
        # Each ROW is one window. In ms, reference means [1,3] and population
        # variances [1,4]; generated means [3,2] and variances [4,1].
        # real||gen: KLs=(log(4)+1/4)/2 and (-log(4)+4)/2; average=17/16.
        reference = torch.tensor([[0.0, 0.002], [0.001, 0.005]], dtype=torch.float64)
        generated = torch.tensor([[0.001, 0.005], [0.001, 0.003]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(generated, reference), 17 / 16, places=12)
        self.assertAlmostEqual(self.kl(reference, generated), 13 / 8, places=12)

    def test_repeating_time_samples_does_not_multiply_kl_by_window_length(self):
        for dimension in (2, 750, 1500, 3000, 6000):
            with self.subTest(dimension=dimension):
                reference = torch.tensor([[0.5, 1.5]], dtype=torch.float64).repeat(1, dimension // 2)
                generated = reference + 0.125
                # Within the window: 0.5 * 125^2 / 250000 = 1/32 nats.
                self.assertAlmostEqual(self.kl(generated, reference), 1 / 32, places=12)

    def test_one_sided_window_permutation_changes_the_paired_metric(self):
        reference = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64)
        generated = reference.flip(0)
        self.assertEqual(self.kl(reference, reference), 0.0)
        # Both windows have variance 62500 ms^2, but their means differ by
        # 1000 ms after swapping: each paired KL is 1000^2/(2*62500)=8.
        self.assertEqual(self.kl(generated, reference), 8.0)
        paired_average = sum(self.kl(generated[index:index + 1], reference[index:index + 1])
                             for index in range(2)) / 2
        self.assertEqual(paired_average, 8.0)

    def test_joint_window_permutation_preserves_kl(self):
        reference = torch.tensor([[0.2, 0.8, 1.4], [0.3, 1.1, 1.2],
                                  [1.3, 0.4, 0.9]], dtype=torch.float64)
        generated = torch.tensor([[0.5, 0.8, 1.0], [0.4, 1.5, 1.1],
                                  [0.9, 0.2, 0.7]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(generated, reference),
                               self.kl(generated[[2, 0, 1]], reference[[2, 0, 1]]), places=12)

    def test_time_permutations_do_not_measure_waveform_alignment(self):
        reference = torch.tensor([[0.5, 0.7, 0.9, 1.1], [1.0, 1.2, 1.6, 1.8]], dtype=torch.float64)
        generated = reference[:, [3, 0, 2, 1]]
        # Rearranging time preserves within-window moments, despite nonzero
        # pointwise error. This metric does not check temporal waveform shape.
        self.assertGreater(float((generated - reference).square().mean()), 0.0)
        self.assertAlmostEqual(self.kl(generated, reference), 0.0, places=12)

    def test_batch_accumulation_matches_numpy_window_mean_with_short_last_batch(self):
        generated_ms = np.array([[100, 200, 300, 400], [150, 250, 350, 450],
                                 [800, 500, 200, 100], [200, 400, 600, 800],
                                 [1000, 900, 800, 700], [0, -100, 300, 600],
                                 [3000, 4000, 5000, 6000]], dtype=np.float64)
        reference_ms = np.array([[200, 100, 500, 300], [500, 250, 0, 900],
                                 [900, 700, 600, 500], [100, 200, 300, 400],
                                 [600, 800, 1000, 1200], [700, 400, 100, -200],
                                 [1000, 1500, 2000, 2500]], dtype=np.float64)
        gm, rm = generated_ms.mean(axis=1), reference_ms.mean(axis=1)
        gv, rv = generated_ms.var(axis=1, ddof=0), reference_ms.var(axis=1, ddof=0)
        window_kl = 0.5 * (np.log(gv / rv) + (rv + (rm - gm) ** 2) / gv - 1)
        expected = window_kl.mean()
        incorrect_batch_average = (window_kl[:3].mean() + window_kl[3:6].mean() + window_kl[6:].mean()) / 3
        self.assertGreater(abs(expected - incorrect_batch_average), 0.1)
        generated, reference = torch.from_numpy(generated_ms / 1000), torch.from_numpy(reference_ms / 1000)
        for boundaries in ((0, 3, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7), (0, 7)):
            with self.subTest(boundaries=boundaries):
                accumulator = DistributionAccumulator()
                for start, stop in zip(boundaries, boundaries[1:]):
                    accumulator.update(generated[start:stop], reference[start:stop])
                actual = accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"]
                self.assertEqual(accumulator.count, 7)
                self.assertEqual(accumulator.kl_sum.shape, torch.Size([]))
                self.assertEqual(accumulator.kl_sum.dtype, torch.float64)
                self.assertEqual(accumulator.kl_sum.device, generated.device)
                self.assertAlmostEqual(float(accumulator.kl_sum), window_kl.sum(), places=12)
                self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(self.kl(generated, reference), expected, places=12)

    def test_variance_floor_is_in_ms_squared_and_does_not_clip_inputs(self):
        generated = torch.tensor([[0.002, 0.002], [-0.002, -0.002]], dtype=torch.float64, requires_grad=True)
        reference = torch.zeros_like(generated)
        saved_generated, saved_reference = generated.detach().clone(), reference.detach().clone()
        self.assertEqual(self.kl(generated, reference, kl_variance_floor_ms2=4.0), 0.5)
        self.assertEqual(self.kl(generated, reference), 2.0)
        accumulator = DistributionAccumulator()
        accumulator.update(generated, reference)
        self.assertFalse(accumulator.kl_sum.requires_grad)
        self.assertIsNone(accumulator.kl_sum.grad_fn)
        self.assertTrue(torch.equal(generated, saved_generated))
        self.assertTrue(torch.equal(reference, saved_reference))
        self.assertIsNone(generated.grad)

    def test_floor_applies_to_each_window_population_variance_in_both_directions(self):
        reference = torch.tensor([[0.0, 0.001]], dtype=torch.float64)
        generated = reference + 0.001
        # Both actual variances are 0.25 ms^2, independently raised to 4;
        # the 1 ms mean difference then gives 1/(2*4)=0.125 nats.
        self.assertEqual(self.kl(generated, reference, kl_variance_floor_ms2=4.0), 0.125)
        zero = torch.zeros((1, 2), dtype=torch.float64)
        nonzero = torch.tensor([[-0.002, 0.002]], dtype=torch.float64)
        self.assertAlmostEqual(self.kl(nonzero, zero), 0.5 * (math.log(4) - 0.75), places=12)
        self.assertAlmostEqual(self.kl(zero, nonzero), 0.5 * (3 - math.log(4)), places=12)

    def test_a_single_window_has_its_own_nonzero_population_variance(self):
        reference = torch.tensor([[-0.001, 0.001]], dtype=torch.float64)
        generated = reference * 2
        expected = 0.5 * (math.log(4) + 0.25 - 1)
        self.assertAlmostEqual(self.kl(generated, reference), expected, places=12)
        self.assertAlmostEqual(self.kl(generated.repeat(3, 1), reference.repeat(3, 1)), expected, places=12)

    def test_nearly_equal_variances_keep_positive_sub_epsilon_kl(self):
        reference = torch.tensor([[-0.001, 0.001]], dtype=torch.float64)
        generated = reference * (1.0 + 1e-8)
        # Independent high-precision evaluation avoids the cancellation in
        # log(vg/vr) + vr/vg - 1 when the two variances almost coincide.
        gv = float((generated.numpy() * 1000).var(axis=1, ddof=0)[0])
        rv = float((reference.numpy() * 1000).var(axis=1, ddof=0)[0])
        with localcontext() as context:
            context.prec = 70
            ratio = Decimal.from_float(rv) / Decimal.from_float(gv)
            expected = float((ratio - 1 - ratio.ln()) / 2)
        actual = self.kl(generated, reference, kl_variance_floor_ms2=1e-6)
        self.assertGreater(actual, 0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0)

    def test_tiny_variance_ratio_uses_log_difference_without_underflow(self):
        reference = torch.zeros((1, 2), dtype=torch.float64)
        generated = torch.tensor([[-1e47, 1e47]], dtype=torch.float64)
        actual = self.kl(generated, reference, kl_variance_floor_ms2=1e-300)
        self.assertAlmostEqual(actual, 0.5 * (400 * math.log(10) - 1), places=10)
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

    def test_cuda_rng_request_without_cuda_is_not_silently_replaced_by_cpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "available CUDA device"):
                with preserve_random_state("cuda"):
                    self.fail("An unavailable CUDA request must not enter the context")
        with self.assertRaises(ValueError):
            with preserve_random_state("meta"):
                self.fail("Unsupported devices must not enter the context")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for streaming KL test")
    def test_cuda_streaming_kl_remains_on_device_and_matches_analytic_value(self):
        reference = torch.tensor([[0.0, 0.002], [0.001, 0.005]], dtype=torch.float64, device="cuda")
        generated = torch.tensor([[0.001, 0.005], [0.001, 0.003]], dtype=torch.float64, device="cuda")
        accumulator = DistributionAccumulator()
        accumulator.update(generated[:1], reference[:1])
        accumulator.update(generated[1:], reference[1:])
        self.assertEqual(accumulator.kl_sum.device, generated.device)
        self.assertEqual(accumulator.kl_sum.dtype, torch.float64)
        self.assertEqual(accumulator.kl_sum.ndim, 0)
        self.assertAlmostEqual(accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"], 17 / 16, places=12)
        with self.assertRaisesRegex(ValueError, "same.*device"):
            accumulator.update(generated.cpu(), reference.cpu())

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
