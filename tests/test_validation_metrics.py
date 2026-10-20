"""Independent numerical cases for the physical-unit validation diagnostics."""
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
    def test_identity_and_population_variance(self):
        values = torch.tensor([[0.5, 1.5], [0.5, 1.5]], dtype=torch.float64)
        result = summarize_distribution(values, values)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["generated_mean_ms"], 1000.0)
        self.assertEqual(aggregate["generated_variance_ms2"], 250000.0)
        self.assertEqual(aggregate["mean_absolute_error_ms"], 0.0)
        self.assertEqual(aggregate["variance_absolute_error_ms2"], 0.0)
        self.assertEqual(aggregate["variance_ratio"], 1.0)
        self.assertEqual(aggregate["mean_vector_l2_ms"], 0.0)
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 0.0)
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

    def test_equal_pooled_moments_do_not_hide_coordinate_mean_error(self):
        reference = torch.tensor([[0.5, 1.5], [0.5, 1.5]], dtype=torch.float64)
        generated = torch.tensor([[1.5, 0.5], [1.5, 0.5]], dtype=torch.float64)
        aggregate = summarize_distribution(generated, reference)["aggregate"]
        self.assertEqual(aggregate["mean_absolute_error_ms"], 0.0)
        self.assertEqual(aggregate["variance_absolute_error_ms2"], 0.0)
        self.assertEqual(aggregate["mean_window_absolute_error_ms"], 0.0)
        self.assertAlmostEqual(aggregate["mean_vector_l2_ms"], 1000.0 * math.sqrt(2.0))
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 0.0)

    def test_equal_window_moments_do_not_hide_coordinate_variance_error(self):
        # Every row has mean 1000 ms and population variance 500000 ms^2.
        # All coordinate means also match, but the varying positions differ.
        reference = torch.tensor([[0.0, 2.0, 1.0, 1.0],
                                  [2.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
        generated = torch.tensor([[1.0, 1.0, 0.0, 2.0],
                                  [1.0, 1.0, 2.0, 0.0]], dtype=torch.float64)
        aggregate = summarize_distribution(generated, reference)["aggregate"]
        self.assertEqual(aggregate["mean_window_absolute_error_ms"], 0.0)
        self.assertEqual(aggregate["variance_window_absolute_error_ms2"], 0.0)
        self.assertEqual(aggregate["mean_absolute_error_ms"], 0.0)
        self.assertEqual(aggregate["variance_absolute_error_ms2"], 0.0)
        self.assertEqual(aggregate["mean_vector_l2_ms"], 0.0)
        # The variance difference vector is [-1e6, -1e6, 1e6, 1e6].
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 2000000.0)

    def test_vector_metrics_are_invariant_to_independent_row_permutations(self):
        reference = torch.tensor([[0.2, 0.8, 1.4], [0.3, 1.1, 1.2],
                                  [1.3, 0.4, 0.9]], dtype=torch.float64)
        generated = torch.tensor([[0.5, 0.8, 1.0], [0.4, 1.5, 1.1],
                                  [0.9, 0.2, 0.7]], dtype=torch.float64)
        original = summarize_distribution(generated, reference)["aggregate"]
        shuffled = summarize_distribution(generated[[2, 0, 1]], reference[[1, 2, 0]])["aggregate"]
        for key in ("mean_vector_l2_ms", "variance_vector_l2_ms2"):
            with self.subTest(metric=key):
                self.assertAlmostEqual(original[key], shuffled[key], places=7)

    def test_3000_dimensional_distance_is_l2_not_coordinate_rmse(self):
        reference = torch.ones((4, 3000), dtype=torch.float64)
        generated = reference + 0.125
        aggregate = summarize_distribution(generated, reference)["aggregate"]
        self.assertAlmostEqual(aggregate["mean_vector_l2_ms"], 125.0 * math.sqrt(3000.0))
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 0.0)

    def test_single_window_has_zero_coordinate_population_variance(self):
        generated = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
        reference = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        aggregate = summarize_distribution(generated, reference)["aggregate"]
        self.assertAlmostEqual(aggregate["mean_vector_l2_ms"], 1000.0 * math.sqrt(5.0))
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 0.0)
        self.assertGreater(aggregate["generated_variance_ms2"], 0.0)
        json.dumps(aggregate, allow_nan=False)

    def test_chunked_accumulation_matches_numpy_all_rows_and_short_last_batch(self):
        # Deliberately unequal chunk sizes and a final row far from the rest
        # catch unweighted batch averages and missing between-batch variance.
        generated_ms = np.array([[100, 200, 300, 400], [150, 250, 350, 450],
                                 [800, 500, 200, 100], [200, 400, 600, 800],
                                 [1000, 900, 800, 700], [0, -100, 300, 600],
                                 [3000, 4000, 5000, 6000]], dtype=np.float64)
        reference_ms = np.array([[200, 100, 500, 300], [500, 250, 0, 900],
                                 [900, 700, 600, 500], [100, 200, 300, 400],
                                 [600, 800, 1000, 1200], [700, 400, 100, -200],
                                 [1000, 1500, 2000, 2500]], dtype=np.float64)
        generated = torch.from_numpy(generated_ms / 1000.0).requires_grad_()
        reference = torch.from_numpy(reference_ms / 1000.0)
        saved_generated, saved_reference = generated.detach().clone(), reference.clone()
        accumulator = DistributionAccumulator()
        for start, stop in ((0, 3), (3, 6), (6, 7)):
            accumulator.update(generated[start:stop], reference[start:stop])
        result = accumulator.finalize()
        aggregate = result["aggregate"]
        expected = {
            "generated_mean_ms": generated_ms.mean(),
            "reference_mean_ms": reference_ms.mean(),
            "generated_variance_ms2": generated_ms.var(ddof=0),
            "reference_variance_ms2": reference_ms.var(ddof=0),
            "mean_vector_l2_ms": np.linalg.norm(generated_ms.mean(axis=0) - reference_ms.mean(axis=0)),
            "variance_vector_l2_ms2": np.linalg.norm(generated_ms.var(axis=0, ddof=0)
                                                    - reference_ms.var(axis=0, ddof=0)),
            "mean_window_absolute_error_ms": np.abs(generated_ms.mean(axis=1)
                                                      - reference_ms.mean(axis=1)).mean(),
            "variance_window_absolute_error_ms2": np.abs(generated_ms.var(axis=1, ddof=0)
                                                           - reference_ms.var(axis=1, ddof=0)).mean(),
        }
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(aggregate[key], value, places=7)
        self.assertEqual(len(result["per_window"]), 7)
        for index, row in enumerate(result["per_window"]):
            with self.subTest(window=index):
                self.assertEqual(row["window_index"], index)
                self.assertAlmostEqual(row["generated_mean_ms"], generated_ms[index].mean())
                self.assertAlmostEqual(row["reference_mean_ms"], reference_ms[index].mean())
                self.assertAlmostEqual(row["generated_variance_ms2"], generated_ms[index].var(ddof=0))
                self.assertAlmostEqual(row["reference_variance_ms2"], reference_ms[index].var(ddof=0))
        direct = summarize_distribution(generated, reference)
        for key, value in direct["aggregate"].items():
            with self.subTest(direct_metric=key):
                if value is None:
                    self.assertIsNone(aggregate[key])
                else:
                    self.assertAlmostEqual(aggregate[key], value, places=7)
        self.assertTrue(torch.equal(generated, saved_generated))
        self.assertTrue(torch.equal(reference, saved_reference))
        self.assertIsNone(generated.grad)
        json.dumps(result, allow_nan=False)

    def test_accumulator_rejects_empty_result_and_feature_count_changes(self):
        with self.assertRaises(ValueError):
            DistributionAccumulator().finalize()
        accumulator = DistributionAccumulator()
        accumulator.update(torch.ones((2, 4)), torch.zeros((2, 4)))
        with self.assertRaises(ValueError):
            accumulator.update(torch.ones((1, 3)), torch.zeros((1, 3)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for streaming metric test")
    def test_cuda_streaming_vector_metrics_match_hand_calculated_distances(self):
        reference = torch.tensor([[0.0, 2.0, 1.0, 1.0],
                                  [2.0, 0.0, 1.0, 1.0]], dtype=torch.float64, device="cuda")
        generated = torch.tensor([[1.0, 1.0, 0.0, 2.0],
                                  [1.0, 1.0, 2.0, 0.0]], dtype=torch.float64, device="cuda")
        accumulator = DistributionAccumulator()
        accumulator.update(generated[:1], reference[:1])
        accumulator.update(generated[1:], reference[1:])
        aggregate = accumulator.finalize()["aggregate"]
        self.assertEqual(aggregate["mean_vector_l2_ms"], 0.0)
        self.assertEqual(aggregate["variance_vector_l2_ms2"], 2000000.0)
        json.dumps(aggregate, allow_nan=False)

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
