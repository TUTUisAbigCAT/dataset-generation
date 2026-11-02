"""Per-window moments saved during validation, with independent toy expectations."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import validation_metrics as metrics


FIELDS = {
    "generated_mean_ms", "reference_mean_ms", "generated_variance_ms2", "reference_variance_ms2",
    "mean_error_ms", "mean_absolute_error_ms", "variance_error_ms2", "variance_absolute_error_ms2",
    "observed_counts", "gaussian_kl_real_to_generated", "kl_mean_term", "kl_variance_term",
}


class WindowStatisticsTests(unittest.TestCase):
    def test_hand_calculated_ms_moments_signed_errors_and_kl_components(self):
        reference = torch.tensor([[0., .002], [.001, .005]], dtype=torch.float64)
        generated = torch.tensor([[.001, .005], [.001, .003]], dtype=torch.float64)
        actual = metrics.window_statistics(generated, reference)
        self.assertEqual(set(actual), FIELDS)
        expected = {
            "generated_mean_ms": [3., 2.], "reference_mean_ms": [1., 3.],
            "generated_variance_ms2": [4., 1.], "reference_variance_ms2": [1., 4.],
            "mean_error_ms": [2., -1.], "mean_absolute_error_ms": [2., 1.],
            "variance_error_ms2": [3., -3.], "variance_absolute_error_ms2": [3., 3.],
            "observed_counts": [2, 2], "kl_mean_term": [.5, .5],
            "kl_variance_term": [.5 * (math.log(4) - .75), .5 * (3 - math.log(4))],
            "gaussian_kl_real_to_generated": [.5 * (math.log(4) + .25), .5 * (4 - math.log(4))],
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(actual[name].numpy(), values, rtol=1e-14, atol=1e-14)
                self.assertEqual(actual[name].shape, (2,))
                self.assertEqual(actual[name].device, generated.device)
                self.assertEqual(actual[name].dtype, torch.int64 if name == "observed_counts" else torch.float64)
        self.assertTrue(torch.equal(actual["gaussian_kl_real_to_generated"],
                                    actual["kl_variance_term"] + actual["kl_mean_term"]))

    def test_mask_selects_same_positions_and_excludes_unknown_reference_values(self):
        reference = torch.tensor([[1., 3., float("nan"), float("inf")],
                                  [.5, 1., 1.5, float("nan")]], dtype=torch.float64)
        generated = torch.tensor([[-3., -1., 1000., -1000.], [.6, 1.1, 1.6, -999.]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        actual = metrics.window_statistics(generated, reference, observed_mask=mask)
        self.assertEqual(actual["observed_counts"].tolist(), [2, 3])
        self.assertEqual(actual["reference_mean_ms"][0], 2000.)
        self.assertEqual(actual["generated_mean_ms"][0], -2000.)
        self.assertEqual(actual["reference_variance_ms2"][0], 1000000.)
        self.assertEqual(actual["generated_variance_ms2"][0], 1000000.)
        self.assertEqual(actual["gaussian_kl_real_to_generated"][0], 8.)
        for index in range(2):
            real_ms = reference[index, mask[index]].numpy() * 1000
            gen_ms = generated[index, mask[index]].numpy() * 1000
            expected = .5 * (np.log(gen_ms.var() / real_ms.var())
                              + (real_ms.var() + (real_ms.mean() - gen_ms.mean()) ** 2) / gen_ms.var() - 1)
            self.assertAlmostEqual(float(actual["gaussian_kl_real_to_generated"][index]), expected, places=12)
        changed = generated.clone()
        changed[~mask] = -1e40
        again = metrics.window_statistics(changed, reference, observed_mask=mask)
        self.assertTrue(all(torch.equal(actual[name], again[name]) for name in FIELDS))

    def test_variance_floor_only_affects_kl_and_never_raw_statistics(self):
        reference = torch.tensor([[0., 0.], [0., .001]], dtype=torch.float64)
        generated = reference + .002
        floor1 = metrics.window_statistics(generated, reference)
        floor4 = metrics.window_statistics(generated, reference, kl_variance_floor_ms2=4.)
        self.assertEqual(floor4["reference_variance_ms2"].tolist(), [0., .25])
        self.assertEqual(floor4["generated_variance_ms2"].tolist(), [0., .25])
        self.assertEqual(floor4["variance_error_ms2"].tolist(), [0., 0.])
        self.assertEqual(floor1["gaussian_kl_real_to_generated"].tolist(), [2., 2.])
        self.assertEqual(floor4["gaussian_kl_real_to_generated"].tolist(), [.5, .5])
        for name in FIELDS - {"kl_mean_term", "kl_variance_term", "gaussian_kl_real_to_generated"}:
            self.assertTrue(torch.equal(floor1[name], floor4[name]), name)

    def test_no_mask_means_all_samples_including_stored_zero_and_negative_generation(self):
        reference = torch.tensor([[0., 1., 2., 3.]])
        generated = torch.tensor([[-1., 0., 1., 2.]])
        actual = metrics.window_statistics(generated, reference)
        self.assertEqual(actual["observed_counts"].tolist(), [4])
        self.assertEqual(actual["reference_mean_ms"].tolist(), [1500.])
        self.assertEqual(actual["generated_mean_ms"].tolist(), [500.])
        self.assertEqual(actual["reference_variance_ms2"].tolist(), [1250000.])
        self.assertEqual(actual["generated_variance_ms2"].tolist(), [1250000.])
        self.assertAlmostEqual(float(actual["gaussian_kl_real_to_generated"][0]), .4)

    def test_float64_outputs_do_not_mutate_inputs_gradients_or_rng(self):
        generated = torch.tensor([[.3, .8, 1.1]], dtype=torch.float32, requires_grad=True)
        reference = torch.tensor([[.4, .9, 1.3]], dtype=torch.float32, requires_grad=True)
        before = [value.detach().clone() for value in (generated, reference)]
        rng = torch.random.get_rng_state().clone()
        actual = metrics.window_statistics(generated, reference)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (generated, reference))))
        self.assertIsNone(generated.grad)
        self.assertIsNone(reference.grad)
        for name, value in actual.items():
            self.assertFalse(value.requires_grad, name)
            self.assertIsNone(value.grad_fn, name)

    def test_accumulator_reuses_one_statistics_pass_and_returns_it(self):
        reference = torch.tensor([[.5, 1.5], [.7, 1.3], [1., 2.]], dtype=torch.float64)
        generated = reference + torch.tensor([[.1], [.2], [.9]], dtype=torch.float64)
        accumulator = metrics.DistributionAccumulator()
        batches = []
        with mock.patch.object(metrics, "window_statistics", wraps=metrics.window_statistics) as statistics, \
             mock.patch.object(torch, "var_mean", wraps=torch.var_mean) as reduction:
            for start, stop in ((0, 2), (2, 3)):
                batches.append(accumulator.update(generated[start:stop], reference[start:stop]))
        self.assertEqual(statistics.call_count, 2)
        self.assertEqual(reduction.call_count, 4)
        per_window = torch.cat([batch["gaussian_kl_real_to_generated"] for batch in batches])
        expected = float(per_window.sum() / 3)
        self.assertEqual(accumulator.finalize()["aggregate"]["gaussian_kl_real_to_generated"], expected)
        self.assertEqual(set(vars(accumulator)), {"kl_variance_floor_ms2", "count", "window_size", "device", "kl_sum"})
        self.assertEqual(accumulator.count, 3)

    def test_invalid_masks_and_nonfinite_generated_positions_are_rejected(self):
        values = torch.tensor([[1., 2., 0.]])
        for mask, error in (([[True, True, False]], TypeError),
                            (torch.ones(1, 3), TypeError),
                            (torch.ones(3, dtype=torch.bool), ValueError),
                            (torch.tensor([[True, False, False]]), ValueError),
                            (torch.empty((1, 3), dtype=torch.bool, device="meta"), ValueError)):
            with self.subTest(mask=mask), self.assertRaises(error):
                metrics.window_statistics(values, values, observed_mask=mask)
        mask = torch.tensor([[True, True, False]])
        for invalid in (float("nan"), float("inf"), -float("inf")):
            generated = values.clone()
            generated[0, -1] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "every position"):
                metrics.window_statistics(generated, values, observed_mask=mask)
        for floor in (0, -1, True, None, "1", float("nan"), float("inf")):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                metrics.window_statistics(values, values, kl_variance_floor_ms2=floor)

    def test_small_variance_difference_retains_stable_positive_kl_component(self):
        reference = torch.tensor([[-.001, .001]], dtype=torch.float64)
        actual = metrics.window_statistics(reference * (1 + 1e-8), reference,
                                           kl_variance_floor_ms2=1e-6)
        self.assertEqual(float(actual["kl_mean_term"][0]), 0.)
        self.assertGreater(float(actual["kl_variance_term"][0]), 0.)
        self.assertTrue(torch.equal(actual["kl_variance_term"], actual["gaussian_kl_real_to_generated"]))

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA is unavailable")
    def test_cuda_statistics_and_counts_stay_on_input_device(self):
        reference = torch.tensor([[1., 3., 0.]], dtype=torch.float32, device="cuda")
        generated = torch.tensor([[2., 4., -99.]], dtype=torch.float32, device="cuda")
        actual = metrics.window_statistics(generated, reference, observed_mask=reference > 0)
        self.assertEqual(float(actual["gaussian_kl_real_to_generated"][0]), .5)
        for name, value in actual.items():
            self.assertEqual(value.device, generated.device, name)
            self.assertEqual(value.dtype, torch.int64 if name == "observed_counts" else torch.float64)


if __name__ == "__main__":
    unittest.main()
