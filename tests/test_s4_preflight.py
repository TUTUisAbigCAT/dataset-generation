"""The S4 preflight must validate numerics and fail closed for requested CUDA."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_s4_backends as preflight


class S4PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_cpu_run_checks_reports_all_numeric_and_gradient_checks(self):
        report = preflight.run_checks("cpu")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["requested_device"], "cpu")
        self.assertEqual(report["torch_version"], torch.__version__)
        self.assertIsInstance(report["cuda_available"], bool)
        self.assertIsInstance(report["cuda_extension_imported"], bool)
        self.assertIsInstance(report["pykeops_imported"], bool)
        combinations = {(row["dtype"], row["backend"], row["z_gradient"])
                        for row in report["resolvent_checks"]}
        expected = {(str(dtype), backend, z_gradient)
                    for dtype in (torch.complex64, torch.complex128)
                    for backend in ("torch", "auto")
                    for z_gradient in (False, True)}
        self.assertEqual(combinations, expected)
        self.assertEqual(len(report["resolvent_checks"]), len(expected))
        for row in report["resolvent_checks"]:
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["device"], "cpu")
            tolerance = 1e-4 if row["dtype"] == str(torch.complex64) else 1e-9
            for name in ("output_max_absolute_error", "gradient_max_absolute_error"):
                self.assertTrue(math.isfinite(row[name]))
                self.assertGreaterEqual(row[name], 0)
                self.assertLess(row[name], tolerance)
        self.assertEqual({row["precision"] for row in report["kernel_checks"]}, {1, 2})
        self.assertEqual(len(report["kernel_checks"]), 2)
        for row in report["kernel_checks"]:
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["device"], "cpu")
            self.assertGreater(row["parameters_with_finite_gradients"], 0)
            tolerance = 1e-4 if row["precision"] == 1 else 1e-9
            self.assertLess(row["dense_recurrence_relative_l2"], tolerance)
            self.assertTrue(math.isfinite(row["dense_recurrence_max_absolute_error"]))
        # A saved successful report must be strict JSON, with no NaN/Infinity.
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False))["status"], "passed")

    def test_requested_cuda_or_both_fails_without_cpu_substitution(self):
        for choice in ("cuda", "both"):
            with self.subTest(choice=choice), \
                 mock.patch.object(torch.cuda, "is_available", return_value=False), \
                 mock.patch.object(preflight, "check_resolvent") as resolvent, \
                 mock.patch.object(preflight, "check_kernel") as kernel:
                with self.assertRaisesRegex(RuntimeError, "CUDA was requested.*no CPU substitution"):
                    preflight.run_checks(choice)
                # Fail before returning any partial CPU checks as requested CUDA.
                resolvent.assert_not_called()
                kernel.assert_not_called()

    def test_invalid_device_choice_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "device must be"):
            preflight.run_checks("automatic")

    def test_legacy_half_state_backend_is_rejected_by_numeric_check(self):
        def legacy_half(v, z, w):
            return (v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))).sum(-2)
        # Reintroduce the actual discovered defect, rather than making the
        # checker itself raise or substituting an arbitrary failed assertion.
        with mock.patch.object(preflight.S4, "cauchy_slow", side_effect=legacy_half), \
             self.assertRaises(AssertionError):
            preflight.run_checks("cpu")

    def test_correct_forward_but_incorrect_backward_is_rejected(self):
        original = preflight.S4.cauchy_resolvent
        def broken_backward(v, z, w):
            correct = original(v, z, w)
            # Forward values are exact; the graph remains connected but every
            # input gradient is deliberately zero, so only backward checks fail.
            return correct.detach() + 0 * correct
        with mock.patch.object(preflight.S4, "cauchy_resolvent", side_effect=broken_backward), \
             self.assertRaises(AssertionError):
            preflight.check_resolvent(torch.device("cpu"), torch.complex128, "torch", True)

    def test_backend_precision_change_is_rejected(self):
        original = preflight.S4.cauchy_resolvent
        def wrong_precision(v, z, w):
            return original(v, z, w).to(torch.complex64)
        with mock.patch.object(preflight.S4, "cauchy_resolvent", side_effect=wrong_precision), \
             self.assertRaisesRegex(AssertionError, "changed device or precision"):
            preflight.check_resolvent(torch.device("cpu"), torch.complex128, "torch", True)

    def test_cli_serializes_cuda_failure_and_returns_nonzero(self):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["check_s4_backends.py", "--device", "cuda", "--threads", "1"]), \
             mock.patch.object(torch.cuda, "is_available", return_value=False), \
             contextlib.redirect_stdout(output):
            exit_code = preflight.main()
        self.assertEqual(exit_code, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["requested_device"], "cuda")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertIn("no CPU substitution", report["error"])


if __name__ == "__main__":
    unittest.main()
