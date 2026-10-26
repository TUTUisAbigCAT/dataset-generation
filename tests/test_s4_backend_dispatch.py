"""Cauchy backend contracts: device dispatch must not change the S4 operator.

Metadata-only tensor fixtures exercise CUDA routing on CPU-only test hosts.
They do not establish CUDA execution: separate, explicitly skipped hardware
tests execute the real fallback and any actually installed accelerators.
"""
import contextlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
S4 = importlib.import_module("TSImputers.S4Model")


def full_state_reference(v, z, w):
    """Independent two-term form, without concatenating stored state arrays."""
    first = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    second = v.conj().unsqueeze(-1) / (z.unsqueeze(-2) - w.conj().unsqueeze(-1))
    return first.sum(-2) + second.sum(-2)


def example(dtype=torch.complex128, *, device="cpu", states=4, z_grad=True):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    rng = torch.Generator(device="cpu").manual_seed(260420)
    v = torch.randn(2, 3, states, dtype=dtype, generator=rng)
    w = -1 - torch.rand(3, states, dtype=real_dtype, generator=rng)
    w = w + 1j * torch.randn(3, states, dtype=real_dtype, generator=rng)
    # Nonzero real z also rejects an incorrect 2*real(half-state sum) shortcut.
    z = .3 + 1j * torch.linspace(0, 5, 9, dtype=real_dtype)
    return tuple(value.to(device).detach().requires_grad_(grad)
                 for value, grad in ((v, True), (z, z_grad), (w, True)))


class ReportedDeviceTensor(torch.Tensor):
    """Real CPU storage with fake CUDA metadata solely for routing tests."""
    @staticmethod
    def __new__(cls, value, device="cuda:0"):
        result = torch.Tensor._make_subclass(cls, value, value.requires_grad)
        result.reported_device = torch.device(device)
        return result

    @property
    def device(self):
        # Tensor operations retain the subclass but not arbitrary attributes.
        return getattr(self, "reported_device", torch.device("cuda:0"))

    @property
    def is_cuda(self):
        return self.device.type == "cuda"


def plain(value):
    return value.as_subclass(torch.Tensor) if isinstance(value, ReportedDeviceTensor) else value


class S4BackendDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_operator_and_gradients(self, result, tensors, *, rtol=1e-10, atol=1e-11):
        v, z, w = tensors
        reference = full_state_reference(*(plain(value) for value in tensors))
        self.assertEqual(result.dtype, v.dtype)
        torch.testing.assert_close(plain(result), reference, rtol=rtol, atol=atol)
        active = tuple(value for value in tensors if value.requires_grad)
        got = torch.autograd.grad(result.abs().square().sum(), active)
        expected = torch.autograd.grad(reference.abs().square().sum(), active)
        for actual, wanted in zip(got, expected):
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(plain(actual), plain(wanted), rtol=rtol, atol=atol)

    def test_cpu_uses_fallback_even_when_both_accelerators_are_installed(self):
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype), \
                 mock.patch.object(S4, "has_cauchy_extension", True), \
                 mock.patch.object(S4, "has_pykeops", True), \
                 mock.patch.object(S4, "cauchy_mult", create=True,
                                   side_effect=AssertionError("CUDA extension received CPU input")), \
                 mock.patch.object(S4, "cauchy_conj", create=True,
                                   side_effect=AssertionError("GPU KeOps received CPU input")), \
                 mock.patch.object(torch.cuda, "device", side_effect=AssertionError("CPU selected CUDA context")):
                tensors = example(dtype)
                result = S4.cauchy_resolvent(*tensors)
                self.assertEqual(result.device.type, "cpu")
                tolerance = 2e-5 if dtype == torch.complex64 else 1e-10
                self.assert_operator_and_gradients(result, tensors, rtol=tolerance, atol=tolerance)

    def test_fallback_is_defined_after_import_with_optional_modules_present(self):
        extension = ModuleType("extensions.cauchy.cauchy")
        extension.cauchy_mult = mock.Mock(side_effect=AssertionError("CPU must not use extension"))
        keops_torch = ModuleType("pykeops.torch")
        keops_torch.Genred = mock.Mock(side_effect=AssertionError("CPU must not construct Genred"))
        installed = {"extensions": ModuleType("extensions"),
                     "extensions.cauchy": ModuleType("extensions.cauchy"),
                     "extensions.cauchy.cauchy": extension,
                     "pykeops": ModuleType("pykeops"), "pykeops.torch": keops_torch}
        spec = importlib.util.spec_from_file_location("_s4_optional_dispatch_test", ROOT / "TSImputers/S4Model.py")
        isolated = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, installed):
            spec.loader.exec_module(isolated)
        self.assertTrue(isolated.has_cauchy_extension)
        self.assertTrue(isolated.has_pykeops)
        self.assertTrue(callable(isolated.cauchy_slow))
        tensors = example()
        self.assert_operator_and_gradients(isolated.cauchy_resolvent(*tensors), tensors)
        extension.cauchy_mult.assert_not_called()
        keops_torch.Genred.assert_not_called()

    def test_cpu_preserves_noncontiguous_and_lazy_conjugate_inputs(self):
        v, z, w = example(states=8)
        tensors = (v[..., ::2].conj(), z, w[..., ::2].conj())
        self.assertFalse(tensors[0].is_contiguous())
        self.assertTrue(tensors[0].is_conj())
        before = [value.detach().clone() for value in tensors]
        self.assert_operator_and_gradients(S4.cauchy_resolvent(*tensors), tensors)
        for actual, expected in zip(tensors, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_complex128_numerical_gradient(self):
        v, z, w = example()
        tensors = (v[0, :1, :2], z[:3], w[:1, :2])
        self.assertTrue(torch.autograd.gradcheck(S4.cauchy_resolvent, tensors,
                                                fast_mode=True, atol=1e-5, rtol=1e-3))

    def fake_route(self, tensors, *, extension=True, keops=True):
        calls = []
        def backend(name):
            def calculate(v, z, w, **kwargs):
                calls.append((name, kwargs))
                for value in (v, z, w):
                    self.assertTrue(value.is_contiguous())
                    self.assertFalse(value.is_conj())
                return full_state_reference(*(plain(value) for value in (v, z, w)))
            return calculate
        wrapped = tuple(ReportedDeviceTensor(value) for value in tensors)
        with mock.patch.object(S4, "has_cauchy_extension", extension), \
             mock.patch.object(S4, "has_pykeops", keops), \
             mock.patch.object(S4, "cauchy_mult", backend("extension"), create=True), \
             mock.patch.object(S4, "cauchy_conj", backend("keops"), create=True), \
             mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()) as cuda_context:
            result = S4.cauchy_resolvent(*wrapped)
        self.assert_operator_and_gradients(result, wrapped, rtol=3e-5, atol=3e-5)
        return calls, cuda_context

    def test_fake_extension_priority_and_symmetric_flag(self):
        calls, context = self.fake_route(example(torch.complex64, z_grad=False))
        self.assertEqual(calls, [("extension", {"symmetric": True})])
        context.assert_called_with(torch.device("cuda:0"))

    def test_fake_extension_receives_materialized_contiguous_conjugates(self):
        v, z, w = example(torch.complex64, states=8, z_grad=False)
        calls, _ = self.fake_route((v[..., ::2].conj(), z, w[..., ::2].conj()))
        self.assertEqual(calls, [("extension", {"symmetric": True})])

    def test_fake_keops_used_for_complex128_and_for_z_gradients(self):
        for dtype, z_grad in ((torch.complex128, False), (torch.complex64, True),
                              (torch.complex128, True)):
            with self.subTest(dtype=dtype, z_grad=z_grad):
                calls, _ = self.fake_route(example(dtype, z_grad=z_grad))
                self.assertEqual(calls, [("keops", {})])

    def test_fake_extension_unsupported_state_sizes_use_keops(self):
        for states in (1, 3, 1025):
            with self.subTest(states=states):
                calls, _ = self.fake_route(example(torch.complex64, states=states, z_grad=False))
                self.assertEqual(calls, [("keops", {})])

    def test_fake_batched_z_uses_keops_instead_of_extension(self):
        v, z, w = example(torch.complex64, z_grad=False)
        calls, _ = self.fake_route((v, z.unsqueeze(0).expand(3, -1), w))
        self.assertEqual(calls, [("keops", {})])

    def test_fake_unavailable_or_unsupported_acceleration_uses_fallback(self):
        for dtype, extension, z_grad in ((torch.complex64, False, False),
                                        (torch.complex128, True, False),
                                        (torch.complex64, True, True)):
            tensors = tuple(ReportedDeviceTensor(value) for value in example(dtype, z_grad=z_grad))
            def cpu_reference(v, z, w):
                return full_state_reference(*(plain(value) for value in (v, z, w)))
            with self.subTest(dtype=dtype, extension=extension, z_grad=z_grad), \
                 mock.patch.object(S4, "has_cauchy_extension", extension), \
                 mock.patch.object(S4, "has_pykeops", False), \
                 mock.patch.object(S4, "cauchy_mult", create=True,
                                   side_effect=AssertionError("Unsupported extension dispatch")), \
                 mock.patch.object(S4, "cauchy_slow", side_effect=cpu_reference) as fallback:
                result = S4.cauchy_resolvent(*tensors)
            fallback.assert_called_once()
            self.assert_operator_and_gradients(result, tensors, rtol=3e-5, atol=3e-5)

    def test_invalid_dtype_device_state_and_batch_shapes_rejected_before_acceleration(self):
        v, z, w = example()
        cases = [(v.real, z.real, w.real), (v, z.to(torch.complex64), w),
                 (v, z, ReportedDeviceTensor(w, "cuda:1")),
                 (v, z, w[..., :-1]), (v[..., :0], z, w[..., :0]),
                 (v, z[:0], w), (v[0, 0, 0], z, w),
                 (v, z[0], w), (v, z.expand(4, -1), w),
                 tuple(value.to("meta") for value in (v, z, w)),
                 (ReportedDeviceTensor(v), ReportedDeviceTensor(z), ReportedDeviceTensor(w, "cuda:1")),
                 (None, z, w)]
        with mock.patch.object(S4, "has_cauchy_extension", True), \
             mock.patch.object(S4, "has_pykeops", True), \
             mock.patch.object(S4, "cauchy_mult", create=True,
                               side_effect=AssertionError("Invalid input reached extension")), \
             mock.patch.object(S4, "cauchy_conj", create=True,
                               side_effect=AssertionError("Invalid input reached KeOps")):
            for index, tensors in enumerate(cases):
                with self.subTest(case=index), self.assertRaises((TypeError, ValueError)):
                    S4.cauchy_resolvent(*tensors)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_real_cuda_fallback_complex64_and_complex128_forward_and_gradients(self):
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype), \
                 mock.patch.object(S4, "has_cauchy_extension", False), \
                 mock.patch.object(S4, "has_pykeops", False):
                tensors = example(dtype, device="cuda")
                result = S4.cauchy_resolvent(*tensors)
                self.assertEqual(result.device, tensors[0].device)
                tolerance = 3e-5 if dtype == torch.complex64 else 1e-10
                self.assert_operator_and_gradients(result, tensors, rtol=tolerance, atol=tolerance)

    @unittest.skipUnless(torch.cuda.is_available() and S4.has_cauchy_extension,
                         "Actual CUDA Cauchy extension unavailable")
    def test_real_cuda_extension_matches_full_state_and_gradients(self):
        tensors = example(torch.complex64, device="cuda", states=4, z_grad=False)
        with mock.patch.object(S4, "has_pykeops", False), \
             mock.patch.object(S4, "cauchy_mult", wraps=S4.cauchy_mult) as accelerated:
            result = S4.cauchy_resolvent(*tensors)
        accelerated.assert_called_once()
        self.assertEqual(result.device, tensors[0].device)
        self.assert_operator_and_gradients(result, tensors, rtol=3e-4, atol=3e-5)

    @unittest.skipUnless(torch.cuda.is_available() and S4.has_pykeops,
                         "Actual CUDA PyKeOps unavailable")
    def test_real_cuda_keops_both_complex_dtypes_and_z_gradients(self):
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype), \
                 mock.patch.object(S4, "has_cauchy_extension", False), \
                 mock.patch.object(S4, "cauchy_conj", wraps=S4.cauchy_conj) as accelerated:
                tensors = example(dtype, device="cuda")
                result = S4.cauchy_resolvent(*tensors)
                accelerated.assert_called_once()
                self.assertEqual(result.device, tensors[0].device)
                tolerance = 3e-4 if dtype == torch.complex64 else 1e-9
                self.assert_operator_and_gradients(result, tensors, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
