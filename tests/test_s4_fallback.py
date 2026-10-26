"""Check the S4 fallback against full-state mathematics, not source similarity."""
import ast
import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
S4 = importlib.import_module("TSImputers.S4Model")


def actual_fallback():
    # The upstream function is conditionally defined depending on installed
    # accelerators. Compile its exact body so this test also runs on GPU hosts.
    tree = ast.parse((ROOT / "TSImputers/S4Model.py").read_text())
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "cauchy_slow"]
    assert len(functions) == 1
    scope = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<actual_fallback>", "exec"), scope)
    return scope["cauchy_slow"]


def full_pair_reference(v, z, w):
    # Sum two separate rational terms; production expands the state dimension.
    first = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    second = v.conj().unsqueeze(-1) / (z.unsqueeze(-2) - w.conj().unsqueeze(-1))
    return first.sum(-2) + second.sum(-2)


class S4FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_full_conjugate_response_and_broadcast(self):
        torch.manual_seed(2601)
        v = torch.randn(2, 3, 4, 5, dtype=torch.cdouble)
        w = -1-torch.rand(4, 5, dtype=torch.double) + 1j*torch.randn(4, 5)
        # A general complex z avoids incorrectly passing a 2*real shortcut.
        z = 0.3+1j*torch.linspace(0, 8, 17, dtype=torch.double)
        result = actual_fallback()(v, z, w)
        expected = full_pair_reference(v, z, w)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
        legacy = (v.unsqueeze(-1)/(z.unsqueeze(-2)-w.unsqueeze(-1))).sum(-2)
        self.assertGreater(float((legacy-expected).abs().max()), 0.1)

    def test_complex_gradients_match_full_state(self):
        torch.manual_seed(2602)
        v = torch.randn(2, 3, dtype=torch.cdouble, requires_grad=True)
        w = (-1-torch.rand(2, 3, dtype=torch.double)+1j*torch.randn(2, 3)).requires_grad_()
        z = (0.3+1j*torch.linspace(0, 4, 5, dtype=torch.double)).requires_grad_()
        fn = actual_fallback()
        self.assertTrue(torch.autograd.gradcheck(fn, (v, z, w), fast_mode=True))
        result = fn(v, z, w)
        reference = full_pair_reference(v, z, w)
        got = torch.autograd.grad(result.abs().square().sum(), (v, z, w))
        wanted = torch.autograd.grad(reference.abs().square().sum(), (v, z, w))
        for actual, expected in zip(got, wanted):
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_s4_kernel_matches_independent_dense_recurrence(self):
        torch.manual_seed(42)
        length, states, features = 32, 8, 2
        kernel = S4.HippoSSKernel(H=features, N=states, L=length, channels=1, precision=2)
        with mock.patch.object(S4, "has_cauchy_extension", False), \
             mock.patch.object(S4, "has_pykeops", False), \
             mock.patch.object(S4, "cauchy_slow", actual_fallback(), create=True):
            actual = kernel.kernel(L=length)[0].detach().numpy()
        k = kernel.kernel
        def full(tensor):
            half = torch.view_as_complex(tensor).detach().numpy()
            return np.concatenate([half, half.conj()], axis=-1)
        w, p, q, b, c = full(k.w), full(k.P), full(k.Q), full(k.B)[0], full(k.C)
        dt = np.exp(k.log_dt.detach().numpy())
        reference = np.empty_like(actual)
        for h in range(features):
            a = np.diag(w[h])-np.einsum("rn,rm->nm", p[:, h], q[:, h])
            identity = np.eye(states, dtype=np.complex128)
            lhs = identity-0.5*dt[h]*a
            ad = np.linalg.solve(lhs, identity+0.5*dt[h]*a)
            bd = np.linalg.solve(lhs, dt[h]*b[h])
            # The stored C already includes the finite-length correction.
            cd = np.linalg.solve((identity-np.linalg.matrix_power(ad, length)).T, c[:, h].T).T
            state = bd
            for index in range(length):
                value = cd@state
                self.assertLess(float(np.abs(value.imag).max()), 1e-10)
                reference[:, h, index] = value.real
                state = ad@state
        np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_fallback_remains_on_cuda(self):
        v = torch.randn(2, 3, dtype=torch.cfloat, device="cuda", requires_grad=True)
        w = -torch.ones_like(v)+1j*torch.randn_like(v)
        z = 1j*torch.arange(7, dtype=torch.float, device="cuda")
        result = actual_fallback()(v, z, w)
        self.assertEqual(result.device.type, "cuda")
        torch.testing.assert_close(result, full_pair_reference(v, z, w))
        result.abs().square().sum().backward()
        self.assertEqual(v.grad.device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(v.grad).all()))


if __name__ == "__main__":
    unittest.main()
