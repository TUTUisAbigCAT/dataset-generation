"""Numerical S4 preflight. Requested CUDA must execute, never skip or use CPU.

Example: python scripts/check_s4_backends.py --device cuda --output s4_cuda_check.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
S4 = importlib.import_module("TSImputers.S4Model")


def reference_resolvent(v, z, w):
    first = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    second = v.conj().unsqueeze(-1) / (z.unsqueeze(-2) - w.conj().unsqueeze(-1))
    return first.sum(-2) + second.sum(-2)


def check_resolvent(device, dtype, backend, z_grad):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    rng = torch.Generator(device=device).manual_seed(260426)
    v = torch.randn(2, 3, 4, dtype=dtype, device=device, generator=rng).requires_grad_()
    w = (-1 - torch.rand(3, 4, dtype=real_dtype, device=device, generator=rng)
         + 1j * torch.randn(3, 4, dtype=real_dtype, device=device, generator=rng)).requires_grad_()
    z = (.3 + 1j * torch.linspace(0, 5, 9, dtype=real_dtype, device=device)).requires_grad_(z_grad)
    with ExitStack() as stack:
        if backend == "torch":
            stack.enter_context(mock.patch.object(S4, "has_cauchy_extension", False))
            stack.enter_context(mock.patch.object(S4, "has_pykeops", False))
        elif backend == "cuda_extension":
            stack.enter_context(mock.patch.object(S4, "has_pykeops", False))
            called = stack.enter_context(mock.patch.object(S4, "cauchy_mult", wraps=S4.cauchy_mult))
        elif backend == "pykeops":
            stack.enter_context(mock.patch.object(S4, "has_cauchy_extension", False))
            called = stack.enter_context(mock.patch.object(S4, "cauchy_conj", wraps=S4.cauchy_conj))
        actual = S4.cauchy_resolvent(v, z, w)
        if backend in ("cuda_extension", "pykeops"):
            called.assert_called_once()
    reference = reference_resolvent(v, z, w)
    tolerance = 3e-4 if dtype == torch.complex64 else 1e-9
    if actual.device != v.device or actual.dtype != dtype:
        raise AssertionError("Cauchy backend changed device or precision")
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    inputs = tuple(t for t in (v, z, w) if t.requires_grad)
    got = torch.autograd.grad(actual.abs().square().sum(), inputs)
    wanted = torch.autograd.grad(reference.abs().square().sum(), inputs)
    for grad, expected, param in zip(got, wanted, inputs):
        if grad.device != param.device or grad.dtype != param.dtype:
            raise AssertionError("Cauchy backward changed device or precision")
        torch.testing.assert_close(grad, expected, rtol=tolerance, atol=tolerance)
    return dict(device=str(device), dtype=str(dtype), backend=backend, z_gradient=z_grad,
                output_max_absolute_error=(actual - reference).abs().max().item(),
                gradient_max_absolute_error=max((a - b).abs().max().item() for a, b in zip(got, wanted)),
                status="passed")


def dense_kernel_reference(kernel, length):
    """Independent bilinear dense-state recurrence, executed on kernel.device."""
    def full(value):
        half = torch.view_as_complex(value.detach())
        return torch.cat((half, half.conj()), dim=-1)
    w, p, q, b, c = (full(value) for value in (kernel.w, kernel.P, kernel.Q, kernel.B, kernel.C))
    b = b[0]
    dt = kernel.log_dt.detach().exp()
    result = []
    for h in range(w.shape[0]):
        a = torch.diag(w[h]) - torch.einsum("rn,rm->nm", p[:, h], q[:, h])
        identity = torch.eye(w.shape[-1], dtype=w.dtype, device=w.device)
        lhs = identity - .5 * dt[h] * a
        ad = torch.linalg.solve(lhs, identity + .5 * dt[h] * a)
        state = torch.linalg.solve(lhs, dt[h] * b[h])
        cd = torch.linalg.solve((identity - torch.linalg.matrix_power(ad, length)).T, c[:, h].T).T
        output = []
        for _ in range(length):
            output.append((cd @ state).real)
            state = ad @ state
        result.append(torch.stack(output, dim=-1))
    return torch.stack(result, dim=1)


def check_kernel(device, precision):
    torch.manual_seed(42)
    length = 32
    model = S4.HippoSSKernel(H=2, N=8, L=length, channels=1,
                           precision=precision, trainable=True).to(device)
    actual = model.kernel(L=length)[0]
    reference = dense_kernel_reference(model.kernel, length)
    tolerance = 3e-4 if precision == 1 else 1e-9
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    if actual.device != next(model.parameters()).device:
        raise AssertionError("S4 kernel output moved devices")
    actual.square().sum().backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    for name, grad in gradients.items():
        if grad is None or not torch.isfinite(grad).all():
            raise AssertionError(f"Missing or nonfinite S4 gradient: {name}")
        if grad.device != actual.device:
            raise AssertionError(f"S4 gradient moved devices: {name}")
    return dict(device=str(device), precision=precision, status="passed",
                dense_recurrence_max_absolute_error=(actual - reference).abs().max().item(),
                dense_recurrence_relative_l2=((actual - reference).norm() / reference.norm()).item(),
                parameters_with_finite_gradients=len(gradients))


def run_checks(device_choice="cpu", cuda_index=0):
    if device_choice in ("cuda", "both") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU substitution or skipped CUDA pass")
    if device_choice not in ("cpu", "cuda", "both"):
        raise ValueError("device must be cpu, cuda or both")
    devices = ([torch.device("cpu")] if device_choice in ("cpu", "both") else [])
    if device_choice in ("cuda", "both"):
        devices.append(torch.device("cuda", cuda_index))
    report = dict(status="running", requested_device=device_choice, torch_version=torch.__version__,
                  cuda_available=torch.cuda.is_available(), cuda_extension_imported=S4.has_cauchy_extension,
                  pykeops_imported=S4.has_pykeops, resolvent_checks=[], kernel_checks=[])
    for device in devices:
        for dtype in (torch.complex64, torch.complex128):
            for backend in ("torch", "auto"):
                for z_grad in (False, True):
                    report["resolvent_checks"].append(check_resolvent(device, dtype, backend, z_grad))
            if device.type == "cuda" and S4.has_pykeops:
                report["resolvent_checks"].append(check_resolvent(device, dtype, "pykeops", True))
        if device.type == "cuda" and S4.has_cauchy_extension:
            report["resolvent_checks"].append(check_resolvent(device, torch.complex64, "cuda_extension", False))
        for precision in (1, 2):
            report["kernel_checks"].append(check_kernel(device, precision))
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.cuda_index < 0:
        parser.error("threads must be positive; cuda-index must be nonnegative")
    torch.set_num_threads(args.threads)
    try:
        report = run_checks(args.device, args.cuda_index)
    except Exception as exc:
        report = dict(status="failed", requested_device=args.device,
                      error_type=type(exc).__name__, error=str(exc))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
