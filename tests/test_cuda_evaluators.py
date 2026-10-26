"""CUDA entry-point contract checks; real device execution is skipped without a GPU."""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_denoising, evaluate_original_sampler
from tests.test_posterior_noise import independent_corrected_pipeline


class CudaEvaluatorPolicyTests(unittest.TestCase):
    def test_denoising_fails_before_reading_checkpoint_without_cuda(self):
        argv = ["evaluate_denoising.py", "--run-manifest", "missing-manifest.json"]
        with mock.patch.object(sys, "argv", argv), mock.patch("cuda_runtime.require_cuda", side_effect=RuntimeError("CUDA required")):
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                evaluate_denoising.main()

    def test_sampler_fails_before_reading_checkpoint_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["evaluate_original_sampler.py", "--training-args-json", "missing-manifest.json",
                    "--output-dir", str(Path(tmp) / "evaluation")]
            with mock.patch.object(sys, "argv", argv), mock.patch("cuda_runtime.require_cuda", side_effect=RuntimeError("CUDA required")):
                with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                    evaluate_original_sampler.main()

    def test_sampler_retains_exact_upstream_excerpts(self):
        noise, generation, compiled, provenance = evaluate_original_sampler.extract_original_pipeline()
        self.assertTrue(provenance["excerpts_byte_identical_to_upstream"])
        self.assertFalse(provenance["sampler_expressions_edited"])
        self.assertFalse(provenance["random_draws_added_or_removed"])
        self.assertEqual(compiled, "if True:\n    if True:\n" + generation)
        self.assertIn("torch.normal(0, 1,", noise)
        compile(noise, "upstream_noise", "exec")
        compile(compiled, "upstream_generation", "exec")

    def test_corrected_sampler_changes_only_the_posterior_noise_coefficient(self):
        noise, generation, compiled, provenance = evaluate_original_sampler.extract_corrected_pipeline()
        expected_noise, expected_compiled = independent_corrected_pipeline()
        self.assertEqual(noise, expected_noise)
        self.assertEqual(ast.dump(ast.parse(compiled)), ast.dump(ast.parse(expected_compiled)))
        self.assertEqual(compiled, "if True:\n    if True:\n" + generation)
        self.assertTrue(provenance["upstream_snapshot_byte_identical"])
        self.assertFalse(provenance["excerpts_byte_identical_to_upstream"])
        self.assertTrue(provenance["sampler_expressions_edited"])
        self.assertFalse(provenance["original_variance_times_noise_preserved"])
        self.assertFalse(provenance["random_draws_added_or_removed"])
        self.assertTrue(provenance["original_guidance_minus_0_1_gradient_preserved"])
        self.assertEqual(provenance["posterior_noise_scale"], "sqrt_posterior_variance")

    def test_sampler_exec_uses_local_cuda_context(self):
        tree = ast.parse(Path(evaluate_original_sampler.__file__).read_text(encoding="utf-8"))
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            contexts = [ast.unparse(item.context_expr) for item in node.items]
            calls = [ast.unparse(item) for item in ast.walk(node) if isinstance(item, ast.Call)]
            if "torch.device(device)" in contexts and any("exec(compile(compiled_source" in call for call in calls):
                matches.append(node)
        self.assertEqual(len(matches), 1)

    def test_denoising_forward_uses_local_cuda_context(self):
        tree = ast.parse(Path(evaluate_denoising.__file__).read_text(encoding="utf-8"))
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                   and "torch.device(device)" in [ast.unparse(item.context_expr) for item in node.items]
                   and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id == "model" for call in ast.walk(node))]
        self.assertEqual(len(matches), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; not a GPU execution result")
    def test_corrected_sampler_tensor_factories_execute_on_cuda(self):
        from cuda_runtime import require_cuda
        device = require_cuda()
        noise, _, compiled, _ = evaluate_original_sampler.extract_corrected_pipeline()
        args = SimpleNamespace(window_size=6, stride=3)
        windows = torch.zeros(2, 6, 2, device=device)
        windows[:, :, 1] = 1
        masks = torch.ones(2, 6, dtype=torch.bool, device=device)
        seen = []

        class Denoiser(torch.nn.Module):
            def forward(self, x, t):
                seen.append((x.device, t.device))
                return (x[:, :, :1] * 0.05).permute(0, 2, 1)

        beta = torch.tensor([[0.0001], [0.01], [0.02]], device=device)
        scope = dict(torch=torch, sqrt=torch.sqrt, args=args, device=device, model=Denoiser(),
                     diffusion_config={"T": 3, "betas": beta, "alphas": 1 - beta,
                                       "alpha_bars": torch.cumprod(1 - beta, dim=0)},
                     test_dataset=SimpleNamespace(inputs=windows), test_dataloader=[windows],
                     mask_dataloader=[masks], hierarchical_column_indices=torch.tensor([1], device=device),
                     non_hier_cols=torch.tensor([0], device=device), num_ops=0, trial=0)
        exec(compile(noise, "upstream_noise", "exec"), scope)
        with torch.device(device), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(compiled, "upstream_generation", "exec"), scope)
        self.assertEqual(scope["synth_tensor"].device, device)
        self.assertEqual(scope["num_ops"], 3)
        self.assertTrue(all(x == device and t == device for x, t in seen))
        for name in ("x", "times", "sampled_noise", "normal_denoising", "grad", "eps"):
            self.assertEqual(scope[name].device, device, name)
        self.assertTrue(torch.equal(scope["synth_tensor"][:, 1], torch.ones(9, device=device)))
        self.assertTrue(bool(torch.isfinite(scope["synth_tensor"]).all()))


if __name__ == "__main__":
    unittest.main()
