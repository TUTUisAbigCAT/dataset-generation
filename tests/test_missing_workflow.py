"""Exercise real adapter/loop/S4 logic with explicit CPU unit-test tensors.

These are implementation checks, not a CUDA training-quality experiment.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor
from missing_data import masked_noise_mse
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from scripts.run_missing_value_ablation import arm_config, compare_preflights
from scripts import run_missing_value_ablation as ablation
from training_utils import fetchModel
from diffusion_schedule import build_diffusion_config


def fixture(directory):
    shutil.copytree(ROOT / "example_data", directory)
    for path in directory.glob("*.npz"):
        with np.load(path, allow_pickle=False) as stored:
            values = {key: stored[key].copy() for key in stored.files}
        flat = values["data"].reshape(-1)
        # Variable coverage in different windows catches wrong loss weighting.
        for epoch, start in enumerate(range(0, len(flat), 750)):
            flat[start:start + 10 + epoch*3] = 0
        np.savez(path, **values)


class MissingWorkflowTests(unittest.TestCase):
    def test_pair_completion_report_with_explicitly_mocked_gpu_subprocesses(self):
        # Real preflights; fake training outputs exercise final report plumbing.
        # This test is not a substitute for either production GPU run.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fixture(base / "data")
            config = base / "config.json"
            config.write_text(json.dumps(dict(data_dir="data", window_epochs=1)), encoding="utf-8")
            output = base / "ab"
            real_run = subprocess.run
            def fake_run(command, **kwargs):
                if "--check-only" in command:
                    return real_run(command, **kwargs)
                target = Path(command[command.index("--output-dir")+1])
                frozen = json.loads(Path(command[command.index("--config")+1]).read_text())
                policy = frozen["missing_value_policy"]
                target.mkdir()
                shutil.copy2(output / f"preflight_{policy}/preflight.json", target / "preflight.json")
                training = target / "training"
                (training / "validation/epoch_0002").mkdir(parents=True)
                (training / "run_manifest.json").write_text(json.dumps(dict(updates=4)), encoding="utf-8")
                (training / "epoch_metrics.csv").write_text("epoch,gaussian_kl_real_to_generated\n1,2.0\n2,1.0\n", encoding="utf-8")
                (training / "validation/epoch_0002/metrics.json").write_text(json.dumps(dict(
                    diagnostics=dict(generated_nonpositive_fraction=.125))), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(ablation.subprocess, "run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                ablation.main(["--config", str(config), "--output-dir", str(output), "--epochs", "2"])
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["arms"]["masked"]["updates"], 4)
            self.assertEqual(result["arms"]["masked"]["generated_nonpositive_fraction"], .125)
            self.assertEqual(result["arms"]["preserve"]["best_kl_epoch"], 2)
            self.assertIn("generated_nonpositive_fraction", (output / "comparison.csv").read_text())

    def test_pair_launcher_freezes_same_reference_windows_and_different_scalers(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fixture(base / "data")
            config = base / "config.json"
            config.write_text(json.dumps(dict(data_dir="data", window_epochs=2)), encoding="utf-8")
            output = base / "ab"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_missing_value_ablation.py"),
                "--config", str(config), "--output-dir", str(output), "--check-only"],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "checked")
            self.assertFalse(summary["training_completed"])
            self.assertTrue(all(summary["shared_input_checks"].values()))
            a, b = summary["arms"]["preserve"], summary["arms"]["masked"]
            self.assertGreater(b["scaler"]["signal_mean_seconds"], a["scaler"]["signal_mean_seconds"])
            self.assertEqual(a["training_windows"], b["training_windows"])
            self.assertEqual(a["command"][-2:], ["--output-dir", str(output / "preserve")])
            for policy in ("preserve", "masked"):
                frozen = json.loads((output / f"config_{policy}.json").read_text())
                self.assertEqual(frozen["validation"]["reference_mask_policy"], "observed_only")
                self.assertEqual(frozen["missing_value_policy"], policy)
                self.assertFalse(frozen["early_stopping"]["enabled"])

    def test_actual_masked_training_loop_updates_real_s4_and_weights_observed_points(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            fixture(data)
            processor = Preprocessor("DREAMT", config_override=dict(data_dir=str(data), window_epochs=1,
                                                                  missing_value_policy="masked"))
            dataset = processor.make_training_dataset(750, 750)
            batches = [next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [0, 1]), batch_size=2))),
                       next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [2]), batch_size=1)))]
            args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS, window_size=750, s4_lmax=750, epochs=2,
                res_channels=8, skip_channels=8, num_res_layers=1, s4_dstate=8,
                diff_step_embed_in=8, diff_step_embed_mid=8, diff_step_embed_out=8,
                timesteps=4, diffusion_schedule="linear_power", terminal_alpha_bar=1e-8))
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
                    torch.manual_seed(42)
                    model = fetchModel(10, 1, args)
                    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    observed_losses, reports = [], []
                    def tracked(predicted, target, mask):
                        loss, total, count = masked_noise_mse(predicted, target, mask)
                        observed_losses.append((float(total), count))
                        return loss, total, count
                    def validate(epoch, live_model, mean_loss):
                        recent = observed_losses[-len(batches):]
                        self.assertAlmostEqual(mean_loss, sum(v for v, n in recent)/sum(n for v, n in recent))
                        self.assertTrue(all(torch.isfinite(p).all() for p in live_model.parameters()))
                        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in live_model.parameters()))
                        reports.append(mean_loss)
                    source = ast.parse((ROOT / "training_wavestitch.py").read_text())
                    loop = next(n for n in ast.walk(source) if isinstance(n, ast.For)
                                and isinstance(n.target, ast.Name) and n.target.id == "epoch")
                    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "actual_masked_loop", "exec")
                    scope = dict(torch=torch, np=np, model=model, optimizer=optimizer, criterion=torch.nn.MSELoss(),
                        device=torch.device("cpu"), args=args, dataloader=batches,
                        diffusion_config=build_diffusion_config(args), non_hier_cols=torch.tensor([0]),
                        randint=torch.randint, normal=torch.normal, sqrt=torch.sqrt, masked_noise_mse=tracked,
                        training_input_dropout=lambda batch: batch,
                        validator=SimpleNamespace(run=validate, should_stop=False))
                    exec(code, scope)
                self.assertEqual(len(reports), 2)
                self.assertEqual(len(observed_losses), 4)
                self.assertTrue(any(not torch.equal(initial[name], p) for name, p in model.named_parameters()))
                self.assertTrue(all(item["data"].shape[2] == 10 for item in batches))
                self.assertTrue(all(torch.all(item["data"][:, :, 0][~item["observed_mask"]] == 0) for item in batches))
            finally:
                torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
