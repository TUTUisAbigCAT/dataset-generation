"""Check orchestration policy; hardware tests explicitly require CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("workflow_runner", ROOT / "scripts/run_dreamt.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class CudaWorkflowTests(unittest.TestCase):
    def test_snapshot_keeps_runtime_policy_and_evaluators(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "workspace"
            runner.snapshot_source(destination)
            for relative in ("cuda_runtime.py", "epoch_validation.py", "validation_metrics.py", "validation_sampling.py", "training_wavestitch.py", "scripts/run_simulation.py",
                             "scripts/evaluate_denoising.py", "scripts/evaluate_original_sampler.py"):
                self.assertEqual((ROOT / relative).read_bytes(), (destination / relative).read_bytes())

    def test_submit_keeps_workflow_entry_and_unix_line_endings(self):
        # Cluster directives now follow the user's supplied submission script.
        new = (ROOT / "submit.sh").read_bytes()
        commands = [line for line in new.splitlines() if line.startswith(b"python ")]
        self.assertEqual(commands, [b"python scripts/run_dreamt.py --epochs 50 --batch-size 8 --threads 4"])
        self.assertNotIn(b"\r", new)

    def test_production_entry_points_reject_missing_cuda(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as temp:
            commands = [
                ["scripts/run_dreamt.py", "--config", str(Path(temp)/"missing.json"), "--output-dir", str(Path(temp)/"run")],
                ["scripts/run_simulation.py", "--dataset", "DREAMT", "--epochs", "1", "--output-dir", str(Path(temp)/"training")],
                ["scripts/check_cuda.py"],
                ["scripts/smoke_watchsleep_adapter.py", "--work-dir", str(Path(temp)/"adapter_work"), "--output-dir", str(Path(temp)/"adapter_output")],
                ["scripts/smoke_cuda.py", "--output-dir", str(Path(temp)/"smoke")],
            ]
            for command in commands:
                with self.subTest(script=command[0]):
                    result = subprocess.run([sys.executable, "-B", *command], cwd=ROOT, env=env,
                                            capture_output=True, text=True, timeout=45)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("CUDA is required", result.stderr)
                    self.assertIn("CPU fallback is disabled", result.stderr)
            status = json.loads((Path(temp)/"run/status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stages"], [])
            self.assertFalse((Path(temp)/"run/workspace").exists())
            self.assertFalse((Path(temp)/"training").exists())
            self.assertFalse((Path(temp)/"smoke").exists())

    def test_configured_output_root_from_foreign_cwd_and_cli_override(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            config_dir, caller = base / "config", base / "caller"
            config_dir.mkdir()
            caller.mkdir()
            config_path = config_dir / "dataset_config.json"
            config = json.loads((ROOT / "dataset_config.json").read_text(encoding="utf-8-sig"))
            config["data_dir"] = str(ROOT / "example_data")
            command = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"),
                       "--config", str(config_path), "--check-only"]
            cases = [
                ("relative", "relative runs", [], config_dir / "relative runs"),
                ("absolute", str(base / "absolute runs"), [], base / "absolute runs"),
                ("cli", "unused runs", ["--output-dir", "exact_run"], caller),
            ]
            for name, output_root, extra, expected_parent in cases:
                with self.subTest(case=name):
                    config["output_root"] = output_root
                    config_path.write_text(json.dumps(config), encoding="utf-8-sig")
                    result = subprocess.run(command + extra, cwd=caller, capture_output=True,
                                            text=True, timeout=45)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    summary = json.loads(result.stdout.strip().splitlines()[-1])
                    output = Path(summary["output_directory"])
                    self.assertEqual(output.parent, expected_parent.resolve())
                    if name == "cli":
                        self.assertEqual(output.name, "exact_run")
                    else:
                        self.assertRegex(output.name, r"^dreamt_[0-9]{8}T[0-9]{6}_[0-9]{6}Z$")
                    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                    self.assertEqual(status["status"], "checked")
                    self.assertEqual(status["output_directory"], str(output))
                    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
                    self.assertEqual(preflight["training_input_shape"], [3000, 10])
            self.assertFalse((config_dir / "unused runs").exists())
            saved_status = (caller / "exact_run/status.json").read_bytes()
            repeated = subprocess.run(command + ["--output-dir", "exact_run"], cwd=caller,
                                      capture_output=True, text=True, timeout=45)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("FileExistsError", repeated.stderr)
            self.assertEqual((caller / "exact_run/status.json").read_bytes(), saved_status)

    def test_dependency_profile_uses_cuda_torch(self):
        dependencies = (ROOT / "requirements_cuda.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.2.2+cu121", dependencies)
        self.assertNotIn("+cpu", dependencies)
        self.assertFalse((ROOT / "requirements_cpu.txt").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "Real CUDA end-to-end verification needs a GPU allocation")
    def test_full_cuda_workflow_on_artificial_records(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "cuda_smoke"
            result = subprocess.run([sys.executable, "-B", "scripts/smoke_cuda.py", "--output-dir", str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=1800)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "cuda_smoke_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "cuda_workflow_passed")
            self.assertTrue(all(device.startswith("cuda:") for device in report["devices"].values()))

if __name__ == "__main__":
    unittest.main()
