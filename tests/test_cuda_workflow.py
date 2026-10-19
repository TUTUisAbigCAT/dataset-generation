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
            for relative in ("cuda_runtime.py", "training_wavestitch.py", "scripts/run_simulation.py",
                             "scripts/evaluate_denoising.py", "scripts/evaluate_original_sampler.py"):
                self.assertEqual((ROOT / relative).read_bytes(), (destination / relative).read_bytes())

    def test_submit_changes_only_python_line(self):
        old = (ROOT / "third_party/WatchSleepNet/train_transfer_sleepconvnet.sh").read_bytes()
        new = (ROOT / "submit.sh").read_bytes()
        expected = old.replace(b"python modeling/train_transfer_1.py --model=sleepconvnet",
            b"python scripts/run_dreamt.py --epochs 10 --batch-size 8 --threads 4")
        self.assertEqual(new, expected)
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
