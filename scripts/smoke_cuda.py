"""Run the complete CUDA workflow on four explicitly artificial short records."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cuda_runtime import require_cuda, cuda_environment
from create_example_data import create_example_data

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    cli = parser.parse_args()
    device = require_cuda()
    output = (cli.output_dir or ROOT / "runs" / ("cuda_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    create_example_data(output / "artificial_data")
    config = output / "dataset_config.json"
    config.write_text(json.dumps(dict(data_dir=str(output / "artificial_data"), metadata_path=None,
        split_path=None, split_seed=42, multiplier=1.0, downsample_rate=1, max_epochs=None), indent=2), encoding="utf-8")
    workflow = output / "workflow"
    command = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"), "--config", str(config),
        "--epochs", "1", "--batch-size", "2", "--threads", str(cli.threads), "--output-dir", str(workflow)]
    with (output / "workflow.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"CUDA workflow failed: see {output / 'workflow.log'} and stage logs")
    status = read_json(workflow / "status.json")
    training = read_json(workflow / "training/training_report.json")
    validation = read_json(workflow / "training/validation/epoch_0001/metrics.json")
    assert training["validation"]["completed_epochs"] == 1
    assert validation["status"] == "completed"
    assert math.isclose(training["mean_loss"], validation["train_mean_loss"], rel_tol=1e-12)
    denoising = read_json(workflow / "denoising/evaluation_report.json")
    generation = read_json(workflow / "generation/evaluation_report.json")
    devices = {"validation": validation["device"], "training": training["device"], "denoising": denoising["runtime"]["device"],
               "generation": generation["runtime"]["device"]}
    assert status["status"] == "completed", status
    assert all(value == str(device) for value in devices.values()), devices
    assert training["all_losses_finite"] and training["checkpoint_all_finite"]
    assert denoising["status"] == "passed"
    assert generation["status"] == "completed"
    assert generation["result"]["all_finite"] and generation["result"]["condition_columns_bitwise_preserved"]
    assert generation["cuda_execution"]["sampler_source_byte_identical"] is False
    assert generation["cuda_execution"]["upstream_snapshot_byte_identical"] is True
    assert generation["cuda_execution"]["posterior_variance_sqrt_corrected"] is True
    report = dict(status="cuda_workflow_passed", runtime=cuda_environment(device), devices=devices,
        training_updates=training["updates"], epochs=1, artificial_data=True,
        stages=["training", "epoch_validation", "denoising", "corrected_200_step_generation"],
        upstream_snapshot_byte_identical=True, posterior_variance_sqrt_corrected=True,
        no_real_data_or_convergence_claim=True, workflow=str(workflow))
    (output / "cuda_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
