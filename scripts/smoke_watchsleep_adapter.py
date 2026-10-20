"""Run five unchanged official optimizer updates on four-key DREAMT fixtures.

Copies source into work/, never writes a checkpoint into the deliverable or an
ongoing experiment. The fixture is artificial and tests integration only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_example_data import create_example_data
from cuda_runtime import require_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work/dreamt_loader_v4_smoke")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/loader_smoke")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Generate and validate the artificial fixture without model execution")
    args = parser.parse_args()
    if not args.prepare_only:
        require_cuda()
    work, output = args.work_dir.resolve(), args.output_dir.resolve()
    if work.exists() or output.exists():
        raise FileExistsError("Smoke test refuses to overwrite an existing work or output directory")
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    names = ("data_utils.py", "watchsleep_loader.py", "training_wavestitch.py", "training_utils.py", "cuda_runtime.py",
             "epoch_validation.py", "validation_metrics.py", "validation_sampling.py", "window_config.py")
    for name in names:
        shutil.copyfile(ROOT / name, work / name)
    for name in ("TSImputers", "utils", "third_party"):
        shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (work / "scripts").mkdir()
    for name in ("run_simulation.py", "verify_upstream_boundary.py", "evaluate_original_sampler.py"):
        shutil.copyfile(ROOT / "scripts" / name, work / "scripts" / name)
    fixture = work / "fixture"
    fixture_info = create_example_data(fixture, fixed_splits=True)
    config = dict(data_dir=str(fixture), metadata_path=str(fixture / "participant_info.csv"),
                  split_path=str(fixture / "splits.json"), generate_splits=False,
                  multiplier=1., downsample_rate=1, max_epochs=None, split_seed=42)
    (work / "dataset_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.prepare_only:
        from data_utils import Preprocessor
        processor = Preprocessor("DREAMT_SIM", config_override=config)
        dataset = processor.make_training_dataset(3000, 750)
        assert len(dataset) == 10 and tuple(dataset[0].shape) == (3000, 10)
        validation = processor.select_validation_windows()
        assert len(validation) == 2
        result = dict(status="prepared_no_model_execution", synthetic_fixture=True,
            fixture=fixture_info, training_windows=len(dataset), validation_windows=len(validation),
            planned_optimizer_updates=5, planned_batch_size=2, input_shape=[2, 3000, 10],
            data_identity=processor.data_identity(), test_npz_opened=False,
            work_directory=str(work))
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "training_windows", "validation_windows", "input_shape")}, indent=2))
        return
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("DREAMT_"):
            environment.pop(key)
    environment.update(DREAMT_CONFIG_PATH=str(work / "dataset_config.json"),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-B", "-u", str(work / "scripts/run_simulation.py"),
               "--batch-size", "2", "--epochs", "1", "--log-every", "1", "--output-dir", str(output / "training")]
    with (output / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    training = read(output / "training/training_report.json")
    manifest = read(output / "training/run_manifest.json")
    assert training["status"] == "completed" and training["updates"] == 5
    assert training["all_losses_finite"] and training["checkpoint_all_finite"] and training["source_unchanged"]
    assert manifest["train_windows"] == 10 and len(manifest["model_input_columns"]) == 10
    assert manifest["data"]["test_npz_opened"] is False
    assert all(r["continuity_source"] == "stored_order_only" for r in manifest["data"]["record_alignment"].values())
    copied_hashes = {name: hashlib.sha256((work / name).read_bytes()).hexdigest() for name in names}
    assert all(copied_hashes[name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names)
    result = dict(status="passed", synthetic_fixture=True, fixture_npz_keys=["data", "stages", "fs", "ahi"],
        fixture=fixture_info,
        original_epoch_indices_supplied=False, continuity_claim="stored order only; this fixture has known artificial origin",
        original_training_loop=True, optimizer_updates=5, batch_size=2, epochs=1, input_shape=[2, 3000, 10],
        output_channels=1, training=training, source_copy_sha256=copied_hashes,
        default_deliverable_checkpoint_created=(ROOT / "saved_models/DREAMT_SIM/model.pth").exists(),
        ongoing_experiment_modified=False, quality_or_convergence_claim=False,
        command=command, work_directory=str(work), checkpoint=str(work / "saved_models/DREAMT_SIM/model.pth"))
    assert result["default_deliverable_checkpoint_created"] is False
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "optimizer_updates", "input_shape", "fixture_npz_keys", "quality_or_convergence_claim")}, indent=2))


if __name__ == "__main__":
    main()
