"""Re-export the paired run's FINAL validation samples without further training.

Recorded model/data identities and the original validation protocol are checked
before sampling. CUDA is mandatory. Existing best checkpoints are not selected.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cuda_runtime import require_cuda, assert_cuda_model, cuda_environment
from data_utils import Preprocessor
from diffusion_schedule import build_diffusion_config
from epoch_validation import EpochValidator
from scripts.evaluate_denoising import load_training_run, bind_training_data, resolve_checkpoint
from scripts.evaluate_original_sampler import load_original_parameters
from training_utils import fetchModel
from validation_metrics import preserve_random_state

ARMS = ("preserve", "masked")
RUNTIME_SOURCES = (
    "training_utils.py", "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py",
    "utils/util.py", "data_utils.py", "watchsleep_loader.py", "window_config.py",
    "diffusion_schedule.py", "validation_sampling.py", "scripts/evaluate_original_sampler.py",
)
PROTOCOL_FIELDS = (
    "window", "windows", "selection", "scaler", "split_fingerprint", "condition_columns",
    "generation", "diffusion_identity", "diffusion_steps", "initialization_seed",
    "validation_seed_reset_each_epoch", "training_rng_restored", "signal_unit", "variance_unit",
    "variance_correction", "gaussian_kl", "metric_dtype", "test_npz_opened",
    "missing_value_policy", "reference_mask_policy", "reference_mask",
)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _options(value):
    """Artifact serialization is the only allowed change to validation options."""
    return {key: item for key, item in value.items() if key != "save_artifacts"}


def _source_hashes(manifest):
    hashes = {}
    for key, value in manifest.get("source_sha256", {}).items():
        name = key.replace("\\", "/")
        if name in hashes and hashes[name] != value:
            raise ValueError("Conflicting source hash aliases: "+name)
        hashes[name] = value
    return hashes


def compare_protocols(expected, actual):
    """Reject changed selection, masks, conditions, scale, schedule or scoring."""
    checks = {key: key in expected and key in actual and expected[key] == actual[key]
              for key in PROTOCOL_FIELDS}
    checks["options"] = _options(expected["options"]) == _options(actual["options"])
    if expected.get("test_npz_opened") is not False or actual.get("test_npz_opened") is not False:
        checks["test_npz_opened"] = False
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Reconstructed validation protocol differs: "+", ".join(failed))
    return checks


def load_reference_pair(reference_run):
    """Read both original final checkpoints before opening any signal data."""
    root = Path(reference_run).expanduser().resolve()
    arms = {}
    for arm in ARMS:
        manifest_path = root/arm/"training/run_manifest.json"
        args, manifest, config_provenance = load_training_run(manifest_path)
        epoch = _positive_integer(manifest.get("checkpoint_epoch"), "checkpoint_epoch")
        updates = _positive_integer(manifest.get("updates"), "updates")
        batches = _positive_integer(manifest.get("batches_per_epoch"), "batches_per_epoch")
        if (manifest.get("status") != "completed" or manifest.get("checkpoint_selection") is not None
                or epoch != manifest.get("completed_epochs") or epoch != args.epochs
                or updates != epoch*batches):
            raise ValueError(f"{arm}: require a completed FINAL checkpoint and complete update count")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(manifest.get("checkpoint_sha256", ""))):
            raise ValueError(f"{arm}: final checkpoint must have its recorded SHA256")
        identity = manifest.get("data_identity", manifest.get("data", {}).get("data_identity"))
        if not isinstance(identity, dict) or identity.get("missing_value_policy", "preserve") != arm:
            raise ValueError(f"{arm}: missing-value identity does not match the requested arm")
        protocol_path = root/arm/"training/validation/protocol.json"
        protocol = read_json(protocol_path)
        if (protocol.get("missing_value_policy") != arm or protocol.get("reference_mask_policy") != "observed_only"
                or protocol.get("test_npz_opened") is not False):
            raise ValueError(f"{arm}: original validation must use observed_only and sealed test data")
        if protocol.get("scaler") != identity.get("scaler") or protocol.get("diffusion_identity") != manifest.get("diffusion_identity"):
            raise ValueError(f"{arm}: original protocol disagrees with the training identity")
        splits = identity.get("splits", {})
        if set(splits) != {"train", "val", "test"}:
            raise ValueError(f"{arm}: complete subject splits are required")
        groups = {key:set(value) for key,value in splits.items()}
        if any(groups[a] & groups[b] for a,b in (("train","val"),("train","test"),("val","test"))):
            raise ValueError(f"{arm}: overlapping subject splits")
        windows = protocol.get("windows")
        if not isinstance(windows, list) or not windows or any(w.get("filename") not in groups["val"] for w in windows):
            raise ValueError(f"{arm}: original windows must belong only to validation subjects")
        config_path = root/arm/"workspace/dataset_config.json"
        config = read_json(config_path)
        recorded = _source_hashes(manifest)
        if sha256(config_path) != recorded.get("dataset_config.json"):
            raise ValueError(f"{arm}: frozen configuration hash changed")
        if (config.get("missing_value_policy", "preserve") != arm
                or _options(config.get("validation", {})) != _options(protocol["options"])):
            raise ValueError(f"{arm}: frozen validation configuration differs from original protocol")
        source_checks = {}
        for name in RUNTIME_SOURCES:
            actual = sha256(ROOT/name)
            source_checks[name] = dict(recorded=recorded.get(name), runtime=actual,
                                      matched=actual == recorded.get(name))
        if not all(item["matched"] for item in source_checks.values()):
            raise ValueError(f"{arm}: model, loader or sampler source differs from recorded training source")
        # Explicit relocation supports downloaded copies without modifying the
        # recorded manifest or ever choosing best_kl_model.pth.
        checkpoint, checkpoint_identity = resolve_checkpoint(
            manifest_path, manifest, root/arm/f"workspace/saved_models/{args.dataset}/model.pth")
        old_metrics_path = root/arm/f"training/validation/epoch_{epoch:04d}/metrics.json"
        old_metrics = read_json(old_metrics_path)
        old_kl = old_metrics.get("aggregate", {}).get("gaussian_kl_real_to_generated")
        if (old_metrics.get("epoch") != epoch or old_metrics.get("status") != "completed"
                or isinstance(old_kl, bool) or not isinstance(old_kl, (int,float)) or not math.isfinite(old_kl)):
            raise ValueError(f"{arm}: complete original final validation metrics are required")
        arms[arm] = dict(args=args, manifest=manifest, manifest_path=manifest_path, identity=identity,
            config=config, protocol=protocol, protocol_path=protocol_path,
            checkpoint=checkpoint, checkpoint_identity=checkpoint_identity,
            config_provenance=config_provenance, source_checks=source_checks,
            epoch=epoch, updates=updates, old_kl=float(old_kl))
    a,b = (arms[arm] for arm in ARMS)
    checks = {key: a[key] == b[key] for key in ("epoch", "updates")}
    for key in ("splits", "split_fingerprint", "metadata_file_sha256", "train_npz_sha256", "columns",
                "window_epochs", "multiplier", "downsample_rate", "max_epochs"):
        checks["identity."+key] = key in a["identity"] and a["identity"][key] == b["identity"].get(key)
    for key in ("windows", "window", "condition_columns", "diffusion_identity", "initialization_seed",
                "validation_seed_reset_each_epoch", "reference_mask", "reference_mask_policy", "gaussian_kl"):
        checks["protocol."+key] = key in a["protocol"] and a["protocol"][key] == b["protocol"].get(key)
    checks["validation_options"] = _options(a["protocol"]["options"]) == _options(b["protocol"]["options"])
    checks["training_arguments"] = {k:v for k,v in vars(a["args"]).items() if k != "validation_output"} == {
        k:v for k,v in vars(b["args"]).items() if k != "validation_output"}
    for key in ("continuous_mean", "continuous_std", "continuous_columns"):
        checks["condition_scaler."+key] = a["identity"]["scaler"][key] == b["identity"]["scaler"][key]
    if not all(checks.values()):
        raise ValueError("Original A/B comparison differs: "+", ".join(k for k,v in checks.items() if not v))
    return arms, checks


def export_arm(context, destination, device):
    """Reconstruct recorded inputs and generate exactly once; no training call."""
    args, manifest = context["args"], context["manifest"]
    np.random.seed(42)
    torch.manual_seed(42)
    preprocessor, data_checks = bind_training_data(args, manifest, context["manifest_path"], Preprocessor)
    model = fetchModel(len(preprocessor.columns), 1, args).to(device)
    assert_cuda_model(model)
    checkpoint_bytes = context["checkpoint"].read_bytes()
    loaded_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    if loaded_hash != context["checkpoint_identity"]["sha256"]:
        raise ValueError("Final checkpoint changed after identity verification")
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    copied = load_original_parameters(model, saved, torch)
    model.eval()
    schedule = {key:value.to(device) if torch.is_tensor(value) else value
                for key,value in build_diffusion_config(args).items()}
    config = deepcopy(context["config"])
    config["validation"] = dict(context["protocol"]["options"], save_artifacts=True)
    validator = EpochValidator(preprocessor, args, model, schedule, device, destination/"validation", config)
    protocol_checks = compare_protocols(context["protocol"], validator.protocol)
    epoch = context["epoch"]
    epoch_dir = validator.output/f"epoch_{epoch:04d}"
    with preserve_random_state(device):
        metrics, model_copy, sampling = validator._generate(model)
    artifacts = validator.save_generation_artifacts(epoch_dir, metrics, epoch=epoch)
    new_kl = float(metrics["aggregate"]["gaussian_kl_real_to_generated"])
    if not math.isfinite(new_kl):
        raise FloatingPointError("Re-exported KL is not finite")
    report = dict(status="completed", epoch=epoch, updates=context["updates"], checkpoint_kind="final",
        checkpoint=str(context["checkpoint"]), checkpoint_sha256=loaded_hash,
        checkpoint_identity=context["checkpoint_identity"], configuration=context["config_provenance"],
        runtime_source_checks=context["source_checks"], data_identity_checks=data_checks,
        protocol_checks=protocol_checks, original_protocol=str(context["protocol_path"]),
        protocol=str(validator.output/"protocol.json"), aggregate=metrics["aggregate"],
        diagnostics=metrics.get("diagnostics", {}), original_final_kl=context["old_kl"],
        kl_difference_from_original_final=new_kl-context["old_kl"],
        original_final_kl_close=math.isclose(new_kl,context["old_kl"],rel_tol=1e-6,abs_tol=1e-8),
        checkpoint_load=copied, model_copy=model_copy, sampling=sampling, artifacts=artifacts,
        device=str(device), training_executed=False, best_checkpoint_selected=False,
        limitation="Same seeds/configuration do not guarantee bitwise equality across CUDA software or hardware versions.")
    save_json(epoch_dir/"metrics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    cli = parser.parse_args(argv)
    if cli.threads < 1:
        parser.error("threads must be positive")
    reference = cli.reference_run.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    if output == reference or output.is_relative_to(reference):
        raise ValueError("Output must be outside the reference run; original artifacts are read only")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", reference_run=str(reference),
                  output_directory=str(output), training_executed=False, checkpoint_kind="final",
                  arms={}, scope="Re-export of existing final validation; no optimizer updates or best-model selection")
    save_json(output/"summary.json", status)
    try:
        contexts, checks = load_reference_pair(cli.reference_run)
        status["shared_reference_checks"] = checks
        device = require_cuda()
        if torch.device(device).type != "cuda":
            raise RuntimeError("CUDA is required; CPU fallback is disabled")
        torch.set_num_threads(cli.threads)
        status.update(status="exporting", threads=cli.threads, cuda_environment=cuda_environment(device))
        save_json(output/"summary.json", status)
        for arm in ARMS:
            print(json.dumps(dict(event="final_validation_export_started",arm=arm,epoch=contexts[arm]["epoch"])),flush=True)
            status["arms"][arm] = export_arm(contexts[arm],output/arm,device)
            save_json(output/"summary.json", status)
            print(json.dumps(dict(event="final_validation_export_completed",arm=arm,
                                 **status["arms"][arm]["aggregate"])),flush=True)
        status["status"] = "completed"
        save_json(output/"summary.json", status)
    except BaseException as error:
        status.update(status="failed",error_type=type(error).__name__,error=str(error))
        save_json(output/"summary.json", status)
        raise


if __name__ == "__main__":
    main()
