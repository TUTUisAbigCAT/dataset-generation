"""Read-only evidence checks before controlled DREAMT generation diagnostics.

This module does not load weights, refit a scaler, or infer a historical run from
the current configuration. Missing evidence stays incomplete. Paths may move;
the corresponding content hashes, transforms and model settings must match.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import PurePosixPath
import re

from diffusion_schedule import diffusion_identity


MODEL_FIELDS = (
    "backbone", "window_size", "num_res_layers", "res_channels", "skip_channels",
    "diff_step_embed_in", "diff_step_embed_mid", "diff_step_embed_out", "s4_lmax",
    "s4_dstate", "s4_dropout", "s4_bidirectional", "s4_layernorm", "propCycEnc",
)
REQUIRED_CORE_SOURCES = (
    "training_utils.py", "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py", "utils/util.py",
)
DATA_SOURCES = ("data_utils.py", "watchsleep_loader.py", "window_config.py", "diffusion_schedule.py")
DATA_FIELDS = (
    "sample_rate_hz", "epoch_samples", "window_epochs", "multiplier", "downsample_rate",
    "max_epochs", "columns", "stage_names", "raw_stage_mapping", "zero_value_policy",
    "scaler", "scaler_subjects", "splits", "metadata_file_sha256",
)
UNUSED_INFERENCE_FIELDS = ("lr", "batch_size", "epochs", "validation_output", "stride", "hdim", "layers")
SAMPLER_SOURCES = ("scripts/evaluate_original_sampler.py", "night_sampling.py", "validation_sampling.py")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _mapping(value):
    if isinstance(value, Mapping):
        return dict(value)
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def _json(value):
    """Keep reports serializable even when an input contains malformed values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return repr(value)


def _digest(value):
    return value.lower() if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _source_key(path):
    value = str(path).replace("\\", "/").strip().lower()
    # Canonicalize a copied checkout or an absolute Windows/Linux source path.
    for marker in ("tsimputers/", "utils/"):
        position = value.rfind("/" + marker)
        if position >= 0:
            return value[position + 1:]
        if value.startswith(marker):
            return value
    for known in REQUIRED_CORE_SOURCES + DATA_SOURCES + SAMPLER_SOURCES:
        key = known.lower()
        if value == key or value.endswith("/" + key):
            return key
    return value.removeprefix("./")


def _source_map(values):
    result, conflicts = {}, []
    for path, value in _mapping(values).items():
        key = _source_key(path)
        normalized = _digest(value)
        if key in result and result[key] != normalized:
            conflicts.append(key)
        result[key] = normalized
    return result, sorted(set(conflicts))


def _data_identity(manifest):
    if isinstance(manifest.get("data_identity"), Mapping):
        return dict(manifest["data_identity"])
    data = _mapping(manifest.get("data"))
    if isinstance(data.get("data_identity"), Mapping):
        return dict(data["data_identity"])
    return data


def _train_hashes(identity):
    for name in ("train_npz_sha256", "training_npz_sha256"):
        if isinstance(identity.get(name), Mapping):
            return dict(identity[name])
    opened = identity.get("opened_npz_sha256")
    train = _mapping(identity.get("splits")).get("train")
    if isinstance(opened, Mapping) and isinstance(train, list) and all(name in opened for name in train):
        return {name: opened[name] for name in train}
    return None


def audit_training_identity(training_manifest, inference_args, current_data_identity,
                            current_source_hashes, checkpoint_sha256,
                            validation_summary=None, inference_summary=None):
    """Return all identity checks; only a fully matched report permits sampling.

    ``checkpoint_sha256`` identifies the bytes that will actually be loaded.
    Best-KL evidence is required; the main manifest's final checkpoint hash is
    deliberately not a fallback. No path in any input is opened by this function.
    """
    manifest = _mapping(training_manifest)
    current_args = _mapping(inference_args)
    current_data = _mapping(current_data_identity)
    checks, differences = [], []

    def add(name, status, detail, **values):
        checks.append(dict(name=name, status=status, detail=detail, **_json(values)))

    def difference(name, detail, **values):
        differences.append(dict(name=name, detail=detail, **_json(values)))

    def compare(name, expected, actual, detail):
        add(name, "matched" if expected == actual else "mismatch", detail,
            expected=expected, actual=actual)

    def fields(prefix, expected, actual, names):
        for name in names:
            if name not in expected or name not in actual:
                add(prefix + "." + name, "incomplete", "Required field is missing",
                    missing_from=[side for side, data in (("historical", expected), ("current", actual)) if name not in data])
            else:
                left, right = expected[name], actual[name]
                if name == "backbone" and isinstance(left, str) and isinstance(right, str):
                    left, right = left.lower(), right.lower()
                compare(prefix + "." + name, left, right, "Compare effective configuration values")

    def audit_args(prefix, historical):
        historical = _mapping(historical)
        fields(prefix + ".model", historical, current_args, MODEL_FIELDS)
        for side, arguments in (("historical", historical), ("current", current_args)):
            invalid = []
            for name in MODEL_FIELDS:
                if name not in arguments:
                    continue
                value = arguments[name]
                if name == "backbone":
                    valid = isinstance(value, str) and value.lower() in ("s4", "s4classic", "s4weaver")
                elif name in ("s4_bidirectional", "s4_layernorm", "propCycEnc"):
                    valid = isinstance(value, bool)
                elif name == "s4_dropout":
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value < 1
                else:
                    valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
                if not valid:
                    invalid.append(name)
            if invalid:
                add(prefix + "." + side + ".model_values", "incomplete", "Model arguments do not establish a supported finite configuration", fields=invalid)
        for name in UNUSED_INFERENCE_FIELDS:
            if name in historical and name in current_args and historical[name] != current_args[name]:
                difference(prefix + "." + name,
                    "Training-only, unused S4 template, or sampling-protocol setting; not a model-state mismatch",
                    historical=historical[name], current=current_args[name])
        return historical

    def schedule(prefix, arguments, recorded, historical=True):
        arguments = dict(arguments)
        if "diffusion_schedule" not in arguments:
            arguments["diffusion_schedule"] = "linear"
            difference(prefix + ".legacy_default", "An absent historical schedule means legacy linear, never current linear_power")
        try:
            effective = diffusion_identity(arguments)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
            add(prefix, "incomplete", "Cannot reconstruct the effective schedule: " + str(error))
            return None
        if recorded is None:
            if historical and effective["diffusion_schedule"] != "linear":
                add(prefix + ".recorded_identity", "incomplete", "A nonlinear historical schedule needs its recorded tensor identity")
            elif historical:
                difference(prefix + ".legacy_reconstruction", "Legacy linear tensor identity reconstructed from explicit T and beta arguments")
        elif not isinstance(recorded, Mapping):
            add(prefix + ".recorded_identity", "incomplete", "Recorded diffusion identity must be an object")
        else:
            compare(prefix + ".recorded_identity", effective, dict(recorded),
                    "Recorded identity, full beta array and tensor hashes must agree with historical arguments")
        return effective

    def audit_data(prefix, historical):
        historical = _mapping(historical)
        # Old four-epoch identities precede the configurable-window field.
        if historical and "window_epochs" not in historical:
            historical["window_epochs"] = 4
            difference(prefix + ".legacy_window", "Historical identity without window_epochs means four 30-second epochs")
        fields(prefix, historical, current_data, DATA_FIELDS)
        for side, identity in (("historical", historical), ("current", current_data)):
            if isinstance(identity.get("train_npz_sha256"), Mapping) and isinstance(identity.get("training_npz_sha256"), Mapping):
                compare(prefix + "." + side + ".training_hash_aliases", identity["train_npz_sha256"],
                        identity["training_npz_sha256"], "Duplicate training-hash aliases must identify the same file contents")
            scaler = _mapping(identity.get("scaler"))
            required_scaler = ("signal_mean_seconds", "signal_std_seconds", "continuous_columns", "continuous_mean", "continuous_std")
            missing = [name for name in required_scaler if name not in scaler]
            if missing:
                add(prefix + "." + side + ".scaler_fields", "incomplete", "Signal and condition affine transforms must be recorded", fields=missing)
            else:
                def finite_number(value):
                    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                mean, std = scaler["signal_mean_seconds"], scaler["signal_std_seconds"]
                means, stds = scaler["continuous_mean"], scaler["continuous_std"]
                valid = (finite_number(mean) and finite_number(std) and std > 0
                    and scaler["continuous_columns"] == ["AHI", "BMI", "AGE"]
                    and isinstance(means, list) and len(means) == 3 and all(finite_number(x) for x in means)
                    and isinstance(stds, list) and len(stds) == 3 and all(finite_number(x) and x > 0 for x in stds))
                if not valid:
                    add(prefix + "." + side + ".scaler_values", "incomplete", "Malformed/nonfinite affine scaling evidence")
            if _digest(identity.get("metadata_file_sha256")) is None:
                add(prefix + "." + side + ".metadata_digest", "incomplete", "Metadata content SHA-256 is required")
            splits = _mapping(identity.get("splits"))
            valid_split = (set(splits) == {"train", "val", "test"}
                and all(isinstance(value, list) and all(isinstance(name, str) for name in value) for value in splits.values())
                and bool(splits.get("train")))
            if not valid_split:
                add(prefix + "." + side + ".split_members", "incomplete", "Nonempty training partition and all three subject lists are required")
            else:
                names = [name for partition in splits.values() for name in partition]
                if len(set(names)) != len(names):
                    add(prefix + "." + side + ".split_members", "mismatch", "Subject partitions overlap or contain duplicate records")
                hashes = _train_hashes(identity)
                if hashes is not None and set(hashes) != set(splits["train"]):
                    add(prefix + "." + side + ".hash_coverage", "incomplete", "Per-file hashes must cover exactly the training partition")
        left_hashes, right_hashes = _train_hashes(historical), _train_hashes(current_data)
        if left_hashes is None or right_hashes is None:
            add(prefix + ".train_npz_sha256", "incomplete", "Per-training-file content hashes are required")
        elif any(_digest(value) is None for value in (*left_hashes.values(), *right_hashes.values())):
            add(prefix + ".train_npz_sha256", "incomplete", "Training-file hashes must be valid SHA-256 digests")
        else:
            compare(prefix + ".train_npz_sha256", {k: v.lower() for k, v in left_hashes.items()},
                    {k: v.lower() for k, v in right_hashes.items()}, "Compare training NPZ contents, not directory spelling")
        for key in ("data_directory", "metadata_path", "split_path", "split_source", "split_seed", "split_file_sha256", "split_fingerprint"):
            if key in historical and key in current_data and historical[key] != current_data[key]:
                difference(prefix + "." + key,
                    "Provenance differs; actual partitions, training-file contents, metadata and scalers are checked separately",
                    historical=historical[key], current=current_data[key])
        # The identity's physical dimensions must agree with the model arguments.
        if all(key in historical for key in ("window_epochs", "epoch_samples")):
            try:
                window = historical["window_epochs"] * historical["epoch_samples"]
                compare(prefix + ".window_samples", window, current_args.get("window_size"),
                        "Model window must equal identity window_epochs * epoch_samples")
            except TypeError:
                add(prefix + ".window_samples", "incomplete", "Invalid identity window dimensions")

    current_sources, conflicts = _source_map(current_source_hashes)
    if conflicts:
        add("current_sources.aliases", "mismatch", "Conflicting hashes identify the same normalized source path", paths=conflicts)

    def audit_sources(prefix, values):
        historical, aliases = _source_map(values)
        if aliases:
            add(prefix + ".aliases", "mismatch", "Conflicting hashes identify the same normalized source path", paths=aliases)
        required = {_source_key(name) for name in REQUIRED_CORE_SOURCES}
        supplemental = {key for key in historical if key.startswith(("tsimputers/", "utils/")) or key in DATA_SOURCES}
        for key in sorted(required | supplemental):
            if historical.get(key) is None or current_sources.get(key) is None:
                add(prefix + "." + key, "incomplete", "Historical and current source SHA-256 evidence is required")
            else:
                compare(prefix + "." + key, historical[key], current_sources[key], "Compare model, embedding, or recorded data-transform source contents")

    historical_args = audit_args("training", manifest.get("training_args"))
    historical_schedule = schedule("training.schedule", historical_args, manifest.get("diffusion_identity"))
    current_schedule = schedule("current.schedule", current_args, None, historical=False)
    if historical_schedule is not None and current_schedule is not None:
        # terminal_alpha_bar is unused for linear; compare the effective arrays.
        for name in ("diffusion_schedule", "timesteps", "beta_0", "beta_T", "tensor_sha256"):
            compare("training.schedule.current." + name, historical_schedule[name], current_schedule[name],
                    "Training and inference effective noise schedules must match")
        if historical_schedule["diffusion_schedule"] == "linear_power":
            compare("training.schedule.current.terminal_alpha_bar", historical_schedule["terminal_alpha_bar"],
                    current_schedule["terminal_alpha_bar"], "Compare nonlinear terminal target")
    audit_data("training.data", _data_identity(manifest))
    audit_sources("training.source", manifest.get("source_sha256"))

    # An interrupted run can have a complete identity manifest and a newer
    # validation summary. The selected best record, never the final model, binds
    # the requested weight bytes. Contradictory best evidence is not ignored.
    candidates = []
    for location, container in (("manifest.validation", manifest.get("validation")),
                                ("validation_summary", validation_summary)):
        best = _mapping(_mapping(container).get("best_checkpoints")).get("kl")
        if isinstance(best, Mapping) and "checkpoint_sha256" in best:
            candidates.append((location + ".best_checkpoints.kl", best.get("checkpoint_sha256")))
    selected = _mapping(manifest.get("checkpoint_selection"))
    basename = PurePosixPath(str(manifest.get("checkpoint", "")).replace("\\", "/")).name.lower()
    if selected.get("metric") == "gaussian_kl_real_to_generated" and basename == "best_kl_model.pth":
        candidates.append(("selected_best_manifest", manifest.get("checkpoint_sha256")))
    loaded_digest = _digest(checkpoint_sha256)
    if loaded_digest is None:
        add("checkpoint.loaded_sha256", "incomplete", "The actual checkpoint-byte SHA-256 is required")
    if not candidates:
        add("checkpoint.best_kl", "incomplete", "No recorded best-KL hash; a filename or final-checkpoint hash does not prove best-checkpoint identity")
    for location, digest in candidates:
        expected = _digest(digest)
        if expected is None:
            add("checkpoint." + location, "incomplete", "Recorded best-KL hash is missing or malformed")
        elif loaded_digest is not None:
            compare("checkpoint." + location, expected, loaded_digest, "Actual loaded bytes must match the recorded best-KL checkpoint")

    if inference_summary is not None:
        previous = _mapping(inference_summary)
        fields("previous.execution", previous, {"sampling_performed": True, "model_state_validated": True},
               ("sampling_performed", "model_state_validated"))
        old_digest = _mapping(previous.get("execution")).get("loaded_checkpoint_sha256")
        identity = _mapping(previous.get("checkpoint_identity"))
        if old_digest is None and identity.get("hash_scope") == "loaded_checkpoint_bytes":
            old_digest = identity.get("sha256")
        if _digest(old_digest) is None:
            add("previous.checkpoint", "incomplete", "Previous summary must identify actually loaded checkpoint bytes, not only a preflight path/hash")
        elif loaded_digest is not None:
            compare("previous.checkpoint", _digest(old_digest), loaded_digest, "Historical comparisons require the same actual checkpoint")
        old_args = audit_args("previous", previous.get("model_args"))
        old_schedule = schedule("previous.schedule", old_args, previous.get("diffusion_identity"))
        if old_schedule is not None and current_schedule is not None:
            compare("previous.schedule.current_tensors", old_schedule["tensor_sha256"], current_schedule["tensor_sha256"],
                    "Previous generation must use the same effective noise arrays")
        old_data = _mapping(previous.get("configured_data_identity"))
        audit_data("previous.data", old_data)
        if "scaler" not in previous:
            add("previous.scaler", "incomplete", "Previous generation scaler is missing")
        else:
            compare("previous.scaler", previous["scaler"], current_data.get("scaler"), "Compare scaler actually reported by previous generation")
        old_sources = _mapping(previous.get("source_sha256"))
        core = _mapping(_mapping(previous.get("source_provenance")).get("imported_core_sha256"))
        # Preserve conflicting aliases for audit_sources rather than overwrite.
        for key, value in core.items():
            old_sources["previous_core/" + str(key).replace("\\", "/")] = value
        audit_sources("previous.source", old_sources)
        normalized_old, _ = _source_map(old_sources)
        for name in SAMPLER_SOURCES:
            key = _source_key(name)
            if key in normalized_old and key in current_sources and normalized_old[key] != current_sources[key]:
                difference("previous.sampler." + name,
                    "Sampler correction is an intended experimental change, not training identity drift; the runtime sampler allowlist/provenance must verify the authorized change separately",
                    historical_sha256=normalized_old[key], current_sha256=current_sources[key])
        difference("previous.sampling_protocol", "Validation seeds, independent-window selection and whole-night stride/grouping are experimental controls, not model architecture settings",
                   previous_sampling_seed=_mapping(previous.get("execution")).get("sampling_seed"),
                   previous_stride_samples=previous.get("stride_samples"), previous_windows_per_group=previous.get("windows_per_group"))

    statuses = {item["status"] for item in checks}
    status = "mismatch" if "mismatch" in statuses else ("incomplete" if "incomplete" in statuses else "matched")
    return _json(dict(schema="dreamt_diagnostic_identity_v1", status=status, can_sample=status == "matched",
        blocking_checks=checks, expected_protocol_differences=differences,
        historical_generation_comparison_requested=inference_summary is not None,
        current_effective_diffusion_identity=current_schedule,
        historical_effective_diffusion_identity=historical_schedule,
        policy="Missing evidence is incomplete; mismatch takes precedence. No model execution or historical identity is inferred from matching filenames."))
