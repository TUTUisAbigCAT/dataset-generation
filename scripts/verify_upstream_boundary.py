"""Audit frozen upstream, explicit schedule configuration, and the sampler correction."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/WaveStitch"
REPLACEMENT = """    # DATA ADAPTER: subject-local, epoch-aligned lazy windows; no algorithm changes.
    training_df = preprocessor.schema_frame
    hierarchical_column_indices = training_df.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    training_dataset = preprocessor.make_training_dataset(args.window_size, args.stride)
    in_dim = len(training_df.columns)
    out_dim = len(training_df.columns) - len(hierarchical_column_indices)
"""
CORE_FILES = ("training_utils.py", "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py",
              "TSImputers/TimeGAN.py", "utils/util.py")
# These three complete literal replacements are the only imported-core edits.
# The first repairs PyKeOps input materialization/device routing; the second
# supplies an always-available conjugate fallback and a validated dispatcher;
# the third routes the existing S4 resolvent call through that dispatcher.
S4_BACKEND_EDITS = [
    dict(id="pykeops_input_and_device", before="""        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2 * cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)""", after="""        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v.resolve_conj().contiguous())
        z = _c2r(z.resolve_conj().contiguous())
        w = _c2r(w.resolve_conj().contiguous())

        backend = 'GPU' if v.device.type == 'cuda' else 'CPU'
        r = 2 * cauchy_mult(v, z, w, backend=backend)
        return _r2c(r)"""),
    dict(id="conjugate_fallback_and_dispatch", before='''except ImportError:
    has_pykeops = False
    if not has_cauchy_extension:
        print("Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for efficiency.")

        def cauchy_slow(v, z, w):
            """
            v, w: (..., N)
            z: (..., L)
            returns: (..., L)
            """
            cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (... N L)
            return torch.sum(cauchy_matrix, dim=-2)''', after='''except ImportError:
    has_pykeops = False


def cauchy_slow(v, z, w):
    """Full conjugate-state resolvent in PyTorch, on the input device.

    v, w: (..., N); z: (..., L); returns: (..., L).
    S4 stores only half the states, so both conjugate terms are required.
    This fallback is always available, including when accelerators are installed.
    """
    v = torch.cat([v, v.conj()], dim=-1)
    w = torch.cat([w, w.conj()], dim=-1)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    return torch.sum(cauchy_matrix, dim=-2)


def cauchy_resolvent(v, z, w):
    """Select a compatible backend without changing precision or device."""
    if not all(torch.is_tensor(t) for t in (v, z, w)):
        raise TypeError("Cauchy inputs v, z and w must be tensors")
    if v.dtype not in (torch.complex64, torch.complex128) or not (v.dtype == z.dtype == w.dtype):
        raise TypeError("Cauchy inputs must have the same complex64 or complex128 dtype")
    if not (v.device == z.device == w.device):
        raise ValueError("Cauchy inputs must be on the same device")
    if v.device.type not in ("cpu", "cuda"):
        raise ValueError("Cauchy inputs must be on CPU or CUDA")
    if any(t.ndim < 1 or t.shape[-1] == 0 for t in (v, z, w)):
        raise ValueError("Cauchy state and frequency dimensions must be nonempty")
    if v.shape[-1] != w.shape[-1]:
        raise ValueError("Cauchy v and w must have the same state dimension")
    try:
        torch.broadcast_shapes(v.shape[:-1], z.shape[:-1], w.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("Cauchy input batch dimensions must broadcast") from exc

    if v.device.type == "cpu":
        return cauchy_slow(v, z, w)

    # The optional CUDA kernel supports float32 complex states, an unbatched z,
    # and power-of-two state counts. Its backward does not provide a z gradient.
    n = v.shape[-1]
    extension_compatible = (
        v.dtype == torch.complex64 and z.ndim == 1 and not z.requires_grad
        and 2 <= n <= 1024 and n & (n - 1) == 0 and z.numel() <= 2**31
    )
    if has_cauchy_extension and extension_compatible:
        with torch.cuda.device(v.device):
            return cauchy_mult(v.resolve_conj().contiguous(), z.resolve_conj().contiguous(),
                               w.resolve_conj().contiguous(), symmetric=True)
    if has_pykeops:
        with torch.cuda.device(v.device):
            return cauchy_conj(v.resolve_conj().contiguous(), z.resolve_conj().contiguous(),
                               w.resolve_conj().contiguous())
    # CUDA tensors stay on CUDA here; no host copy or dtype conversion.
    return cauchy_slow(v, z, w)'''),
    dict(id="s4_resolvent_call", before='''        if has_cauchy_extension and z.dtype == torch.cfloat:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            r = cauchy_slow(v, z, w)''', after='''        r = cauchy_resolvent(v, z, w)'''),
]
# Each edit must match exactly once. No AST-wide or unrestricted replacement is
# allowed. These device edits retain optimizer, loss and diffusion expressions.
CUDA_DEVICE_EDITS = [
    [
        "from torch.utils.data import DataLoader",
        "from torch.utils.data import DataLoader\nfrom cuda_runtime import require_cuda, assert_cuda_model"
    ],
    [
        "    device = device('cuda' if torch.cuda.is_available() else 'cpu')",
        "    device = require_cuda()"
    ],
    [
        "    model = fetchModel(in_dim, out_dim, args).to(device)",
        "    model = fetchModel(in_dim, out_dim, args).to(device)\n    assert_cuda_model(model)"
    ],
    [
        "    diffusion_config = fetchDiffusionConfig(args)",
        "    diffusion_config = fetchDiffusionConfig(args)\n    diffusion_config = {key: value.to(device) if torch.is_tensor(value) else value\n                        for key, value in diffusion_config.items()}"
    ],
    [
        "    non_hier_cols = np.array(remaining_indices)",
        "    non_hier_cols = np.array(remaining_indices)\n    non_hier_cols = torch.as_tensor(non_hier_cols, dtype=torch.long, device=device)"
    ],
    [
        "            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)",
        "            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],), device=device)"
    ],
    [
        "            sigmas = normal(0, 1, size=batch.shape).to(device)",
        "            sigmas = normal(0, 1, size=batch.shape, device=device)"
    ],
    [
        "            conditional_mask = np.ones(batch.shape)",
        "            conditional_mask = torch.ones(batch.shape, dtype=torch.float32, device=device)"
    ],
    [
        "            conditional_mask = from_numpy(conditional_mask).float().to(device)\n",
        ""
    ],
    [
        "            sigmas_predicted = model(batch_noised, timesteps)",
        "            with torch.device(device):\n                sigmas_predicted = model(batch_noised, timesteps)"
    ]
]


# The new schedule is external to the preserved upstream builder. Only this import and two
# parser fields change; the batch noising/epsilon-MSE/optimizer AST is retained.
DIFFUSION_SCHEDULE_EDITS = [
    ("from training_utils import MyDataset, fetchModel, fetchDiffusionConfig",
     "from training_utils import MyDataset, fetchModel\n"
     "from diffusion_schedule import build_diffusion_config as fetchDiffusionConfig"),
    ("    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')",
     "    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')\n"
     "    parser.add_argument('-diffusion_schedule', choices=('linear', 'linear_power'), default='linear_power')\n"
     "    parser.add_argument('-terminal_alpha_bar', type=float, default=1e-8)"),
]


# Exact user-authorized epoch validation and scalar loss reporting additions.
EPOCH_REPORTING_EDITS = [('from cuda_runtime import require_cuda, assert_cuda_model',
  'from cuda_runtime import require_cuda, assert_cuda_model\n'
  'from epoch_validation import create_epoch_validator'),
 ('    args = parser.parse_args()',
  '    parser.add_argument("-validation_output", type=str, default=None)\n'
  '    args = parser.parse_args()'),
 ('    """TRAINING"""',
  '    validator = create_epoch_validator(preprocessor, args, model, diffusion_config, device)\n'
  '    """TRAINING"""'),
 ('        total_loss = 0.0', '        total_loss = 0.0\n        total_samples = 0'),
 ('            total_loss += loss',
  '            total_loss += float(loss.detach()) * batch.shape[0]\n'
  '            total_samples += batch.shape[0]'),
 ("        print(f'epoch: {epoch}, loss: {total_loss}')",
  '        mean_loss = total_loss / total_samples\n'
  '        print(f"epoch: {epoch + 1}, mean_loss: {mean_loss}", flush=True)\n'
  '        validator.run(epoch + 1, model, mean_loss)')]

# Only the explicitly requested epoch-end stop is allowed; updates stay unchanged.
EARLY_STOPPING_EDITS = [(
    "        validator.run(epoch + 1, model, mean_loss)",
    "        validator.run(epoch + 1, model, mean_loss)\n"
    "        if validator.should_stop:\n            break",
)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_core_source(name, original):
    """Apply three exact reviewed backend edits; all other imported bytes stay fixed."""
    if name not in CORE_FILES:
        raise ValueError(f"Unknown imported core file: {name}")
    if name != "TSImputers/S4Model.py":
        return original
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    result = original
    for edit in S4_BACKEND_EDITS:
        before = edit["before"].encode().replace(b"\n", newline)
        after = edit["after"].encode().replace(b"\n", newline)
        if result.count(before) != 1:
            raise ValueError(f"S4 backend edit must match exactly one upstream block: {edit['id']}")
        result = result.replace(before, after, 1)
    return result


def expected_training_source(original):
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    start = original.index(b"    df = preprocessor.df_cleaned")
    end = original.index(b"    model = fetchModel", start)
    result = original[:start] + REPLACEMENT.encode().replace(b"\n", newline) + original[end:]
    for before, after in CUDA_DEVICE_EDITS + DIFFUSION_SCHEDULE_EDITS + EPOCH_REPORTING_EDITS + EARLY_STOPPING_EDITS:
        before_bytes = before.encode().replace(b"\n", newline)
        after_bytes = after.encode().replace(b"\n", newline)
        assert result.count(before_bytes) == 1, f"Training edit allowlist is ambiguous: {before}"
        result = result.replace(before_bytes, after_bytes, 1)
    return result


def normalized_training_loop(tree):
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "epoch"]
    assert len(loops) == 1
    return ast.dump(loops[0], include_attributes=False)


def audit_runtime_sampler():
    """Allow only sqrt(posterior variance); do not execute the sampler."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluate_original_sampler import extract_original_pipeline, extract_corrected_pipeline

    original_noise, original_generation, original_compiled, original_provenance = extract_original_pipeline()
    noise, generation, compiled, provenance = extract_corrected_pipeline()
    before = "beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal("
    after = "torch.sqrt(beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t))) * torch.normal("
    assert original_generation.count(before) == 1, "Posterior-noise correction must have one upstream target"
    assert original_compiled.count(before) == 1, "Compiled sampler correction target is ambiguous"
    assert noise == original_noise, "Sampler noise function or random draws changed"
    assert generation == original_generation.replace(before, after, 1), "Sampler changed beyond the authorized posterior-variance square root"
    assert compiled == original_compiled.replace(before, after, 1), "Executable sampler differs from the authorized correction"
    compile(compiled, "<audited_corrected_sampler>", "exec")
    assert original_provenance["excerpts_byte_identical_to_upstream"] is True
    expected_flags = dict(excerpts_byte_identical_to_upstream=False,
                          imported_core_byte_identical=False,
                          s4_fallback_conjugate_corrected=True,
                          s4_device_dtype_dispatch_corrected=True,
                          sampler_expressions_edited=True,
                          original_variance_times_noise_preserved=False,
                          upstream_snapshot_byte_identical=True,
                          posterior_noise_scale="sqrt_posterior_variance",
                          random_draws_added_or_removed=False)
    for key, expected in expected_flags.items():
        assert provenance[key] == expected, f"Incorrect corrected-sampler provenance: {key}"
    return dict(upstream_snapshot_byte_identical=True,
                runtime_sampler_source_byte_identical=False,
                runtime_sampler_single_sqrt_correction_verified=True,
                posterior_variance_sqrt_corrected=True,
                posterior_noise_scale="sqrt_posterior_variance",
                random_draws_added_or_removed=False,
                authorized_sampler_edit=dict(before=before, after=after),
                original_sampler_generation_sha256=hashlib.sha256(original_generation.encode("utf-8")).hexdigest(),
                runtime_sampler_generation_sha256=hashlib.sha256(generation.encode("utf-8")).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/upstream_boundary.json")
    args = parser.parse_args()
    manifest = json.loads((UPSTREAM / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "2b6628026c0689815ce79becf11536a571d86b1a"
    hashes = {}
    for record in manifest["unmodified_upstream_files"]:
        source = UPSTREAM / record["path"]
        assert sha(source) == record["sha256"], f"Changed upstream snapshot: {record['path']}"
        hashes[record["path"]] = sha(source)
    for name in CORE_FILES:
        assert (ROOT / name).read_bytes() == expected_core_source(name, (UPSTREAM / name).read_bytes()), f"Imported core exceeds exact correction allowlist: {name}"
    original = (UPSTREAM / "training_wavestitch.py").read_bytes()
    actual = (ROOT / "training_wavestitch.py").read_bytes()
    assert actual == expected_training_source(original), "Training modifications exceeded the exact data/device/epoch-reporting allowlist"

    # Reverse exactly the reviewed device/reporting edits to verify the unchanged training
    # algorithm. The removed mask conversion needs its original insertion point.
    normalized = actual.decode().replace("\r\n", "\n")
    for before, after in reversed(CUDA_DEVICE_EDITS + DIFFUSION_SCHEDULE_EDITS + EPOCH_REPORTING_EDITS + EARLY_STOPPING_EDITS):
        if not after:
            anchor = "            conditional_mask[:, :, non_hier_cols] = 0\n"
            assert normalized.count(anchor) == 1
            normalized = normalized.replace(anchor, anchor + before, 1)
        else:
            assert normalized.count(after) == 1
            normalized = normalized.replace(after, before, 1)
    original_text = original.decode().replace("\r\n", "\n")
    old_loop = normalized_training_loop(ast.parse(original_text))
    new_loop = normalized_training_loop(ast.parse(actual.decode()))
    normalized_loop = normalized_training_loop(ast.parse(normalized))
    assert old_loop == normalized_loop, "Training formulas/structure changed beyond allowed device allocation and epoch reporting"
    marker = "    model = fetchModel"
    assert normalized[normalized.index(marker):] == original_text[original_text.index(marker):]
    imported_core = {name: sha(ROOT / name) for name in CORE_FILES}
    sampler_audit = audit_runtime_sampler()
    watchsleep_path = ROOT / "third_party/WatchSleepNet/data_setup.py"
    assert sha(watchsleep_path) == "47936ea03412afcacb98121c3239cf19a240ee0b700b14381a19c93cbd87504e", "Changed supplied WatchSleepNet loader snapshot"
    report = dict(status="passed", official_commit=manifest["commit"],
        watchsleep_snapshot_sha256=sha(watchsleep_path),
        watchsleep_epoch_adapter_sha256=sha(ROOT / "watchsleep_loader.py"),
        sleep_stage_classes=["W", "Light", "REM", "Deep"],
        upstream_snapshot_files_checked=len(hashes), upstream_sha256=hashes,
        imported_core_sha256=imported_core, imported_core_byte_identical=False,
        s4_fallback_conjugate_corrected=True,
        s4_device_dtype_dispatch_corrected=True,
        s4_backend_edits=[dict(path="TSImputers/S4Model.py", **edit) for edit in S4_BACKEND_EDITS],
        training_entry_sha256=sha(ROOT / "training_wavestitch.py"),
        data_adapter_sha256=sha(ROOT / "data_utils.py"),
        window_configuration_sha256=sha(ROOT / "window_config.py"),
        diffusion_schedule_adapter_sha256=sha(ROOT / "diffusion_schedule.py"),
        cuda_runtime_sha256=sha(ROOT / "cuda_runtime.py"),
        exact_training_edit="Data/window preparation, explicit schedule adapter import/configuration, enumerated CUDA-only device edits and epoch validation/reporting/early stopping",
        diffusion_schedule_edits=[dict(before=before, after=after) for before, after in DIFFUSION_SCHEDULE_EDITS],
        early_stopping_edits=[dict(before=before, after=after) for before, after in EARLY_STOPPING_EDITS],
        epoch_reporting_edits=[dict(before=before, after=after) for before, after in EPOCH_REPORTING_EDITS],
        cuda_device_edits=[dict(before=before, after=after) for before, after in CUDA_DEVICE_EDITS],
        training_algorithm_and_save_byte_identical=False,
        training_loop_ast_identical=(old_loop == new_loop),
        training_loop_ast_identical_after_allowed_normalization=True,
        training_suffix_byte_identical_after_allowed_normalization=True,
        optimizer_loss_diffusion_formulas_and_update_order_unchanged=True,
        gpu_rng_note="Random draws now use the CUDA generator; distributions are unchanged but CPU seeded trajectories are not expected to match",
        official_sampler_snapshot_unchanged=True, sampler_execution_performed=False,
        **sampler_audit,
        training_window_stride_change="Original ignores CLI stride and unfolds by 1 point; adapted data windows honor explicit -stride 750 (30 seconds)",
        runtime_configuration="CUDA required; 64 channels, 4 residual layers, state 64, T200, Adam 1e-4; explicit external schedule adapter selects legacy linear or linear_power (new-run default terminal alpha_bar 1e-8); CLI window_size/s4_lmax = window_epochs * 750 (default 3000); batch/epochs are runtime budget arguments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    diff = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
        actual.decode().splitlines(keepends=True), fromfile="official/training_wavestitch.py", tofile="adapted/training_wavestitch.py"))
    args.output.with_suffix(".patch").write_text(diff, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "upstream_snapshot_files_checked",
        "imported_core_byte_identical", "training_loop_ast_identical",
        "training_loop_ast_identical_after_allowed_normalization",
        "optimizer_loss_diffusion_formulas_and_update_order_unchanged",
        "runtime_sampler_single_sqrt_correction_verified",
        "upstream_snapshot_byte_identical")}, indent=2))


if __name__ == "__main__":
    main()
