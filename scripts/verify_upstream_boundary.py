"""Verify frozen upstream files and the exact data/device/epoch-reporting adaptation allowlist."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


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


def expected_training_source(original):
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    start = original.index(b"    df = preprocessor.df_cleaned")
    end = original.index(b"    model = fetchModel", start)
    result = original[:start] + REPLACEMENT.encode().replace(b"\n", newline) + original[end:]
    for before, after in CUDA_DEVICE_EDITS + EPOCH_REPORTING_EDITS + EARLY_STOPPING_EDITS:
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
        assert (ROOT / name).read_bytes() == (UPSTREAM / name).read_bytes(), f"Changed imported core: {name}"
    original = (UPSTREAM / "training_wavestitch.py").read_bytes()
    actual = (ROOT / "training_wavestitch.py").read_bytes()
    assert actual == expected_training_source(original), "Training modifications exceeded the exact data/device/epoch-reporting allowlist"

    # Reverse exactly the reviewed device/reporting edits to verify the unchanged training
    # algorithm. The removed mask conversion needs its original insertion point.
    normalized = actual.decode().replace("\r\n", "\n")
    for before, after in reversed(CUDA_DEVICE_EDITS + EPOCH_REPORTING_EDITS + EARLY_STOPPING_EDITS):
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
    watchsleep_path = ROOT / "third_party/WatchSleepNet/data_setup.py"
    assert sha(watchsleep_path) == "47936ea03412afcacb98121c3239cf19a240ee0b700b14381a19c93cbd87504e", "Changed supplied WatchSleepNet loader snapshot"
    report = dict(status="passed", official_commit=manifest["commit"],
        watchsleep_snapshot_sha256=sha(watchsleep_path),
        watchsleep_epoch_adapter_sha256=sha(ROOT / "watchsleep_loader.py"),
        sleep_stage_classes=["W", "Light", "REM", "Deep"],
        upstream_snapshot_files_checked=len(hashes), upstream_sha256=hashes,
        imported_core_sha256=imported_core, imported_core_byte_identical=True,
        training_entry_sha256=sha(ROOT / "training_wavestitch.py"),
        data_adapter_sha256=sha(ROOT / "data_utils.py"),
        cuda_runtime_sha256=sha(ROOT / "cuda_runtime.py"),
        exact_training_edit="Data/window preparation, enumerated CUDA-only device edits and epoch validation/reporting/early stopping",
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
        training_window_stride_change="Original ignores CLI stride and unfolds by 1 point; adapted data windows honor explicit -stride 1500 (60 seconds)",
        runtime_configuration="CUDA required; 64 channels, 4 residual layers, state 64, T200 linear 0.0001 to 0.02, Adam 1e-4; CLI window_size/s4_lmax 3000; batch/epochs are runtime budget arguments")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    diff = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
        actual.decode().splitlines(keepends=True), fromfile="official/training_wavestitch.py", tofile="adapted/training_wavestitch.py"))
    args.output.with_suffix(".patch").write_text(diff, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "upstream_snapshot_files_checked",
        "imported_core_byte_identical", "training_loop_ast_identical",
        "training_loop_ast_identical_after_allowed_normalization",
        "optimizer_loss_diffusion_formulas_and_update_order_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
