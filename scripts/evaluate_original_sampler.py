"""WaveStitch pipeline adapter with the DDPM posterior noise scale corrected.

The immutable upstream excerpt is verified before one explicit correction:
scale standard normal noise by sqrt(posterior variance), not the variance.
Noise draws, guidance and model buffers otherwise stay unchanged. The schedule
is restored from the selected training manifest through the local adapter.
Tensor factories run in a local CUDA device context; no CPU fallback is
permitted. All requested windows belong to one subject and one batch. The
adapter's clean condition columns are supplied; IBI is zero and wholly unknown.
This is a window-level check, not evidence of whole-night generation quality.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from numbers import Integral
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/WaveStitch'
SOURCE = UPSTREAM / 'synthesis_wavestitch_pipeline_strided_preconditioning.py'
sys.path.insert(0, str(ROOT))

from window_config import EPOCH_SAMPLES, FS
from diffusion_schedule import build_diffusion_config, diffusion_identity
from scripts.verify_upstream_boundary import CORE_FILES, expected_core_source


def digest(value):
    return hashlib.sha256(value).hexdigest()


def extract_original_pipeline():
    """Keep source bytes, including comments, indentation and CRLF, unchanged."""
    source_bytes = SOURCE.read_bytes()
    source = source_bytes.decode('utf-8')
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'create_pipelined_noise']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one upstream create_pipelined_noise')
    main_blocks = [node for node in tree.body if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Compare)
                   and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__']
    trials = [node for node in main_blocks[0].body if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == 'trial']
    blocks = [node for node in trials[0].body if isinstance(node, ast.With)]
    if len(blocks) != 1:
        raise RuntimeError('Expected exactly one original no_grad generation block')
    block = blocks[0]
    if not isinstance(block.body[-1], ast.For):
        raise RuntimeError('Original sampler batch-loop boundary changed')
    step_loops = [node for node in ast.walk(block) if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name) and node.target.id == 'step']
    if len(step_loops) != 1:
        raise RuntimeError('Expected the single complete upstream DDPM loop')

    def excerpt(node):
        return ''.join(lines[node.lineno - 1:node.end_lineno])

    noise_source = excerpt(functions[0])
    generation_source = excerpt(block)
    step_source = excerpt(step_loops[0])
    # The original block is indented by 8 spaces. Two inert lexical wrappers
    # let Python compile those exact bytes, without dedenting or rewriting them.
    if block.col_offset != 8:
        raise RuntimeError('Unexpected original sampler indentation')
    compiled_source = 'if True:\n    if True:\n' + generation_source
    if compiled_source.split('    if True:\n', 1)[1] != generation_source:
        raise RuntimeError('Sampler excerpt was changed during wrapping')
    manifest = json.loads((UPSTREAM / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    entries = {entry['path']: entry['sha256'] for entry in manifest['unmodified_upstream_files']}
    expected = entries[SOURCE.name]
    if digest(source_bytes) != expected:
        raise RuntimeError('Official sampler differs from the fixed upstream manifest')
    provenance = dict(upstream_commit=manifest['commit'], source_path=str(SOURCE),
        source_sha256=digest(source_bytes), manifest_sha256=expected,
        noise_function_sha256=digest(noise_source.encode('utf-8')),
        generation_block_sha256=digest(generation_source.encode('utf-8')),
        ddpm_for_step_sha256=digest(step_source.encode('utf-8')),
        noise_function_lines=[functions[0].lineno, functions[0].end_lineno],
        generation_block_lines=[block.lineno, block.end_lineno],
        ddpm_for_step_lines=[step_loops[0].lineno, step_loops[0].end_lineno],
        excerpts_byte_identical_to_upstream=True,
        generation_wrapper='Two if True blocks preserve the original 8-space indentation.',
        sampler_expressions_edited=False, random_draws_added_or_removed=False,
        original_variance_times_noise_preserved=True,
        original_guidance_minus_0_1_gradient_preserved=True)
    core_hashes = {}
    for name in CORE_FILES:
        original = (UPSTREAM / name).read_bytes()
        if digest(original) != entries[name]:
            raise RuntimeError('Official core snapshot differs from its manifest: ' + name)
        actual = (ROOT / name).read_bytes()
        if actual != expected_core_source(name, original):
            raise RuntimeError('Imported source exceeds the exact core correction allowlist: ' + name)
        core_hashes[name] = digest(actual)
    provenance.update(imported_core_byte_identical=False,
                      s4_fallback_conjugate_corrected=True,
                      s4_device_dtype_dispatch_corrected=True,
                      imported_core_sha256=core_hashes)
    return noise_source, generation_source, compiled_source, provenance


def extract_corrected_pipeline():
    """Apply the single authorized posterior-standard-deviation correction.

    Keep the vendored source and manifest intact for reproducibility. Every
    production sampling entry point uses this function; the raw extractor is
    retained only for source audits and regression comparisons.
    """
    noise_source, original_generation, _, original_provenance = extract_original_pipeline()
    before = 'vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal'
    after = 'vari = torch.sqrt(beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t))) * torch.normal'
    if original_generation.count(before) != 1:
        raise RuntimeError('Expected exactly one reviewed posterior variance noise coefficient')
    generation_source = original_generation.replace(before, after, 1)
    compiled_source = 'if True:\n    if True:\n' + generation_source
    tree = ast.parse(compiled_source)
    step_loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Name) and node.target.id == 'step']
    if len(step_loops) != 1:
        raise RuntimeError('Corrected sampler must retain the single upstream DDPM loop')
    lines = compiled_source.splitlines(keepends=True)
    step_source = ''.join(lines[step_loops[0].lineno - 1:step_loops[0].end_lineno])
    provenance = dict(original_provenance)
    provenance.update(
        upstream_snapshot_byte_identical=True,
        upstream_generation_block_sha256=original_provenance['generation_block_sha256'],
        upstream_ddpm_for_step_sha256=original_provenance['ddpm_for_step_sha256'],
        generation_block_sha256=digest(generation_source.encode('utf-8')),
        ddpm_for_step_sha256=digest(step_source.encode('utf-8')),
        executed_sampler_sha256=digest(compiled_source.encode('utf-8')),
        excerpts_byte_identical_to_upstream=False,
        sampler_expressions_edited=True,
        original_variance_times_noise_preserved=False,
        posterior_noise_scale='sqrt_posterior_variance',
        runtime_corrections=[dict(id='posterior_variance_to_std', before=before, after=after)],
    )
    return noise_source, generation_source, compiled_source, provenance


def load_original_parameters(model, saved, torch):
    """Match original named-parameter copy; verify untouched fixed buffers first."""
    if set(saved) != set(model.state_dict()):
        raise RuntimeError('Checkpoint keys do not match the original model configuration')
    comparisons = []
    for name, buffer in model.named_buffers():
        expected = saved[name]
        equal = buffer.shape == expected.shape and buffer.dtype == expected.dtype \
            and buffer.device == expected.device and torch.equal(buffer.detach(), expected.detach())
        comparisons.append(dict(name=name, shape=list(buffer.shape),
                                 dtype=str(buffer.dtype), bitwise_equal=equal))
    failures = [row['name'] for row in comparisons if not row['bitwise_equal']]
    if failures:
        raise RuntimeError('Seed-42 reconstructed buffers differ; no buffer repair permitted: '
                           + ', '.join(failures))
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved[name])
            param.requires_grad = True
    return dict(method='Original named_parameters copy; no load_state_dict or buffer copy.',
                initialization_seed=42, all_buffers_bitwise_equal=True,
                checked_buffers=comparisons)


def select_sampling_window(preprocessor, subject=None, start_sample=None, windows=1, stride=None):
    """Choose a recorded legal span; never stitch around invalid epochs or gaps."""
    window_size = preprocessor.window_size
    stride = min(1500, window_size) if stride is None else stride
    if (isinstance(windows, bool) or not isinstance(windows, Integral) or windows < 1
            or isinstance(stride, bool) or not isinstance(stride, Integral)
            or stride <= 0 or stride % EPOCH_SAMPLES or stride > window_size):
        raise ValueError('Expected positive integer windows and an epoch-aligned stride no larger than window_size')
    if subject is None:
        selected = preprocessor.select_validation_windows(windows_per_subject=1)
        filename = selected[0][0]
    else:
        filename = subject if subject.endswith('.npz') else subject + '.npz'
    if filename not in preprocessor.splits['train'] + preprocessor.splits['val']:
        raise ValueError('Selected subject must belong to the recorded train or validation split')
    length = window_size + stride * (windows - 1)
    if start_sample is not None:
        preprocessor.window_array(filename, start_sample, length)
        return filename, start_sample
    record = preprocessor.record(filename)
    for start in range(0, len(record['ibi']) - length + 1, EPOCH_SAMPLES):
        try:
            preprocessor.window_array(filename, start, length)
        except ValueError:
            continue
        return filename, start
    raise ValueError('Selected subject has no valid continuous span of the requested length')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        help='Explicit weights override; default: checkpoint recorded in the manifest')
    parser.add_argument('--training-args-json', type=Path,
                        help='Training manifest (or explicit specification) with complete training_args and data identity')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subject', help='A train/validation subject ID; default: first validation subject with a legal window')
    parser.add_argument('--start-sample', type=int, default=None,
                        help='Explicit epoch-aligned start; default: first legal continuous window in the selected subject')
    parser.add_argument('--windows', type=int, default=1,
                        help='Contiguous overlapping windows in one original pipeline batch.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--extract-only', action='store_true',
                        help='Verify the upstream snapshot and save the corrected sampler without inference.')
    cli = parser.parse_args()
    if cli.windows <= 0 or cli.threads <= 0:
        parser.error('--windows and --threads must be positive')
    if not cli.extract_only and cli.training_args_json is None:
        parser.error('--training-args-json is required for evaluation')
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', artificial_data=None, real_DREAMT_quality_claim=False,
        checkpoint=None, input_subject=cli.subject,
        signal_all_unknown=True, signal_input_zero=True, groups=1, windows=cli.windows,
        cross_group_anchoring=False, no_backend_override=True, no_model_patch=True)

    def save_report():
        (cli.output_dir / 'evaluation_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')

    try:
        noise_source, generation_source, compiled_source, provenance = extract_corrected_pipeline()
        report['source_provenance'] = provenance
        (cli.output_dir / 'original_create_pipelined_noise.py').write_bytes(noise_source.encode('utf-8'))
        (cli.output_dir / 'corrected_generation_block.py.txt').write_bytes(generation_source.encode('utf-8'))
        save_report()
        if cli.extract_only:
            compile(noise_source, str(SOURCE), 'exec')
            compile(compiled_source, str(SOURCE), 'exec')
            report['status'] = 'extraction_verified_no_inference'
            save_report()
            print(json.dumps(provenance, indent=2))
            return

        import numpy as np
        import torch
        from cuda_runtime import require_cuda, assert_cuda_model, cuda_environment
        device = require_cuda()
        from torch.utils.data import DataLoader
        from data_utils import Preprocessor
        from training_utils import MyDataset, fetchModel
        from scripts.evaluate_denoising import load_training_run, bind_training_data, resolve_checkpoint

        torch.set_num_threads(cli.threads)
        report['runtime'] = {**cuda_environment(device),
            'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
            'threads': cli.threads, 'compute_policy': 'CUDA required; no CPU fallback',
            'host_boundaries': ['file I/O and WatchSleepNet preprocessing',
                'original seeded constructor and recorded diffusion schedule initialization',
                'final result serialization']}
        args, manifest, config_provenance = load_training_run(cli.training_args_json)
        cli.checkpoint, checkpoint_identity = resolve_checkpoint(cli.training_args_json, manifest, cli.checkpoint)
        report.update(checkpoint=str(cli.checkpoint), checkpoint_identity=checkpoint_identity,
                      config_provenance=config_provenance)
        if args.dataset not in ('DREAMT', 'DREAMT_SIM'):
            raise ValueError('Expected DREAMT or DREAMT_SIM')
        if args.stride <= 0 or args.stride % EPOCH_SAMPLES or args.stride > args.window_size:
            raise ValueError('Stride must align with 30-second epochs and not exceed the window')
        if args.backbone.lower() != 's4':
            raise ValueError('Only the original S4 backbone is supported')
        report['training_args'] = vars(args)
        report['diffusion_identity'] = diffusion_identity(args)
        np.random.seed(42)
        torch.manual_seed(42)
        preprocessor, identity_checks = bind_training_data(args, manifest, cli.training_args_json, Preprocessor)
        report['training_data_identity_checks'] = identity_checks
        report['artificial_data'] = preprocessor.provenance().get('data_synthetic')
        report['source_sha256'] = {name: digest((ROOT / name).read_bytes()) for name in
             ('data_utils.py', 'watchsleep_loader.py', 'window_config.py', 'diffusion_schedule.py', 'scripts/evaluate_denoising.py',
             'scripts/evaluate_original_sampler.py')}
        columns = preprocessor.columns
        input_features = len(columns)
        expected_columns = ('IBI', 'stage_W', 'stage_Light', 'stage_REM', 'stage_Deep',
                            'AHI', 'BMI', 'AGE', 'gender_F', 'gender_M')
        hierarchical_column_indices = columns.get_indexer(preprocessor.hierarchical_features_cyclic)
        non_hier_cols = np.setdiff1d(np.arange(input_features), hierarchical_column_indices)
        if (tuple(columns) != expected_columns
                or hierarchical_column_indices.tolist() != list(range(1, input_features))
                or non_hier_cols.tolist() != [0]):
            raise ValueError('Expected the four-stage W/Light/REM/Deep schema and one IBI output')
        report.update(total_input_channels=input_features,
            clean_condition_channels=len(hierarchical_column_indices), columns=list(columns),
            stage_columns=list(columns[1:5]), condition_columns=list(columns[hierarchical_column_indices]))
        filename, start_sample = select_sampling_window(preprocessor, cli.subject, cli.start_sample,
                                                        cli.windows, args.stride)
        total_length = args.window_size + args.stride * (cli.windows - 1)
        reference = preprocessor.window_array(filename, start_sample, total_length)
        all_unknown_input = reference.copy()
        all_unknown_input[:, 0] = 0.0
        reference_tensor = torch.from_numpy(reference).to(device=device, dtype=torch.float32)
        windows = torch.from_numpy(all_unknown_input).to(device=device, dtype=torch.float32).unfold(
            0, args.window_size, args.stride).transpose(1, 2)
        masks = torch.ones(total_length, dtype=torch.bool, device=device).unfold(0, args.window_size, args.stride)
        if windows.shape != (cli.windows, args.window_size, input_features):
            raise RuntimeError('Unexpected aligned model input shape')
        test_dataset = MyDataset(windows.float())
        mask_dataset = MyDataset(masks)
        # Original seeded construction is preserved; every inference tensor moves to CUDA.
        model = fetchModel(input_features, 1, args).to(device)
        assert_cuda_model(model)
        diffusion_config = {key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in build_diffusion_config(args).items()}
        test_dataloader = DataLoader(test_dataset, batch_size=cli.windows)
        mask_dataloader = DataLoader(mask_dataset, batch_size=cli.windows)
        saved_params = torch.load(cli.checkpoint, map_location=device)
        report['checkpoint_sha256'] = digest(cli.checkpoint.read_bytes())
        report['checkpoint_load'] = load_original_parameters(model, saved_params, torch)
        report['data'] = preprocessor.provenance()
        report['data'].update(selected_subject=Path(filename).stem,
            start_sample=start_sample, generated_samples=total_length)
        model.eval()
        # Preserve the original setup names consumed by the corrected excerpt.
        scope = dict(torch=torch, sqrt=torch.sqrt, args=args, device=device, model=model,
            diffusion_config=diffusion_config, test_dataset=test_dataset,
            test_dataloader=test_dataloader, mask_dataloader=mask_dataloader,
            hierarchical_column_indices=torch.as_tensor(hierarchical_column_indices, device=device, dtype=torch.long),
            non_hier_cols=torch.as_tensor(non_hier_cols, device=device, dtype=torch.long), num_ops=0, trial=0)
        exec(compile(noise_source, str(SOURCE), 'exec'), scope)
        save_report()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        # Preserve draw count/order; only the posterior noise scale is corrected.
        # A local device context places its normal/full/empty and model embedding
        # factories directly on CUDA; it changes the RNG backend, not the schedule.
        with torch.device(device):
            exec(compile(compiled_source, str(SOURCE), 'exec'), scope)
        torch.cuda.synchronize(device)
        report['sampling_seconds'] = time.perf_counter() - start
        generated = scope['synth_tensor'].detach()
        if generated.device != device:
            raise RuntimeError('Original sampler returned a tensor on the wrong device')
        generated_seconds = (generated[:, 0] * preprocessor.scaler['signal_std_seconds']
                             + preprocessor.scaler['signal_mean_seconds'])
        reference_seconds = (reference_tensor[:, 0] * preprocessor.scaler['signal_std_seconds']
                             + preprocessor.scaler['signal_mean_seconds'])
        finite = bool(torch.isfinite(generated).all())
        report['result'] = dict(all_finite=finite,
            condition_columns_bitwise_preserved=bool(torch.equal(generated[:, 1:], reference_tensor[:, 1:])),
            samples=len(generated), duration_seconds=len(generated) / FS,
            denoiser_calls=int(scope['num_ops']),
            generated_mean_ms=float(generated_seconds.mean() * 1000) if finite else None,
            reference_mean_ms=float(reference_seconds.mean() * 1000),
            generated_std_ms=float(generated_seconds.std(unbiased=False) * 1000) if finite else None,
            reference_std_ms=float(reference_seconds.std(unbiased=False) * 1000),
            generated_min_ms=float(generated_seconds.min() * 1000) if finite else None,
            generated_max_ms=float(generated_seconds.max() * 1000) if finite else None)
        report['cuda_execution'] = dict(device=str(device),
            sampler_source_byte_identical=False, upstream_snapshot_byte_identical=True,
            posterior_variance_sqrt_corrected=True, local_default_device_context=True,
            random_draw_backend='CUDA', random_draw_count_unchanged=True,
            schedule_matches_training=True,
            schedule_is_legacy_linear=report['diffusion_identity']['diffusion_schedule'] == 'linear',
            cpu_bitwise_random_stream_equivalence_claim=False)
        # The only device-to-host array transfers are final NPZ serialization.
        np.savez_compressed(cli.output_dir / 'generated_samples.npz',
            generated_standardized=generated[:, 0].cpu().numpy(),
            generated_seconds=generated_seconds.cpu().numpy(),
            reference_seconds=reference_seconds.cpu().numpy(), clean_conditions=reference[:, 1:],
            generated_conditions=generated[:, 1:].cpu().numpy(), sample_rate_hz=np.array(FS),
            subject=np.array(Path(filename).stem), start_sample=np.array(start_sample))
        report['status'] = 'completed' if finite else 'nonfinite_output'
        save_report()
        print(json.dumps(report['result'], indent=2))
        if not finite:
            raise RuntimeError('Corrected sampler returned nonfinite values; saved without clipping')
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        save_report()
        raise


if __name__ == '__main__':
    main()
