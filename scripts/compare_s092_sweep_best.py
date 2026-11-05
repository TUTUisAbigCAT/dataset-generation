"""Freeze selected sweep runs' current best and compare the same S092 conditions.

Training may still be running. A live validation summary and matching weight
bytes identify the selected epoch; a final best manifest is not required.
Each inference worker imports its run's verified source snapshot. No training,
scaler refitting, real-IBI prefix, stitching or output clipping is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from host_runtime import configure_host_threads
if __name__ == '__main__':
    configure_host_threads()
from scripts import compare_s092_best_minmax as shared

REQUIRED_SOURCES = set(shared.REQUIRED_SOURCES) | {
    'training_input_dropout.py', 's4_dropout_config.py', 'training_wavestitch.py',
    'dataset_config.json',
}
PROTOCOL = dict(mode='all_subjects_train_single_subject_eval', subject='S092', expected_subjects=88)
read_json, save_json, sha256 = shared.read_json, shared.save_json, shared.sha256


def _source_checks(workspace, recorded):
    if not isinstance(recorded, dict):
        raise ValueError('Missing frozen source hashes')
    normalized = {}
    for name, digest in recorded.items():
        name = name.replace('\\', '/')
        if name in normalized and normalized[name] != digest:
            raise ValueError('Conflicting frozen source hash aliases: ' + name)
        normalized[name] = digest
    if not REQUIRED_SOURCES.issubset(normalized):
        raise ValueError('Missing required frozen source identity: ' + ', '.join(sorted(REQUIRED_SOURCES-set(normalized))))
    for name, digest in normalized.items():
        source = (workspace / name).resolve()
        if not source.is_relative_to(workspace) or not source.is_file():
            raise ValueError('Missing or nonlocal frozen source: ' + name)
        if sha256(source) != digest:
            raise ValueError('Frozen source SHA256 differs: ' + name)
    return normalized


def _validate_run(manifest, configuration, workspace):
    # Configuration checks must not cache current training modules under their
    # real names: a worker subsequently imports its frozen versions. These
    # validators are stdlib-only and loaded under private, non-runtime aliases.
    def helper(name):
        spec = importlib.util.spec_from_file_location('_s092_binding_' + name, ROOT/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    normalization = helper('signal_normalization')
    signal_affine_parameters = normalization.signal_affine_parameters
    normalization_spec = normalization.normalization_spec
    input_dropout_options = helper('training_input_dropout').input_dropout_options
    s4_dropout_options = helper('s4_dropout_config').s4_dropout_options

    args, identity = manifest.get('training_args'), manifest.get('data_identity')
    if not isinstance(args, dict) or not isinstance(identity, dict):
        raise ValueError('Training has not recorded its arguments/data identity yet')
    splits = identity.get('splits', {})
    train = splits.get('train', [])
    if (identity.get('evaluation_protocol') != PROTOCOL or set(splits) != {'train', 'val', 'test'}
            or len(train) != 88 or len(set(train)) != 88 or 'S092.npz' not in train
            or splits['val'] or splits['test'] or identity.get('evaluation_filenames') != ['S092.npz']):
        raise ValueError('Require all 88 training subjects including S092 and S092-only evaluation')
    if identity.get('missing_value_policy', 'preserve') != 'preserve':
        raise ValueError('This sweep requires preserve missing-value policy')
    if args.get('dataset') != 'DREAMT' or args.get('window_size') != 3000 or identity.get('window_epochs') != 4:
        raise ValueError('This comparison requires the recorded four-epoch / 3000-point DREAMT model')
    scaler = identity['scaler']
    norm = normalization_spec(scaler)
    expected_range = configuration['normalization_range']
    if (norm is None or norm['min_seconds'] != 0.0 or norm['max_seconds'] != 2.0
            or norm.get('output_range', [-1., 1.]) != expected_range
            or normalization_spec(identity) != norm):
        raise ValueError('Frozen normalization differs from the selected configuration')
    expected_affine = (1., 1.) if expected_range == [-1., 1.] else (0., 2.)
    if signal_affine_parameters(scaler) != expected_affine:
        raise ValueError('Frozen normalization affine parameters disagree')
    options = manifest.get('training_input_dropout', {}).get('options')
    if (not isinstance(options, dict) or input_dropout_options({'training_input_dropout': options})
            != {'probability': configuration['input_probability'], 'seed': 42}):
        raise ValueError('Recorded training input dropout differs from the selected configuration')
    if s4_dropout_options(args) != configuration['s4_probability']:
        raise ValueError('Recorded S4 dropout differs from the selected configuration')
    if 's4_dropout' in manifest and s4_dropout_options(manifest) != configuration['s4_probability']:
        raise ValueError('Manifest S4 dropout differs from training arguments')
    config = read_json(workspace / 'dataset_config.json')
    if (normalization_spec(config) != norm or input_dropout_options(config) != input_dropout_options({'training_input_dropout': options})
            or s4_dropout_options(config) != configuration['s4_probability']
            or config.get('evaluation_protocol') != PROTOCOL):
        raise ValueError('Frozen dataset configuration differs from recorded training settings')
    return identity


def freeze_configuration(sweep_root, configuration, output_dir, attempts=3, retry_seconds=.2):
    """Capture a consistent live best, without touching the training directory."""
    if type(attempts) is not int or attempts < 1 or retry_seconds < 0:
        raise ValueError('Snapshot retry bounds must be nonnegative with positive attempts')
    # Reuse the sweep's canonical label validation, after host policy setup.
    from scripts.run_all88_s092_sweep import validate_configuration
    configuration = validate_configuration(configuration)
    run = Path(sweep_root).resolve() / f'config_{configuration["id"]}' / 'run'
    workspace = (run / 'workspace').resolve()
    training = run / 'training'
    output = Path(output_dir).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError('Inference output must be outside the training run')
    manifest_path, summary_path = training/'run_manifest.json', training/'validation/summary.json'
    weights_path = training/'best_kl_model.pth'
    for attempt in range(attempts):
        try:
            manifest_bytes = manifest_path.read_bytes()
            summary_bytes = summary_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
            validation = json.loads(summary_bytes.decode('utf-8-sig'))
            best = validation.get('best_checkpoints', {}).get('kl')
            if not isinstance(best, dict):
                raise ValueError('No best KL checkpoint has been recorded yet')
            epoch, value = best.get('epoch'), best.get('value')
            completed = validation.get('completed_epochs')
            if (type(epoch) is not int or epoch < 1 or type(completed) is not int or epoch > completed
                    or best.get('selection_rule') != 'current_kl'
                    or best.get('metric') != 'gaussian_kl_real_to_generated'
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('Inconsistent current_kl best epoch/metric metadata')
            if best.get('raw_metrics', {}).get('gaussian_kl_real_to_generated') != value:
                raise ValueError('Best raw KL differs from its current_kl selection value')
            weight_bytes = weights_path.read_bytes()
            digest = hashlib.sha256(weight_bytes).hexdigest()
            if digest != best.get('checkpoint_sha256'):
                raise ValueError('Live best checkpoint SHA256 differs from validation summary')
            if summary_bytes != summary_path.read_bytes() or manifest_bytes != manifest_path.read_bytes():
                raise ValueError('Live best metadata changed while taking snapshot')
            break
        except (OSError, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f'Cannot freeze config_{configuration["id"]} current best: {error}') from error
            time.sleep(retry_seconds)
    checks = _source_checks(workspace, manifest.get('source_sha256'))
    identity = _validate_run(manifest, configuration, workspace)
    snapshot_dir = output / 'checkpoint_snapshot'
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = snapshot_dir / 'best_kl_model.pth'
    checkpoint.write_bytes(weight_bytes)
    selected_path = snapshot_dir / 'best_snapshot_manifest.json'
    selected = deepcopy(manifest)
    selected.update(status='checkpoint_snapshot', checkpoint=str(checkpoint), checkpoint_sha256=digest,
        checkpoint_epoch=epoch, checkpoint_selection={k: v for k, v in best.items()
            if k not in ('checkpoint', 'checkpoint_sha256')}, validation=validation,
        parent_run_manifest=str(manifest_path),
        snapshot=dict(taken_at_utc=datetime.now(timezone.utc).isoformat(),
            source_run_status=manifest.get('status'), source_best_path=str(weights_path),
            source_run_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            source_validation_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
            validated_epochs_at_snapshot=completed, live_reads=attempt+1,
            training_completion_required=False, copied_weights=True))
    save_json(selected_path, selected)
    context = dict(arm='minmax' if configuration['normalization_range'][0] == -1 else 'minmax01',
        configuration=configuration, root=str(run.parent), workspace=str(workspace),
        manifest_path=str(selected_path), manifest=selected, identity=identity, epoch=epoch,
        checkpoint=str(checkpoint), checkpoint_sha256=digest,
        best_manifest_sha256=sha256(selected_path), source_checks=checks)
    save_json(output/'context.json', context)
    return context


def select_configurations(grid, configuration_ids=None):
    """Select explicit IDs before accessing run files; never skip failures silently."""
    ids = [row['id'] for row in grid] if configuration_ids is None else configuration_ids
    if (not isinstance(ids, list) or not ids
            or any(type(value) is not int or not 0 <= value < 12 for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Configuration IDs must be a nonempty list of unique integers from 0 to 11')
    rows = sorted((row for row in grid if row['id'] in ids), key=lambda row: row['id'])
    if len(rows) != len(ids) or {row['id'] for row in rows} != set(ids):
        raise ValueError('Every selected configuration must exist exactly once in the grid')
    return rows


def compare_contexts(contexts, *, expected_ids=None):
    """Allow the requested dropout/range differences; keep data and math paired."""
    rows = [context['configuration'] for context in contexts]
    select_configurations(rows)
    if expected_ids is not None:
        select_configurations(rows, expected_ids)
        if {row['id'] for row in rows} != set(expected_ids):
            raise ValueError('Captured configurations differ from the requested IDs')
    checks = {}
    base = contexts[0]
    def data_key(context):
        identity = deepcopy(context['identity'])
        identity.pop('signal_normalization', None)
        for name in ('signal_normalization', 'signal_center_seconds', 'signal_scale_seconds'):
            identity['scaler'].pop(name, None)
        return identity
    def args_key(context):
        return {k: v for k, v in context['manifest']['training_args'].items()
                if k not in ('s4_dropout', 'validation_output')}
    for context in contexts:
        label = f'config_{context["configuration"]["id"]}'
        _validate_run(context['manifest'], context['configuration'], Path(context['workspace']))
        checks[label+'.data'] = data_key(context) == data_key(base)
        checks[label+'.args'] = args_key(context) == args_key(base)
        checks[label+'.schedule'] = context['manifest']['diffusion_identity'] == base['manifest']['diffusion_identity']
        checks[label+'.core_sources'] = all(context['source_checks'][name] == base['source_checks'][name]
            for name in REQUIRED_SOURCES-{'dataset_config.json'})
    if not all(checks.values()):
        raise ValueError('Configurations differ beyond the intended sweep settings: ' + ', '.join(k for k,v in checks.items() if not v))
    return checks


def load_snapshot_context(path):
    context = read_json(path)
    if (sha256(context['manifest_path']) != context['best_manifest_sha256']
            or read_json(context['manifest_path']) != context['manifest']
            or sha256(context['checkpoint']) != context['checkpoint_sha256']):
        raise ValueError('Immutable inference checkpoint snapshot changed')
    manifest = context['manifest']
    expected_arm = 'minmax' if context['configuration']['normalization_range'][0] == -1 else 'minmax01'
    if (manifest['checkpoint_epoch'] != context['epoch']
            or manifest['checkpoint_selection']['epoch'] != context['epoch']
            or manifest['data_identity'] != context['identity']
            or manifest['checkpoint_sha256'] != context['checkpoint_sha256']
            or Path(manifest['checkpoint']).resolve() != Path(context['checkpoint']).resolve()
            or context['arm'] != expected_arm):
        raise ValueError('Snapshot identity, epoch or checkpoint differs from context')
    checks = _source_checks(Path(context['workspace']).resolve(), context['manifest']['source_sha256'])
    if checks != context['source_checks']:
        raise ValueError('Snapshot source identity differs from context')
    _validate_run(context['manifest'], context['configuration'], Path(context['workspace']))
    return context


def matching_inputs(output, configurations):
    import numpy as np
    baseline = None
    checks = {}
    for row in configurations:
        label = f'config_{row["id"]}'
        with np.load(output/label/'inputs.npz', allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        if baseline is None:
            baseline = arrays
        checks[label] = set(arrays) == set(baseline) and all(np.array_equal(v, baseline[k]) for k,v in arrays.items())
    if not all(checks.values()):
        raise ValueError('S092 references, stages or conditions differ between configurations')
    with (output/'reference.npz').open('xb') as stream:
        np.savez_compressed(stream, **baseline)
    return checks


def write_window_tables(output, reports):
    """Use the recorded source indices and weight every complete window equally."""
    import numpy as np
    with np.load(output/'reference.npz', allow_pickle=False) as reference:
        starts = reference['start_samples'].tolist()
        window_size = int(reference['window_size'])
        sample_rate = int(reference['sample_rate_hz'])
        # Older inputs were created by the nonoverlapping full-window protocol.
        stride_seconds = int(reference['window_stride_seconds']) if 'window_stride_seconds' in reference else window_size // sample_rate
        counts = reference['scoring_mask'].reshape(len(starts), window_size).sum(axis=1).tolist()
    moments = ('reference_mean_ms', 'generated_mean_ms', 'mean_error_ms', 'mean_absolute_error_ms',
               'reference_variance_ms2', 'generated_variance_ms2', 'variance_error_ms2',
               'variance_absolute_error_ms2', 'gaussian_kl_real_to_generated')
    detail, summary = [], []
    for report in reports:
        identity = dict(configuration_id=report['configuration']['id'], checkpoint_epoch=report['epoch'],
                        seed=report['seed'])
        windows = report['per_window_metrics']
        if ([row['start_sample'] for row in windows] != starts
                or [row['observed_counts'] for row in windows] != counts):
            raise ValueError('Window starts or scoring counts differ from the common S092 reference')
        for index, row in enumerate(windows):
            detail.append(dict(**identity, window_index=index, start_sample=starts[index],
                               start_second=starts[index]/sample_rate, observed_samples=counts[index],
                               **{key: row[key] for key in moments}))
        # The worker reduced one GPU scalar per window across all batches.
        summary.append(dict(**identity, windows=len(starts), window_seconds=window_size/sample_rate,
                            window_stride_seconds=stride_seconds,
                            **{'avg_'+key: report['metrics'][key] for key in moments}))
    for filename, records in [('per_window_metrics.csv', detail), ('all_window_summary.csv', summary)]:
        with (output/filename).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def launch_worker(cli, context, phase, destination):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', phase,
        '--context', str(destination/'context.json'), '--output-dir', str(destination),
        '--seed', str(cli.seed), '--threads', str(cli.threads)]
    if cli.all_windows:
        command += ['--all-windows', '--sampling-batch-size', str(cli.sampling_batch_size),
                    '--window-stride-seconds', str(cli.window_stride_seconds)]
    else:
        command += ['--start-second', str(cli.start_second), '--windows', str(cli.windows)]
    identity = context['identity']
    if phase == 'preflight':
        command += ['--npz', str(cli.npz or Path(identity['data_directory'])/'S092.npz'),
            '--metadata', str(cli.metadata or identity['metadata_path'])]
    with (destination/(phase+'.log')).open('x', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'config_{context["configuration"]["id"]} {phase} failed; see {destination/(phase+".log")}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root', type=Path)
    parser.add_argument('--grid', type=Path, default=ROOT/'configs/all88_s092_sweep.json')
    parser.add_argument('--config-ids', type=int, nargs='+',
                        help='Configuration IDs to compare (default: all 12); excluded runs are not read')
    parser.add_argument('--npz', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--all-windows', action='store_true',
                        help='Sample complete eligible S092 windows at the requested stride')
    parser.add_argument('--window-stride-seconds', type=int, choices=(30, 60, 90, 120),
                        help='S092 source-window step in --all-windows mode (default: 30 seconds)')
    parser.add_argument('--start-second', type=int)
    parser.add_argument('--windows', type=int)
    parser.add_argument('--sampling-batch-size', type=int,
                        help='Number of windows per GPU batch in --all-windows mode (default: 8)')
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', choices=('preflight', 'sample'), help=argparse.SUPPRESS)
    parser.add_argument('--context', type=Path, help=argparse.SUPPRESS)
    cli = parser.parse_args(argv)
    if cli.all_windows and (cli.windows is not None or cli.start_second is not None):
        parser.error('--all-windows selects starts across the full source record')
    if cli.window_stride_seconds is not None and not cli.all_windows:
        parser.error('--window-stride-seconds requires --all-windows')
    if cli.window_stride_seconds is None:
        cli.window_stride_seconds = 30
    if cli.sampling_batch_size is not None and (not cli.all_windows or cli.sampling_batch_size < 1):
        parser.error('--sampling-batch-size requires --all-windows and a positive size')
    cli.sampling_batch_size = cli.sampling_batch_size or 8
    if cli.start_second is None:
        cli.start_second = 0 if cli.all_windows else 18900
    if cli.windows is None:
        cli.windows = 3
    if cli.threads < 1 or cli.windows < 1 or cli.start_second < 0 or cli.start_second % 30 or not 0 <= cli.seed < 2**32:
        parser.error('Require positive threads/windows, unsigned seed, and a nonnegative 30-second-aligned start')
    policy = configure_host_threads(cli.threads)
    if cli.worker:
        if cli.context is None:
            parser.error('Worker requires --context')
        context = load_snapshot_context(cli.context)
        if cli.worker == 'preflight':
            shared.worker_preflight(context, cli.npz, cli.metadata, cli.start_second, cli.output_dir,
                windows=cli.windows, require_all_stages=True, all_windows=cli.all_windows,
                window_stride_seconds=cli.window_stride_seconds)
        else:
            report = shared.worker_sample(context, cli.output_dir, cli.seed, cli.threads,
                batch_size=cli.sampling_batch_size if cli.all_windows else None)
            report.update(configuration=context['configuration'], host_thread_runtime=policy,
                inference_dropout=dict(training_input_dropout_applied=False, s4_mode='eval'),
                reference_sha256=sha256(cli.output_dir/'inputs.npz'), evaluation_is_held_out=False)
            save_json(cli.output_dir/'summary.json', report)
        return
    if cli.sweep_root is None:
        parser.error('--sweep-root is required')
    output = cli.output_dir.expanduser().resolve()
    sweep = cli.sweep_root.expanduser().resolve()
    if output == sweep or output.is_relative_to(sweep):
        parser.error('Use a new output directory outside the training sweep')
    from scripts.run_all88_s092_sweep import load_grid
    try:
        configurations = select_configurations(load_grid(cli.grid), cli.config_ids)
    except ValueError as error:
        parser.error(str(error))
    selected_ids = [row['id'] for row in configurations]
    excluded_ids = sorted(set(range(12))-set(selected_ids))
    if not cli.check_only:
        from cuda_runtime import require_cuda
        require_cuda()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='checking', sweep_root=str(sweep), output_directory=str(output),
        configurations=configurations, seed=cli.seed, start_second=cli.start_second,
        selected_configuration_ids=selected_ids, excluded_configuration_ids=excluded_ids,
        windows=cli.windows, window_seconds=120, duration_seconds=cli.windows*120,
        training_executed=False, sampling_performed=False, independent_windows=True,
        real_ibi_used_as_input=False, overlap_samples=0, evaluation_is_held_out=False,
        host_thread_runtime=policy, snapshots={}, check_only=cli.check_only,
        script_sha256=sha256(__file__))
    if cli.all_windows:
        status.update(window_selection=('all_nonoverlapping_validation_windows' if cli.window_stride_seconds == 120
                                        else 'eligible_source_windows_at_requested_stride'),
                      start_second=None, windows=None, duration_seconds=None, sampling_batch_size=cli.sampling_batch_size,
                      window_stride_seconds=cli.window_stride_seconds, overlap_samples=(120-cli.window_stride_seconds)*25,
                      stitching_overlap_samples=0)
    try:
        save_json(output/'summary.json', status)
        print(json.dumps(dict(event='sweep_configuration_selection',
                              selected=selected_ids, excluded=excluded_ids)), flush=True)
        contexts = []
        for row in configurations:
            destination = output/f'config_{row["id"]}'
            context = freeze_configuration(sweep, row, destination)
            contexts.append(context)
            status['snapshots'][str(row['id'])] = dict(epoch=context['epoch'],
                checkpoint_sha256=context['checkpoint_sha256'], snapshot=context['manifest']['snapshot'])
        status['shared_training_checks'] = compare_contexts(contexts, expected_ids=selected_ids)
        save_json(output/'summary.json', status)
        for context in contexts:
            destination = output/f'config_{context["configuration"]["id"]}'
            print(json.dumps(dict(event='sweep_preflight', configuration=context['configuration']['id'], epoch=context['epoch'])), flush=True)
            launch_worker(cli, context, 'preflight', destination)
        status['shared_input_checks'] = matching_inputs(output, configurations)
        status['reference_sha256'] = sha256(output/'reference.npz')
        if cli.all_windows:
            first = read_json(output/f'config_{selected_ids[0]}'/'preflight.json')
            for key in ('windows', 'start_second', 'duration_seconds', 'source_samples', 'trailing_samples',
                        'uncovered_samples', 'window_stride_seconds', 'window_overlap_seconds',
                        'selected_sample_count', 'unique_covered_samples', 'sampled_duration_seconds',
                        'validation_anchor_windows'):
                status[key] = first[key]
            print(json.dumps(dict(event='s092_all_windows_checked', windows=status['windows'],
                                  window_stride_seconds=status['window_stride_seconds'],
                                  window_overlap_seconds=status['window_overlap_seconds'],
                                  trailing_samples=status['trailing_samples'])), flush=True)
        if cli.check_only:
            status['status'] = 'checked'
        else:
            reports = []
            for context in contexts:
                cid = context['configuration']['id']
                destination = output/f'config_{cid}'
                print(json.dumps(dict(event='sweep_sampling', configuration=cid, epoch=context['epoch'], seed=cli.seed)), flush=True)
                launch_worker(cli, context, 'sample', destination)
                reports.append(read_json(destination/'summary.json'))
                status['sampling_performed'] = True
                status['completed_configurations'] = [r['configuration']['id'] for r in reports]
                save_json(output/'summary.json', status)
            status['paired_cuda_rng_states_match'] = all(
                all(r[k] == reports[0][k] for k in ('cuda_rng_before', 'cuda_rng_after')) for r in reports)
            if not status['paired_cuda_rng_states_match']:
                raise RuntimeError('Paired CUDA noise streams differ across configurations')
            if cli.all_windows:
                write_window_tables(output, reports)
                status.update(per_window_metrics=str(output/'per_window_metrics.csv'),
                              all_window_summary=str(output/'all_window_summary.csv'),
                              window_weighting='equal per window; shared source positions enter each covering window; generated minus reference; population variance')
            else:
                status.update(status='completed', plots_directory=str(output/'plots'))
                save_json(output/'summary.json', status)
                subprocess.run([sys.executable, '-B', str(ROOT/'scripts/plot_s092_sweep_best.py'),
                                '--result-dir', str(output)], check=True, cwd=ROOT)
            status['status'] = 'completed'
        save_json(output/'summary.json', status)
        print(json.dumps(dict(event='s092_sweep_finished', status=status['status'],
                              configuration_ids=selected_ids, output_directory=str(output))), flush=True)
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        save_json(output/'summary.json', status)
        raise


if __name__ == '__main__':
    main()
