"""Train all 88 DREAMT subjects; select by current S092 KL in 12 CUDA runs.

S092 participates in both training and checkpoint selection. This protocol has
no held-out subjects. Each task starts fresh and owns an isolated workspace.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_missing_value_ablation import (
    _read_json, _epoch_rows, _check_training_args, compare_reference_preflights,
    expected_training_args, save_json, validate_training_budget,
)
from scripts.run_minmax_experiment import validate_dropout_report
from scripts.run_dropout_sweep import validate_s4_report
from scripts.run_preprocessing_experiment import sha256, final_score
from signal_normalization import normalization_spec, signal_affine_parameters
from s4_dropout_config import s4_dropout_options
from training_input_dropout import input_dropout_options
from diffusion_schedule import diffusion_identity

PROBABILITIES = (0.0, 0.03, 0.05)
RANGES = ((-1.0, 1.0), (0.0, 1.0))
GRID_PATH = ROOT / 'configs/all88_s092_sweep.json'
PROTOCOL = dict(mode='all_subjects_train_single_subject_eval', subject='S092', expected_subjects=88)
SNAPSHOT_SOURCES = ('training_wavestitch.py', 'training_utils.py', 'TSImputers/SSSDS4Imputer.py',
    'TSImputers/S4Model.py', 'data_utils.py', 'epoch_validation.py', 'validation_sampling.py',
    'validation_metrics.py', 'diffusion_schedule.py', 'signal_normalization.py',
    'training_input_dropout.py', 's4_dropout_config.py', 'scripts/evaluate_original_sampler.py')


def validate_configuration(row):
    fields = {'id', 'name', 'normalization_range', 'dropout_location', 'input_probability', 's4_probability'}
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError('Incomplete or unknown all88 configuration fields')
    if type(row['id']) is not int or not 0 <= row['id'] < 12:
        raise ValueError('Configuration id must be an integer from 0 to 11')
    bounds = row['normalization_range']
    if (not isinstance(bounds, list) or tuple(bounds) not in RANGES
            or any(type(v) not in (int, float) for v in bounds)):
        raise ValueError('Expected normalization range [-1,1] or [0,1]')
    p_input = input_dropout_options({'training_input_dropout': {'probability': row['input_probability']}})['probability']
    p_s4 = s4_dropout_options({'s4_dropout': row['s4_probability']})
    location = row['dropout_location']
    if location == 'input' and p_input in PROBABILITIES and p_s4 == 0:
        probability = p_input
    elif location == 's4' and p_s4 in PROBABILITIES and p_input == 0:
        probability = p_s4
    else:
        raise ValueError('Only the selected dropout location may use 0, 0.03 or 0.05')
    tag = 'm11' if bounds[0] == -1 else '01'
    if row['name'] != f"minmax_{tag}_{location}_p{str(probability).replace('.', 'p')}":
        raise ValueError('Configuration name must match its range, location and probability')
    return deepcopy(row)


def load_grid(path=GRID_PATH):
    data = _read_json(path)
    if data.get('schema') != 'dreamt_all88_s092_sweep_v1' or not isinstance(data.get('configurations'), list):
        raise ValueError('Invalid all88 sweep schema')
    rows = [validate_configuration(row) for row in data['configurations']]
    expected = {(bounds, location, p) for bounds in RANGES for location in ('input', 's4') for p in PROBABILITIES}
    actual = {(tuple(r['normalization_range']), r['dropout_location'], r['input_probability'] or r['s4_probability']) for r in rows}
    if (len(rows) != 12 or {r['id'] for r in rows} != set(range(12))
            or len({r['name'] for r in rows}) != 12 or actual != expected):
        raise ValueError('Grid must contain exactly 12 distinct labeled configurations')
    return sorted(rows, key=lambda row: row['id'])


def select_configuration(grid, configuration_id):
    if type(configuration_id) is not int or not 0 <= configuration_id < 12:
        raise ValueError('configuration-id must be an integer from 0 to 11')
    rows = [row for row in grid if row['id'] == configuration_id]
    if len(rows) != 1:
        raise ValueError('Expected exactly one selected configuration')
    return validate_configuration(rows[0])


def sweep_config(base, config_path, row):
    selected = validate_configuration(row)
    if not isinstance(base, dict):
        raise ValueError('Base configuration must be an object')
    config = deepcopy(base)
    config_path = Path(config_path).expanduser().resolve()
    if not config.get('data_dir'):
        raise ValueError('Base configuration must specify data_dir')
    for key in ('data_dir', 'metadata_path'):
        value = config.get(key)
        if value is not None:
            path = Path(value).expanduser()
            config[key] = str((path if path.is_absolute() else config_path.parent / path).resolve())
    config.update(evaluation_protocol=deepcopy(PROTOCOL), split_path=None, generate_splits=False,
        multiplier=1.0, downsample_rate=1, max_epochs=None, window_epochs=4,
        missing_value_policy='preserve', diffusion_schedule='linear_power', terminal_alpha_bar=1e-8,
        signal_normalization=dict(method='minmax', min_seconds=0.0, max_seconds=2.0),
        training_input_dropout=dict(probability=selected['input_probability'], seed=42),
        s4_dropout=selected['s4_probability'],
        early_stopping=dict(enabled=False, patience=10, selection_mode='current', smoothing_window=1, min_delta_kl=0.0),
        validation=dict(max_windows=None, batch_size=8, seed=20261020, kl_variance_floor_ms2=1.0,
                        reference_mask_policy='observed_only', save_artifacts=True))
    if selected['normalization_range'][0] == 0:
        config['signal_normalization']['output_range'] = [0.0, 1.0]
    return config


def validate_preflight(preflight, row):
    selected = validate_configuration(row)
    identity = preflight['data_identity']
    splits = identity['splits']
    names = splits['train']
    counts = preflight.get('training_subject_window_counts', {})
    windows = preflight['validation_windows']
    starts = [item['start_sample'] for item in windows]
    selection = preflight['validation_selection']
    bounds = selected['normalization_range']
    spec = dict(method='minmax', min_seconds=0.0, max_seconds=2.0)
    if bounds[0] == 0:
        spec['output_range'] = [0.0, 1.0]
    checks = dict(
        passed=preflight.get('status') == 'passed',
        protocol=identity.get('evaluation_protocol') == PROTOCOL,
        all_subjects=identity.get('split_source') == 'all_subjects' and len(names) == len(set(names)) == 88,
        no_holdout=splits.get('val') == splits.get('test') == [],
        evaluation_in_training='S092.npz' in names and identity.get('evaluation_filenames') == ['S092.npz'],
        all_subjects_have_training_windows=(set(counts) == set(names)
            and all(type(v) is int and v > 0 for v in counts.values())
            and sum(counts.values()) == preflight['training_windows']),
        all_training_hashes=set(identity['train_npz_sha256']) == set(names),
        all_subject_scaler=(identity['scaler']['metadata_fit_subjects'] == 88
            and set(identity['scaler_subjects']) == {Path(name).stem for name in names}),
        full_records=identity.get('max_epochs') is None and identity.get('multiplier') == identity.get('downsample_rate') == 1,
        dimensions=preflight.get('window_epochs') == 4 and preflight.get('window_size') == 3000,
        only_s092=bool(windows) and all(item['filename'] == 'S092.npz' for item in windows),
        nonoverlapping=starts == sorted(set(starts)) and all(b-a >= 3000 for a, b in zip(starts, starts[1:])),
        all_evaluation_windows=(preflight['validation_options'].get('max_windows') is None
            and len(windows) == selection.get('available_nonoverlapping_windows') == selection.get('selected_windows')
            and selection.get('selected_subjects') == selection.get('available_subjects') == 1),
        validation_options=preflight['validation_options'] == dict(max_windows=None, batch_size=8, seed=20261020,
            kl_variance_floor_ms2=1.0, reference_mask_policy='observed_only', save_artifacts=True),
        current_kl=preflight['early_stopping'] == dict(enabled=False, patience=10, selection_mode='current',
            smoothing_window=1, min_delta_kl=0.0),
        input_dropout=input_dropout_options(preflight) == dict(probability=selected['input_probability'], seed=42),
        s4_dropout=s4_dropout_options(preflight) == selected['s4_probability'],
        preserve_zeros=preflight.get('missing_value_policy') == 'preserve',
        normalization=normalization_spec(identity) == normalization_spec(identity['scaler']) == spec,
        affine=signal_affine_parameters(identity['scaler']) == ((0.0, 2.0) if bounds[0] == 0 else (1.0, 1.0)),
        diffusion=preflight['diffusion_settings'] == dict(diffusion_schedule='linear_power', terminal_alpha_bar=1e-8),
        full_diffusion_identity=preflight.get('diffusion_identity') == diffusion_identity(dict(
            diffusion_schedule='linear_power', terminal_alpha_bar=1e-8, beta_0=.0001, beta_T=.02, timesteps=200)),
    )
    if not all(checks.values()):
        raise ValueError('All88/S092 preflight mismatch: ' + ', '.join(key for key, okay in checks.items() if not okay))
    return checks


def validate_best_checkpoint(run_dir, manifest, rows):
    """Verify best means the first epoch attaining the minimum current KL."""
    scores = [float(row['gaussian_kl_real_to_generated']) for row in rows]
    if not scores or any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError('Invalid per-epoch KL values')
    if any('smoothed_gaussian_kl_real_to_generated' in row or 'smoothing_ready' in row for row in rows):
        raise ValueError('Current-KL runs must not contain moving-average metric columns')
    expected_epoch = scores.index(min(scores)) + 1
    best = manifest['validation']['best_checkpoints']['kl']
    if (best.get('selection_rule') != 'current_kl' or best.get('epoch') != expected_epoch
            or not math.isclose(best.get('value', math.nan), min(scores), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Best checkpoint is not the epoch with minimum current S092 KL')
    checkpoint = Path(run_dir) / 'training/best_kl_model.pth'
    if not checkpoint.is_file() or sha256(checkpoint) != best.get('checkpoint_sha256'):
        raise ValueError('Best checkpoint SHA256 mismatch')
    selected = _read_json(Path(run_dir) / 'training/best_kl_manifest.json')
    if (selected.get('checkpoint_epoch') != expected_epoch
            or selected.get('checkpoint_sha256') != best['checkpoint_sha256']
            or selected.get('checkpoint_selection', {}).get('selection_rule') != 'current_kl'
            or selected.get('checkpoint_selection', {}).get('value') != best['value']
            or selected.get('data_identity') != manifest.get('data_identity')
            or selected.get('training_args') != manifest.get('training_args')):
        raise ValueError('Best checkpoint manifest differs from current-KL selection or training configuration')
    return dict(best, checkpoint=str(checkpoint))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=GRID_PATH)
    parser.add_argument('--config', type=Path, default=ROOT / 'dataset_config.json')
    parser.add_argument('--configuration-id', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    cli = parser.parse_args(argv)
    if min(cli.epochs, cli.batch_size, cli.threads) < 1:
        parser.error('epochs, batch-size and threads must be positive')
    row = select_configuration(load_grid(cli.grid), cli.configuration_id)
    config = sweep_config(_read_json(cli.config), cli.config, row)
    output = cli.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha256(ROOT / name) for name in SNAPSHOT_SOURCES}
    status = dict(status='checking', configuration=row, output_directory=str(output), training_completed=False,
        epochs=cli.epochs, batch_size=cli.batch_size, threads=cli.threads,
        initialization='fresh model/optimizer, training seed 42; no resume',
        evaluation_protocol=deepcopy(PROTOCOL), evaluation_is_held_out=False,
        checkpoint_selection='minimum current epoch S092 Gaussian KL; no smoothing',
        duplicate_zero_baseline=row['input_probability'] == row['s4_probability'] == 0,
        grid_sha256=sha256(cli.grid), source_sha256=hashes,
        limitation='S092 is training data; this measures fitting, not held-out generalization. Native S4 dropout consumes the training RNG.')
    save_json(output / 'summary.json', status)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('DREAMT_')}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        config_path = output / 'config.json'
        save_json(config_path, config)
        common = [sys.executable, '-B', str(ROOT / 'scripts/run_dreamt.py'), '--config', str(config_path),
            '--epochs', str(cli.epochs), '--batch-size', str(cli.batch_size), '--threads', str(cli.threads)]
        subprocess.run(common + ['--check-only', '--output-dir', str(output / 'preflight')],
            cwd=ROOT, env=environment, check=True)
        preflight = _read_json(output / 'preflight/preflight.json')
        status['preflight_checks'] = validate_preflight(preflight, row)
        status['training_subjects'] = len(preflight['data_identity']['splits']['train'])
        status['training_windows'] = preflight['training_windows']
        status['expected_updates'] = math.ceil(preflight['training_windows']/cli.batch_size)*cli.epochs
        status['evaluation_windows'] = len(preflight['validation_windows'])
        status['command'] = common + ['--skip-generation', '--output-dir', str(output / 'run')]
        status['status'] = 'checked' if cli.check_only else 'training'
        save_json(output / 'summary.json', status)
        if cli.check_only:
            print(json.dumps(dict(event='all88_s092_checked', configuration=row, output_directory=str(output))), flush=True)
            return status
        print(json.dumps(dict(event='all88_s092_training_started', configuration=row,
                              log=str(output/'training.log'))), flush=True)
        with (output/'training.log').open('x', encoding='utf-8') as stream:
            subprocess.run(status['command'], cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, check=True)
        actual = _read_json(output/'run/preflight.json')
        compare_reference_preflights(preflight, actual)
        validate_preflight(actual, row)
        manifest = _read_json(output/'run/training/run_manifest.json')
        rows = _epoch_rows(output/'run/training/epoch_metrics.csv')
        budget = validate_training_budget(manifest, rows, preflight, cli.epochs, cli.batch_size)
        _check_training_args(manifest['training_args'], expected_training_args(preflight, cli.epochs, cli.batch_size))
        for key in ('data_identity', 'diffusion_identity'):
            if manifest.get(key) != preflight[key]:
                raise ValueError('Completed training/preflight mismatch: ' + key)
        recorded_hashes = {name.replace('\\', '/'): value for name, value in manifest.get('source_sha256', {}).items()}
        if any(recorded_hashes.get(name) != value for name, value in hashes.items()):
            raise ValueError('Frozen training source differs from the preflight source')
        status['completed_input_dropout'] = validate_dropout_report(manifest, preflight, budget)
        status['completed_s4_dropout'] = validate_s4_report(manifest, row['s4_probability'])
        best = validate_best_checkpoint(output/'run', manifest, rows)
        checkpoint = output/'run/workspace/saved_models/DREAMT/model.pth'
        if not checkpoint.is_file() or sha256(checkpoint) != manifest.get('checkpoint_sha256'):
            raise ValueError('Final checkpoint SHA256 mismatch')
        score = final_score(output/'run', row['name'], preflight, budget)
        status.update(status='completed', training_completed=True, completed_budget=budget,
            final_metrics=score, best_checkpoint=best, final_checkpoint=str(checkpoint),
            final_checkpoint_sha256=manifest['checkpoint_sha256'])
        save_json(output/'summary.json', status)
        print(json.dumps(dict(event='all88_s092_training_completed', configuration=row,
                              output_directory=str(output), best_checkpoint=best)), flush=True)
        return status
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        save_json(output/'summary.json', status)
        raise


if __name__ == '__main__':
    main()
