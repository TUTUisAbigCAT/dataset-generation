"""Live best-checkpoint snapshots using artificial bytes, never model execution.

The fixtures copy recorded code and configuration while all checkpoint payloads
are deliberately invalid tensor files. Only binding/provenance code reads them.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from diffusion_schedule import diffusion_identity
from scripts import compare_s092_sweep_best as driver


ROOT = Path(__file__).resolve().parents[1]
GRID = json.loads((ROOT / 'configs/all88_s092_sweep.json').read_text())['configurations']
PROTOCOL = dict(mode='all_subjects_train_single_subject_eval', subject='S092', expected_subjects=88)
COLUMNS = ['IBI', 'stage_W', 'stage_Light', 'stage_REM', 'stage_Deep', 'AHI', 'BMI', 'AGE', 'gender_F', 'gender_M']
SOURCES = (
    'training_wavestitch.py', 'training_utils.py', 'data_utils.py', 'watchsleep_loader.py',
    'cuda_runtime.py', 'epoch_validation.py', 'validation_metrics.py', 'validation_sampling.py',
    'validation_artifacts.py', 'window_config.py', 'diffusion_schedule.py', 'missing_data.py',
    'signal_normalization.py', 'training_input_dropout.py', 's4_dropout_config.py', 'host_runtime.py',
    'TSImputers/SSSDS4Imputer.py', 'TSImputers/S4Model.py', 'TSImputers/TimeGAN.py', 'utils/util.py',
    'third_party/WatchSleepNet/data_setup.py', 'scripts/evaluate_original_sampler.py',
    'scripts/evaluate_denoising.py', 'scripts/run_simulation.py', 'scripts/verify_upstream_boundary.py',
)
ARGS = dict(dataset='DREAMT', backbone='S4', beta_0=.0001, beta_T=.02, timesteps=200,
    diffusion_schedule='linear_power', terminal_alpha_bar=1e-8, hdim=64, lr=.0001,
    batch_size=8, epochs=50, layers=4, window_size=3000, stride=750,
    num_res_layers=4, res_channels=64, skip_channels=64, diff_step_embed_in=32,
    diff_step_embed_mid=64, diff_step_embed_out=64, s4_lmax=3000, s4_dstate=64,
    s4_dropout=0., s4_bidirectional=True, s4_layernorm=True, propCycEnc=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


class SweepBestBindingTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'work/sweep_best_binding_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.allowed_temp_root = parent.resolve()
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.base = Path(self.temp.name)
        self.sweep = self.base / 'live_sweep'
        self.sweep.mkdir()
        self.output = self.base / 'comparison'
        self.output.mkdir()
        self.records = {}
        self.make_configuration(GRID[0])

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        if not target.is_relative_to(self.allowed_temp_root) or target == self.allowed_temp_root:
            raise RuntimeError('Unsafe test cleanup path')
        self.temp.cleanup()

    def make_configuration(self, row, *, status='training', epoch=3):
        root = self.sweep / f"config_{row['id']}"
        workspace, training = root / 'run/workspace', root / 'run/training'
        workspace.mkdir(parents=True)
        (training / 'validation').mkdir(parents=True)
        norm = dict(method='minmax', min_seconds=0., max_seconds=2.)
        if row['normalization_range'][0] == 0:
            norm['output_range'] = [0., 1.]
        files = [f'S{i:03d}.npz' for i in range(1, 88)] + ['S092.npz']
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in files}
        scaler = dict(signal_mean_seconds=.89, signal_std_seconds=.31,
            continuous_columns=['AHI', 'BMI', 'AGE'], continuous_mean=[24., 33., 56.],
            continuous_std=[32., 8., 17.], continuous_raw_std=[32., 8., 17.],
            constant_columns=[], signal_fit_samples=264000, metadata_fit_subjects=88,
            fit_split='train', signal_normalization=norm,
            signal_center_seconds=1. if row['normalization_range'][0] == -1 else 0.,
            signal_scale_seconds=1. if row['normalization_range'][0] == -1 else 2.)
        identity = dict(schema='dreamt_data_identity_v1', schema_version=1, dataset='DREAMT',
            data_directory=str(self.base / 'data'), metadata_path=str(self.base / 'data/participant_info.csv'),
            metadata_file_sha256='1'*64, split_path=None, split_source='all_subjects',
            split_file_sha256=None, split_seed=42, split_fingerprint='2'*64,
            splits=dict(train=files, val=[], test=[]), sample_rate_hz=25, epoch_samples=750,
            window_epochs=4, multiplier=1., downsample_rate=1, max_epochs=None,
            columns=COLUMNS, stage_names=['W', 'Light', 'REM', 'Deep'],
            zero_value_policy='preserve stored zero missing-value markers in scaler and unmodified MSE; no imputation',
            raw_stage_mapping={'0': 0, '1': 1, '2': 1, '3': 3, '4': 2, 'other': -1},
            scaler=scaler, scaler_subjects=[Path(name).stem for name in files],
            train_npz_sha256=hashes, training_npz_sha256=hashes,
            signal_normalization=norm, evaluation_protocol=PROTOCOL, evaluation_filenames=['S092.npz'])
        config = dict(data_dir=identity['data_directory'], metadata_path=identity['metadata_path'],
            evaluation_protocol=PROTOCOL, split_path=None, generate_splits=False, split_seed=42,
            multiplier=1., downsample_rate=1, max_epochs=None, window_epochs=4,
            missing_value_policy='preserve', signal_normalization=norm,
            training_input_dropout=dict(probability=row['input_probability'], seed=42),
            s4_dropout=row['s4_probability'], diffusion_schedule='linear_power', terminal_alpha_bar=1e-8,
            validation=dict(max_windows=None, batch_size=8, seed=20261020, kl_variance_floor_ms2=1.,
                            reference_mask_policy='observed_only', save_artifacts=True),
            early_stopping=dict(enabled=False, patience=10, selection_mode='current', smoothing_window=1, min_delta_kl=0.))
        sources = {}
        for name in SOURCES:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            sources[name] = digest(target)
        write_json(workspace / 'dataset_config.json', config)
        sources['dataset_config.json'] = digest(workspace / 'dataset_config.json')
        args = dict(ARGS, s4_dropout=row['s4_probability'], validation_output=str(training / 'validation'))
        checkpoint = training / 'best_kl_model.pth'
        checkpoint.write_bytes(f'ARTIFICIAL BYTES, NOT A MODEL: config {row["id"]}, epoch {epoch}'.encode())
        best = dict(epoch=epoch, metric='gaussian_kl_real_to_generated', selection_rule='current_kl',
            value=.4, raw_metrics={'gaussian_kl_real_to_generated': .4},
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        validation = dict(completed_epochs=epoch+1, best_checkpoints={'kl': best},
            early_stopping=dict(options=config['early_stopping'], improvement_rule='current_kl'))
        manifest = dict(status=status, training_args=args, source_sha256=sources,
            data_identity=identity, diffusion_identity=diffusion_identity(args),
            training_input_dropout=dict(options=config['training_input_dropout']),
            s4_dropout=row['s4_probability'], missing_value_policy='preserve',
            model_input_columns=COLUMNS, checkpoint=str(workspace / 'saved_models/DREAMT/model.pth'))
        write_json(training / 'run_manifest.json', manifest)
        write_json(training / 'validation/summary.json', validation)
        record = dict(root=root, workspace=workspace, training=training, manifest=manifest,
                      validation=validation, checkpoint=checkpoint, configuration=deepcopy(row))
        self.records[row['id']] = record
        return record

    def freeze(self, configuration_id=0, suffix=''):
        destination = self.output / f'config_{configuration_id}{suffix}'
        destination.mkdir()
        return driver.freeze_configuration(self.sweep, GRID[configuration_id], destination,
                                           attempts=3, retry_seconds=0)

    def update_manifest(self, manifest, configuration_id=0):
        write_json(self.records[configuration_id]['training'] / 'run_manifest.json', manifest)

    def test_running_best_is_frozen_without_best_manifest_or_final_weights(self):
        record = self.records[0]
        self.assertFalse((record['training'] / 'best_kl_manifest.json').exists())
        context = self.freeze()
        copied = Path(context['checkpoint'])
        self.assertTrue(copied.is_relative_to(self.output))
        self.assertEqual(copied.read_bytes(), record['checkpoint'].read_bytes())
        self.assertEqual(context['epoch'], 3)
        self.assertEqual(context['checkpoint_sha256'], digest(copied))
        self.assertEqual(context['manifest']['checkpoint_epoch'], 3)
        self.assertEqual(context['manifest']['checkpoint_sha256'], digest(copied))
        self.assertEqual(Path(context['workspace']), record['workspace'])
        self.assertEqual(digest(context['manifest_path']), context['best_manifest_sha256'])
        self.assertEqual(context['manifest']['training_args'], record['manifest']['training_args'])

    def test_completed_run_still_selects_live_best_not_final_epoch_or_stale_best_manifest(self):
        record = self.records[0]
        manifest = deepcopy(record['manifest'])
        manifest.update(status='completed', checkpoint_epoch=50, checkpoint_sha256='f'*64)
        self.update_manifest(manifest)
        (record['training'] / 'best_kl_manifest.json').write_text('stale incomplete file', encoding='utf-8')
        context = self.freeze()
        self.assertEqual(context['epoch'], 3)
        self.assertEqual(context['checkpoint_sha256'], digest(record['checkpoint']))

    def test_missing_best_never_falls_back_to_final(self):
        record = self.records[0]
        record['checkpoint'].unlink()
        final = record['workspace'] / 'saved_models/DREAMT/model.pth'
        final.parent.mkdir(parents=True)
        final.write_bytes(b'FINAL MODEL MUST NOT BE READ')
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            self.freeze()

    def test_snapshot_stays_bound_after_training_updates_live_best(self):
        record = self.records[0]
        context = self.freeze()
        frozen_bytes = Path(context['checkpoint']).read_bytes()
        record['checkpoint'].write_bytes(b'A later checkpoint from ongoing training')
        changed = deepcopy(record['validation'])
        changed['completed_epochs'] = 6
        changed['best_checkpoints']['kl'].update(epoch=6, value=.2, checkpoint_sha256=digest(record['checkpoint']))
        write_json(record['training'] / 'validation/summary.json', changed)
        self.assertEqual(Path(context['checkpoint']).read_bytes(), frozen_bytes)
        self.assertEqual(context['epoch'], 3)
        self.assertEqual(context['checkpoint_sha256'], digest(context['checkpoint']))
        reread = driver.load_snapshot_context(Path(context['manifest_path']).parent.parent / 'context.json')
        self.assertEqual(reread['epoch'], 3)
        self.assertEqual(reread['checkpoint_sha256'], context['checkpoint_sha256'])

    def test_reloading_snapshot_rejects_changed_weights_or_detached_context_identity(self):
        context = self.freeze()
        context_path = Path(context['manifest_path']).parent.parent / 'context.json'
        copied = Path(context['checkpoint'])
        original = copied.read_bytes()
        copied.write_bytes(b'CHANGED COPIED WEIGHTS')
        with self.assertRaises(ValueError):
            driver.load_snapshot_context(context_path)
        copied.write_bytes(original)
        detached = deepcopy(context)
        detached['identity']['scaler']['continuous_mean'][0] = 999.
        write_json(context_path, detached)
        with self.assertRaises(ValueError):
            driver.load_snapshot_context(context_path)

    def test_frozen_imports_restore_both_dropout_locations_without_loading_a_model(self):
        # Each actual worker must import its own frozen namespace in a fresh
        # process. Tiny fake weight files make any deserialization a hard error.
        code = r'''
import json
from pathlib import Path
import sys
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('This integration test must not deserialize weights or run a model')
torch.load = forbidden
torch.nn.Module.__call__ = forbidden
from scripts import compare_s092_sweep_best as driver
context = driver.load_snapshot_context(Path(sys.argv[1]))
modules, args, provenance = driver.shared.frozen_imports(context)
print(json.dumps(dict(epoch=context['epoch'], s4_dropout=args.s4_dropout,
    input_dropout=context['manifest']['training_input_dropout']['options']['probability'],
    window_size=args.window_size, dataset=args.dataset,
    sampler_source=modules['validation_sampling'].__file__,
    recorded_status=context['manifest']['status'],
    s092_in_train='S092.npz' in context['identity']['splits']['train'])))
'''
        for configuration_id in (2, 5):
            row = GRID[configuration_id]
            record = self.make_configuration(row)
            context = self.freeze(configuration_id)
            context_path = Path(context['manifest_path']).parent.parent / 'context.json'
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-B', '-c', code, str(context_path)],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(actual['s4_dropout'], row['s4_probability'])
            self.assertEqual(actual['input_dropout'], row['input_probability'])
            self.assertEqual(actual['window_size'], 3000)
            self.assertEqual(actual['dataset'], 'DREAMT')
            self.assertTrue(actual['s092_in_train'])
            self.assertEqual(Path(actual['sampler_source']).resolve(), record['workspace'] / 'validation_sampling.py')

    def test_live_summary_change_during_weight_read_retries_consistently(self):
        record = self.records[0]
        original_read = Path.read_bytes
        replacement = b'ARTIFICIAL NEW BEST EPOCH FIVE'
        changed = False

        def changing_read(path):
            nonlocal changed
            result = original_read(path)
            if path.resolve() == record['checkpoint'].resolve() and not changed:
                changed = True
                record['checkpoint'].write_bytes(replacement)
                summary = deepcopy(record['validation'])
                summary['completed_epochs'] = 5
                summary['best_checkpoints']['kl'].update(epoch=5, value=.2,
                    raw_metrics={'gaussian_kl_real_to_generated': .2},
                    checkpoint_sha256=hashlib.sha256(replacement).hexdigest())
                write_json(record['training'] / 'validation/summary.json', summary)
            return result

        with mock.patch.object(Path, 'read_bytes', changing_read):
            context = self.freeze()
        self.assertTrue(changed)
        self.assertEqual(context['epoch'], 5)
        self.assertEqual(Path(context['checkpoint']).read_bytes(), replacement)

    def test_persistent_live_hash_mismatch_is_rejected(self):
        self.records[0]['checkpoint'].write_bytes(b'UNRECORDED WEIGHTS')
        with self.assertRaises((ValueError, RuntimeError)):
            self.freeze()
        self.assertFalse(any(self.output.rglob('best_snapshot_manifest.json')))

    def test_selection_requires_current_finite_kl_and_a_completed_epoch(self):
        record = self.records[0]
        mutations = [dict(selection_rule='smoothed_kl'), dict(epoch=0), dict(epoch=True),
                     dict(epoch=99), dict(value=float('nan')), dict(value=-.1),
                     dict(metric='mean_error_ms')]
        for number, change in enumerate(mutations):
            summary = deepcopy(record['validation'])
            summary['best_checkpoints']['kl'].update(change)
            write_json(record['training'] / 'validation/summary.json', summary)
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                self.freeze(suffix=f'_{number}')

    def test_sources_must_exist_match_and_stay_inside_frozen_workspace(self):
        record = self.records[0]
        original = deepcopy(record['manifest'])
        variants = []
        bad = deepcopy(original)
        bad['source_sha256']['training_utils.py'] = '0'*64
        variants.append(bad)
        bad = deepcopy(original)
        del bad['source_sha256']['signal_normalization.py']
        variants.append(bad)
        bad = deepcopy(original)
        bad['source_sha256']['TSImputers\\S4Model.py'] = '0'*64
        variants.append(bad)
        outside = record['workspace'].parent / 'outside.py'
        outside.write_text('# outside frozen workspace', encoding='utf-8')
        bad = deepcopy(original)
        bad['source_sha256']['../outside.py'] = digest(outside)
        variants.append(bad)
        for index, manifest in enumerate(variants):
            self.update_manifest(manifest)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                self.freeze(suffix=f'_{index}')

    def test_all88_protocol_normalization_and_dropout_must_match_configuration(self):
        original = self.records[0]['manifest']
        mutations = [
            lambda m: m['data_identity']['splits']['train'].remove('S092.npz'),
            lambda m: m['data_identity']['splits']['val'].append('S092.npz'),
            lambda m: m['data_identity']['evaluation_protocol'].update(mode='holdout'),
            lambda m: m['data_identity']['scaler'].update(signal_scale_seconds=2.),
            lambda m: m['data_identity']['signal_normalization'].update(output_range=[0., 1.]),
            lambda m: m['training_args'].update(s4_dropout=.03),
            lambda m: m['training_input_dropout']['options'].update(probability=.05),
        ]
        for index, mutate in enumerate(mutations):
            manifest = deepcopy(original)
            mutate(manifest)
            self.update_manifest(manifest)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                self.freeze(suffix=f'_{index}')

    def all_contexts(self):
        for row in GRID[1:]:
            self.make_configuration(row, epoch=row['id']+2)
        return [self.freeze(row['id']) for row in GRID]

    def test_comparison_accepts_all_twelve_ranges_dropouts_and_distinct_best_epochs(self):
        contexts = self.all_contexts()
        checks = driver.compare_contexts(contexts)
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()))
        self.assertEqual({c['configuration']['id'] for c in contexts}, set(range(12)))
        self.assertGreater(len({c['epoch'] for c in contexts}), 1)

    def test_comparison_rejects_nonexperimental_data_schedule_model_changes(self):
        contexts = self.all_contexts()
        mutations = [
            lambda c: c['identity'].update(metadata_file_sha256='0'*64),
            lambda c: c['identity']['scaler']['continuous_mean'].__setitem__(0, 999.),
            lambda c: c['identity']['columns'].reverse(),
            lambda c: c['identity']['train_npz_sha256'].update({'S092.npz': '0'*64}),
            lambda c: c['manifest']['diffusion_identity'].update(terminal_alpha_bar=.1),
            lambda c: c['manifest']['training_args'].update(lr=.001),
            lambda c: c['source_checks'].update({'TSImputers/S4Model.py': '0'*64}),
        ]
        for mutate in mutations:
            altered = deepcopy(contexts)
            mutate(altered[1])
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, RuntimeError)):
                driver.compare_contexts(altered)


if __name__ == '__main__':
    unittest.main()
