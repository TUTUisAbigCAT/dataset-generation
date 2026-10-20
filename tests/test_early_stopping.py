"""Early-stop decisions and the real launcher loop with explicit CPU test mocks."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epoch_validation import EarlyStopping, early_stopping_options, EpochValidator
from scripts import run_dreamt, run_simulation
from scripts.evaluate_denoising import resolve_checkpoint, load_training_run
from validation_metrics import summarize_distribution


def metrics(kl):
    return dict(gaussian_kl_real_to_generated=kl)


class EarlyStoppingTests(unittest.TestCase):
    def test_default_warmup_and_ten_plateau_epochs_stop_at_fifteen(self):
        stopping = EarlyStopping({})
        for epoch in range(1, 16):
            result = stopping.update(epoch, metrics(10.))
            self.assertEqual(result['bad_epochs'], max(0, epoch - 5))
            self.assertEqual(result['stopped_early'], epoch == 15)
            self.assertEqual(result['smoothing_ready'], epoch >= 5)
            self.assertEqual(result['metric_improvement'], epoch == 5)
            self.assertNotIn('joint_improvement', result)
            self.assertEqual(result['improvement_rule'], 'smoothed_kl')
            self.assertEqual(result['smoothed_metrics'], metrics(10.) if epoch >= 5 else metrics(None))
        self.assertEqual(result['best_metrics']['kl']['epoch'], 1)
        with self.assertRaises(ValueError):
            stopping.update(16, metrics(9.))

    def test_sliding_window_eviction_and_snapshot_independence(self):
        stopping = EarlyStopping({'early_stopping': {'smoothing_window': 3}})
        results = [stopping.update(i, metrics(value)) for i, value in enumerate((3, 6, 9, 0), 1)]
        self.assertEqual(results[1]['smoothed_metrics'], metrics(None))
        self.assertEqual(results[2]['smoothed_metrics'], metrics(6.))
        self.assertEqual(results[3]['smoothed_metrics'], metrics(5.))
        self.assertTrue(results[3]['metric_improvement'])
        self.assertEqual(results[0]['best_metrics']['kl']['value'], 3.)

    def test_smoothed_improvement_can_disagree_with_raw_direction(self):
        stopping = EarlyStopping({'early_stopping': {'smoothing_window': 3}})
        # Raw rises 0->9 while SMA falls 5->4; then raw falls 9->8 while SMA rises.
        results = [stopping.update(i, metrics(value)) for i, value in enumerate((12, 3, 0, 9, 8), 1)]
        self.assertEqual(results[2]['smoothed_metrics'], metrics(5.))
        self.assertEqual(results[3]['smoothed_metrics'], metrics(4.))
        self.assertTrue(results[3]['metric_improvement'])
        self.assertEqual(results[3]['best_metrics']['kl']['value'], 0.)
        self.assertFalse(results[4]['metric_improvement'])
        self.assertEqual(results[4]['bad_epochs'], 1)

    def test_single_kl_improvement_resets_patience(self):
        stopping = EarlyStopping({'early_stopping': {'patience': 3, 'smoothing_window': 1}})
        results = [stopping.update(i, metrics(value)) for i, value in enumerate((10, 9, 10, 8, 8, 9, 8), 1)]
        self.assertEqual([r['bad_epochs'] for r in results], [0, 0, 1, 0, 1, 2, 3])
        self.assertEqual([r['metric_improvement'] for r in results], [True, True, False, True, False, False, False])
        self.assertEqual(results[-1]['stopped_epoch'], 7)
        self.assertEqual(results[-1]['best_metrics']['kl']['epoch'], 4)
        self.assertEqual(results[0]['best_metrics']['kl']['epoch'], 1)

    def test_threshold_is_strict_and_small_improvements_accumulate(self):
        stopping = EarlyStopping({'early_stopping': {
            'patience': 4, 'smoothing_window': 1, 'min_delta_kl': 1.}})
        results = [stopping.update(i, metrics(value)) for i, value in enumerate((10, 9, 8.9, 8, 7.8), 1)]
        self.assertFalse(results[1]['metric_improvement'])  # Exactly min_delta is insufficient.
        self.assertEqual(results[1]['best_metrics']['kl']['value'], 9.)
        self.assertEqual([r['bad_epochs'] for r in results], [0, 1, 0, 1, 0])
        self.assertTrue(results[2]['metric_improvement'])
        self.assertFalse(results[3]['metric_improvement'])
        self.assertTrue(results[4]['metric_improvement'])

    def test_obsolete_mean_variance_values_do_not_control_stopping(self):
        stopping = EarlyStopping({'early_stopping': {'patience': 5, 'smoothing_window': 1}})
        for epoch in range(1, 7):
            row = dict(**metrics(100.), mean_vector_l2_ms=100. - epoch, variance_vector_l2_ms2=1000. - epoch)
            result = stopping.update(epoch, row)
            self.assertEqual(result['bad_epochs'], epoch - 1)
            self.assertEqual(result['stopped_early'], epoch == 6)

    def test_disabled_stopping_and_zero_kl(self):
        stopping = EarlyStopping({'early_stopping': {'enabled': False, 'patience': 1, 'smoothing_window': 1}})
        for epoch in range(1, 9):
            self.assertFalse(stopping.update(epoch, metrics(0.))['stopped_early'])
        self.assertEqual(stopping.summary()['bad_epochs'], 7)

    def test_legacy_zero_thresholds_migrate_without_reintroducing_old_keys(self):
        for old in ({'min_delta_mean_ms': 0}, {'min_delta_variance_ms2': 0.},
                    {'min_delta_mean_ms': 0., 'min_delta_variance_ms2': 0}):
            with self.subTest(legacy=old):
                result = early_stopping_options({'early_stopping': {**old, 'min_delta_kl': .2}})
                self.assertEqual(result['min_delta_kl'], .2)
                self.assertNotIn('min_delta_mean_ms', result)
                self.assertNotIn('min_delta_variance_ms2', result)
        for key in ('min_delta_mean_ms', 'min_delta_variance_ms2'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                early_stopping_options({'early_stopping': {key: 1.}})

    def test_rejects_invalid_options_metrics_and_epoch_order(self):
        invalid = [None, [], {'unknown': 1}, {'enabled': 1}, {'patience': 0},
                   {'patience': True}, {'patience': 1.5}]
        invalid += [{'min_delta_kl': value} for value in (-1, float('nan'), float('inf'), True, '1', None)]
        invalid += [{key: value} for key in ('min_delta_mean_ms', 'min_delta_variance_ms2')
                    for value in (-1, float('nan'), float('inf'), True, '0', None)]
        invalid += [{'smoothing_window': value} for value in (0, -1, True, 1.5, '5', None)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                early_stopping_options({'early_stopping': options})
        stopping = EarlyStopping({})
        for epoch, value in ((2, 1), (1, -1), (1, float('nan')), (1, float('inf'))):
            with self.subTest(epoch=epoch, value=value), self.assertRaises(ValueError):
                stopping.update(epoch, metrics(value))
        self.assertEqual(stopping.summary()['completed_epochs'], 0)

    def test_completion_accepts_only_full_validated_epochs(self):
        validation = dict(completed_epochs=3, early_stopping=dict(stopped_early=True, stopped_epoch=3))
        result = run_simulation.validate_completion(validation, 6, 2, 10)
        self.assertEqual(result['completed_epochs'], 3)
        self.assertEqual(result['stop_reason'], 'early_stopping')
        for steps, requested in ((5, 10), (7, 10), (6, 2)):
            with self.assertRaises(RuntimeError):
                run_simulation.validate_completion(validation, steps, 2, requested)
        validation['early_stopping'] = dict(stopped_early=False, stopped_epoch=None)
        with self.assertRaisesRegex(RuntimeError, 'without an early-stopping decision'):
            run_simulation.validate_completion(validation, 6, 2, 10)
        self.assertEqual(run_simulation.validate_completion(validation, 6, 2, 3)['stop_reason'], 'max_epochs')


class TinyTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.05))

    def forward(self, batch, time):
        return (batch[:, :, :1] * self.weight).permute(0, 2, 1)


class LauncherEarlyStoppingTests(unittest.TestCase):
    def test_launcher_stops_normally_saves_best_and_final_and_supports_disabled_mode(self):
        # Actual training script, trace observer, validation metrics and file I/O.
        # CUDA guards and generation are mocked explicitly; this is not a GPU run.
        for enabled, requested, completed in ((True, 8, 7), (False, 7, 7), (False, 3, 3)):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                workspace, output = base / 'workspace', base / 'training'
                run_dreamt.snapshot_source(workspace)
                config = dict(data_dir=str(ROOT / 'example_data'), split_seed=42,
                              early_stopping=dict(enabled=enabled, patience=2))
                config_path = workspace / 'dataset_config.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                env = {key: value for key, value in os.environ.items() if not key.startswith('DREAMT_')}
                env['DREAMT_CONFIG_PATH'] = str(config_path)
                argv = ['run_simulation.py', '--dataset', 'DREAMT', '--epochs', str(requested),
                        '--batch-size', '4', '--output-dir', str(output)]

                def generate(validator, model):
                    return summarize_distribution(validator.reference, validator.reference), {'unit_test_mock': True}, []

                with mock.patch.dict(os.environ, env, clear=True), \
                     mock.patch.object(sys, 'argv', argv), \
                     mock.patch.object(run_simulation, 'ROOT', workspace), \
                     mock.patch.object(run_simulation, 'require_cuda', return_value=torch.device('cpu')), \
                     mock.patch.object(run_simulation, 'cuda_environment', return_value={'device': 'cpu_test_mock'}), \
                     mock.patch('cuda_runtime.require_cuda', return_value=torch.device('cpu')), \
                     mock.patch('cuda_runtime.assert_cuda_model'), \
                     mock.patch('epoch_validation._require_device', return_value=torch.device('cpu')), \
                     mock.patch('training_utils.fetchModel', side_effect=lambda *args: TinyTestModel()), \
                     mock.patch.object(EpochValidator, '_generate', generate), \
                     mock.patch('torch.cuda.synchronize'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run_simulation.main()

                read = lambda name: json.loads((output / name).read_text(encoding='utf-8'))
                report, manifest = read('training_report.json'), read('run_manifest.json')
                self.assertEqual(report['status'], 'completed')
                self.assertEqual(report['epochs'], completed)
                self.assertEqual(report['requested_epochs'], requested)
                self.assertEqual(report['updates'], 3 * completed)
                self.assertEqual(manifest['training_args']['stride'], 750)
                self.assertEqual(manifest['train_windows'], 10)
                self.assertEqual(report['stopped_early'], enabled)
                self.assertEqual(manifest['checkpoint_epoch'], completed)
                self.assertEqual(manifest['completed_epochs'], completed)
                self.assertFalse((output / 'failure.json').exists())
                self.assertEqual(len(list((output / 'validation').glob('epoch_*'))), completed)
                final = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=True)
                for obsolete in ('mean', 'variance', 'joint'):
                    self.assertFalse((output / f'best_{obsolete}_model.pth').exists())
                    self.assertFalse((output / f'best_{obsolete}_manifest.json').exists())
                with (output / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    self.assertEqual(reader.fieldnames, [
                        'epoch', 'train_mean_loss', 'gaussian_kl_real_to_generated',
                        'smoothed_gaussian_kl_real_to_generated', 'validation_seconds',
                        'smoothing_ready', 'metric_improvement', 'early_stopping_bad_epochs', 'stopped_early'])
                self.assertEqual(len(rows), completed)
                self.assertEqual([row['smoothed_gaussian_kl_real_to_generated'] for row in rows[:4]],
                                 [''] * min(4, completed))
                if completed < 5:
                    self.assertFalse((output / 'best_kl_model.pth').exists())
                    self.assertFalse((output / 'best_kl_manifest.json').exists())
                else:
                    best_path = output / 'best_kl_manifest.json'
                    _, selected, _ = load_training_run(best_path)
                    checkpoint, identity = resolve_checkpoint(best_path, selected)
                    self.assertTrue(identity['recorded_digest_verified'])
                    self.assertEqual(selected['checkpoint_epoch'], 5)
                    selection = selected['checkpoint_selection']
                    self.assertEqual(selection['metric'], 'gaussian_kl_real_to_generated')
                    self.assertEqual(selection['selection_rule'], 'smoothed_kl')
                    self.assertEqual(selection['smoothing_window'], 5)
                    self.assertEqual(selection['window_start_epoch'], 1)
                    self.assertEqual(selection['value'], 0.)
                    self.assertEqual(selection['raw_metrics'], metrics(0.))
                    self.assertEqual(rows[4]['metric_improvement'], 'True')
                    self.assertEqual(float(rows[4]['smoothed_gaussian_kl_real_to_generated']), 0.)
                    best = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    self.assertFalse(torch.equal(best['weight'], final['weight']))


if __name__ == '__main__':
    unittest.main()
