"""Early-stop decisions and the real launcher loop with explicit CPU test mocks."""
import contextlib
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


def metrics(mean, variance):
    return dict(mean_window_absolute_error_ms=mean, variance_window_absolute_error_ms2=variance)


class EarlyStoppingTests(unittest.TestCase):
    def test_default_warmup_and_ten_plateau_epochs_stop_at_fifteen(self):
        stopping = EarlyStopping({})
        for epoch in range(1, 16):
            result = stopping.update(epoch, metrics(10., 1000000.))
            self.assertEqual(result['bad_epochs'], max(0, epoch - 5))
            self.assertEqual(result['stopped_early'], epoch == 15)
            self.assertEqual(result['smoothing_ready'], epoch >= 5)
            self.assertEqual(result['joint_improvement'], epoch == 5)
            self.assertEqual(result['smoothed_metrics'], metrics(10., 1000000.) if epoch >= 5 else metrics(None, None))
        self.assertEqual(result['best_metrics']['mean']['epoch'], 1)
        with self.assertRaises(ValueError):
            stopping.update(16, metrics(9., 0.))

    def test_sliding_window_eviction_and_snapshot_independence(self):
        stopping = EarlyStopping({'early_stopping': {'smoothing_window': 3}})
        results = [stopping.update(i, metrics(value, value * 10))
                   for i, value in enumerate((3, 6, 9, 0), 1)]
        self.assertEqual(results[1]['smoothed_metrics'], metrics(None, None))
        self.assertEqual(results[2]['smoothed_metrics'], metrics(6., 60.))
        self.assertEqual(results[3]['smoothed_metrics'], metrics(5., 50.))
        self.assertTrue(results[3]['joint_improvement'])

    def test_smoothed_improvement_can_disagree_with_raw_direction(self):
        stopping = EarlyStopping({'early_stopping': {'smoothing_window': 3}})
        # Raw rises 0->9 while SMA falls 5->4; then raw falls 9->8 while SMA rises.
        results = [stopping.update(i, metrics(value, value * 10))
                   for i, value in enumerate((12, 3, 0, 9, 8), 1)]
        self.assertEqual(results[2]['smoothed_metrics'], metrics(5., 50.))
        self.assertEqual(results[3]['smoothed_metrics'], metrics(4., 40.))
        self.assertTrue(results[3]['joint_improvement'])
        self.assertEqual(results[3]['new_best_metrics'], [])
        self.assertFalse(results[4]['joint_improvement'])
        self.assertEqual(results[4]['bad_epochs'], 1)

    def test_only_simultaneous_improvement_resets_patience(self):
        stopping = EarlyStopping({'early_stopping': {'patience': 3, 'smoothing_window': 1}})
        values = [(10, 100), (9, 100), (9, 90), (8, 80), (7, 90), (8, 70), (7, 70)]
        results = [stopping.update(i, metrics(*pair)) for i, pair in enumerate(values, 1)]
        self.assertEqual([r['bad_epochs'] for r in results], [0, 1, 2, 0, 1, 2, 3])
        self.assertEqual([r['joint_improvement'] for r in results], [True, False, False, True, False, False, False])
        self.assertEqual(results[2]['improved_metrics'], ['variance'])
        self.assertEqual(results[4]['improved_metrics'], ['mean'])
        self.assertEqual(results[-1]['stopped_epoch'], 7)
        self.assertEqual(results[-1]['best_metrics']['mean']['epoch'], 5)
        self.assertEqual(results[-1]['best_metrics']['variance']['epoch'], 6)
        # Earlier rows must remain immutable when later best values change.
        self.assertEqual(results[0]['best_metrics']['mean']['epoch'], 1)

    def test_threshold_is_strict_and_small_improvements_accumulate(self):
        stopping = EarlyStopping({'early_stopping': {
            'patience': 4, 'smoothing_window': 1, 'min_delta_mean_ms': 1., 'min_delta_variance_ms2': 5.}})
        stopping.update(1, metrics(10, 100))
        result = stopping.update(2, metrics(9, 95))
        self.assertEqual(result['improved_metrics'], [])  # Exactly min_delta is insufficient.
        self.assertEqual(set(result['new_best_metrics']), {'mean', 'variance'})
        self.assertEqual(result['threshold_anchors'], {'mean': 10., 'variance': 100.})
        result = stopping.update(3, metrics(8.9, 95))
        self.assertEqual(result['improved_metrics'], ['mean'])
        self.assertEqual(result['bad_epochs'], 2)
        self.assertFalse(result['joint_improvement'])
        result = stopping.update(4, metrics(8.9, 94.9))
        self.assertEqual(result['improved_metrics'], ['variance'])
        self.assertEqual(result['bad_epochs'], 3)
        result = stopping.update(5, metrics(7.8, 89.8))
        self.assertTrue(result['joint_improvement'])
        self.assertEqual(result['bad_epochs'], 0)

    def test_repeated_single_metric_improvement_cannot_prevent_stopping(self):
        for improving in ('mean', 'variance'):
            with self.subTest(improving=improving):
                stopping = EarlyStopping({'early_stopping': {'patience': 5, 'smoothing_window': 1}})
                for epoch in range(1, 7):
                    value = 100 - epoch
                    row = metrics(value, 100) if improving == 'mean' else metrics(100, value)
                    result = stopping.update(epoch, row)
                    self.assertEqual(result['bad_epochs'], epoch - 1)
                    self.assertEqual(result['stopped_early'], epoch == 6)

    def test_disabled_stopping_and_zero_errors(self):
        stopping = EarlyStopping({'early_stopping': {'enabled': False, 'patience': 1, 'smoothing_window': 1}})
        for epoch in range(1, 9):
            self.assertFalse(stopping.update(epoch, metrics(0., 0.))['stopped_early'])
        self.assertEqual(stopping.summary()['bad_epochs'], 7)

    def test_rejects_invalid_options_metrics_and_epoch_order(self):
        invalid = [None, [], {'unknown': 1}, {'enabled': 1}, {'patience': 0},
                   {'patience': True}, {'patience': 1.5}, {'min_delta_mean_ms': -1},
                   {'min_delta_variance_ms2': float('inf')}, {'min_delta_mean_ms': float('nan')},
                   {'min_delta_mean_ms': True}, {'min_delta_variance_ms2': '1'}]
        invalid += [{'smoothing_window': value} for value in (0, -1, True, 1.5, '5', None)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                early_stopping_options({'early_stopping': options})
        stopping = EarlyStopping({})
        for epoch, value in ((2, 1), (1, -1), (1, float('nan')), (1, float('inf'))):
            with self.subTest(epoch=epoch, value=value), self.assertRaises(ValueError):
                stopping.update(epoch, metrics(value, 0))
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
                    return validator.reference.clone(), {'unit_test_mock': True}, []

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
                kinds = ('mean', 'variance', 'joint') if completed >= 5 else ('mean', 'variance')
                if completed < 5:
                    self.assertFalse((output / 'best_joint_model.pth').exists())
                    self.assertFalse((output / 'best_joint_manifest.json').exists())
                for metric in kinds:
                    best_path = output / f'best_{metric}_manifest.json'
                    _, selected, _ = load_training_run(best_path)
                    checkpoint, identity = resolve_checkpoint(best_path, selected)
                    self.assertTrue(identity['recorded_digest_verified'])
                    self.assertEqual(selected['checkpoint_epoch'], 5 if metric == 'joint' else 1)
                    if metric == 'joint':
                        selection = selected['checkpoint_selection']
                        self.assertEqual(selection['smoothing_window'], 5)
                        self.assertEqual(selection['window_start_epoch'], 1)
                        self.assertEqual(selection['value'], metrics(0., 0.))
                        self.assertEqual(selection['raw_metrics'], metrics(0., 0.))
                    best = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    self.assertFalse(torch.equal(best['weight'], final['weight']))


if __name__ == '__main__':
    unittest.main()
