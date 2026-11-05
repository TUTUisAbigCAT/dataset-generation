"""Explicit S092 inference subsets: failed runs are excluded only by request.

Snapshots contain deliberately invalid weight bytes. No model is loaded or run.
"""
from __future__ import annotations

from copy import deepcopy
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import compare_s092_sweep_best as driver
from tests import test_compare_s092_sweep_best as fixtures


class SweepSelectionTests(unittest.TestCase):
    def test_exported_window_rows_and_equal_weight_means_with_missing_samples(self):
        import numpy as np
        starts = [0, 3750]
        mask = np.ones((2, 3000), dtype=bool)
        mask[0, 2:] = False
        windows = [
            dict(start_sample=start, observed_counts=count,
                 reference_mean_ms=1000., generated_mean_ms=1000.+error, mean_error_ms=error,
                 mean_absolute_error_ms=abs(error), reference_variance_ms2=100.,
                 generated_variance_ms2=100.+v_error, variance_error_ms2=v_error,
                 variance_absolute_error_ms2=abs(v_error), gaussian_kl_real_to_generated=1.)
            for start, count, error, v_error in [(0, 2, 40., -80.), (3750, 3000, -60., 20.)]]
        metrics = {key: (windows[0][key]+windows[1][key])/2 for key in windows[0]}
        report = dict(configuration=dict(id=1), epoch=25, seed=42, per_window_metrics=windows, metrics=metrics)
        with tempfile.TemporaryDirectory(dir=driver.ROOT) as tmp:
            output = Path(tmp)
            np.savez(output/'reference.npz', start_samples=starts, window_size=3000,
                     sample_rate_hz=25, scoring_mask=mask.ravel())
            driver.write_window_tables(output, [report])
            with (output/'per_window_metrics.csv').open() as src:
                detail = list(csv.DictReader(src))
            self.assertEqual([row['start_second'] for row in detail], ['0.0', '150.0'])
            self.assertEqual([row['observed_samples'] for row in detail], ['2', '3000'])
            with (output/'all_window_summary.csv').open() as src:
                summary, = csv.DictReader(src)
            for key, expected in dict(avg_generated_mean_ms=990., avg_mean_error_ms=-10.,
                    avg_mean_absolute_error_ms=50., avg_generated_variance_ms2=70.,
                    avg_variance_error_ms2=-30., avg_variance_absolute_error_ms2=50.).items():
                self.assertEqual(float(summary[key]), expected)

    def test_default_keeps_all_twelve_and_explicit_selection_is_sorted(self):
        original = deepcopy(fixtures.GRID)
        self.assertEqual(driver.select_configurations(fixtures.GRID), fixtures.GRID)
        selected = driver.select_configurations(fixtures.GRID, [8, 0, 6])
        self.assertEqual([row['id'] for row in selected], [0, 6, 8])
        self.assertEqual(fixtures.GRID, original)

    def test_empty_duplicate_unknown_and_noninteger_ids_are_rejected(self):
        for ids in ([], [0, 0], [-1], [12], [True], ['6'], [6.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                driver.select_configurations(fixtures.GRID, ids)


class SweepSubsetOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SweepBestBindingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def make_nine(self):
        for row in fixtures.GRID[1:9]:
            self.fixture.make_configuration(row, epoch=row['id'] + 2)
        for index in (9, 10, 11):
            self.assertFalse((self.fixture.sweep / f'config_{index}').exists())

    def test_nine_contexts_are_paired_and_missing_requested_context_is_rejected(self):
        self.make_nine()
        contexts = [self.fixture.freeze(index) for index in range(9)]
        checks = driver.compare_contexts(contexts, expected_ids=list(range(9)))
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()))
        with self.assertRaises(ValueError):
            driver.compare_contexts(contexts, expected_ids=list(range(12)))
        with self.assertRaises(ValueError):
            driver.compare_contexts(contexts + [contexts[0]])
        with self.assertRaises(ValueError):
            driver.compare_contexts([])

    def test_check_only_reads_and_snapshots_only_the_nine_explicit_runs(self):
        import torch

        self.make_nine()
        output = self.fixture.base / 'selected_check'
        preflights = []
        real_read = Path.read_bytes

        def guarded_read(path):
            if any(part in ('config_9', 'config_10', 'config_11') for part in path.parts):
                raise AssertionError('An explicitly excluded run was read')
            return real_read(path)

        def worker(cli, context, phase, destination):
            self.assertEqual(phase, 'preflight')
            index = context['configuration']['id']
            self.assertEqual(destination, output / f'config_{index}')
            preflights.append(index)

        def inputs(destination, rows):
            self.assertEqual([row['id'] for row in rows], list(range(9)))
            (destination / 'reference.npz').write_bytes(b'ARTIFICIAL INPUT CHECK PLACEHOLDER')
            return {f'config_{row["id"]}': True for row in rows}

        argv = ['--sweep-root', str(self.fixture.sweep), '--output-dir', str(output),
                '--check-only', '--config-ids', *(str(index) for index in range(9))]
        with (mock.patch.object(Path, 'read_bytes', guarded_read),
              mock.patch.object(driver, 'launch_worker', side_effect=worker),
              mock.patch.object(driver, 'matching_inputs', side_effect=inputs),
              mock.patch.object(torch, 'load', side_effect=AssertionError('No weight loading')),
              mock.patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('No model execution')),
              mock.patch('cuda_runtime.require_cuda', side_effect=AssertionError('No model runtime in check-only')),
              contextlib.redirect_stdout(io.StringIO())):
            driver.main(argv)

        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'checked')
        self.assertEqual(summary['selected_configuration_ids'], list(range(9)))
        self.assertEqual(summary['excluded_configuration_ids'], [9, 10, 11])
        self.assertEqual([row['id'] for row in summary['configurations']], list(range(9)))
        self.assertEqual(set(summary['snapshots']), {str(index) for index in range(9)})
        self.assertEqual(preflights, list(range(9)))
        self.assertFalse(summary['sampling_performed'])
        for index in range(9):
            self.assertTrue((output / f'config_{index}/checkpoint_snapshot/best_kl_model.pth').is_file())
        for index in (9, 10, 11):
            self.assertFalse((output / f'config_{index}').exists())

    def test_explicitly_requested_missing_run_still_fails_without_auto_skipping(self):
        output = self.fixture.base / 'missing_requested_check'
        argv = ['--sweep-root', str(self.fixture.sweep), '--output-dir', str(output),
                '--check-only', '--config-ids', '0', '9']
        with (mock.patch.object(driver, 'launch_worker') as worker,
              mock.patch.object(driver.time, 'sleep'),
              contextlib.redirect_stdout(io.StringIO()),
              self.assertRaisesRegex(RuntimeError, 'config_9')):
            driver.main(argv)
        worker.assert_not_called()
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['selected_configuration_ids'], [0, 9])
        self.assertFalse(summary['sampling_performed'])


if __name__ == '__main__':
    unittest.main()
