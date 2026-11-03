"""Saved artificial output fixtures only; never construct or execute a model."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import plot_s092_sweep_best as plotter

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def synthetic_result(root):
    """Twelve labeled fake generation outputs, with all four original stages."""
    root = Path(root)
    root.mkdir(parents=True)
    configs = json.loads((ROOT/'configs/all88_s092_sweep.json').read_text())['configurations']
    stages = np.array([0, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 3], dtype=np.int64)
    time = np.arange(9000)/25
    real = (1.1+.18*np.sin(time*.5)+.08*np.cos(time*.09)).astype(np.float64)
    real[np.arange(9000) % 750 < 10] = 0
    conditions = np.zeros((9000, 9), dtype=np.float32)
    conditions[:, :4] = np.eye(4)[np.repeat(stages, 750)]
    conditions[:, 4:] = [.1, .2, .3, 1., 0.]
    arrays = dict(reference_ibi_seconds=real, scoring_mask=real > 0, conditions=conditions,
                  sleep_stages=stages, start_second=np.array(18900), sample_rate_hz=np.array(25),
                  window_size=np.array(3000))
    np.savez_compressed(root/'reference.npz', **arrays)
    for config in configs:
        path = root/f'config_{config["id"]}'
        path.mkdir()
        np.savez_compressed(path/'inputs.npz', **arrays)
        center, scale = (1., 1.) if config['normalization_range'][0] == -1 else (0., 2.)
        desired = real+.02*config['id']+.04*np.sin(time*.7)
        desired[0:4] = -.05-config['id']*.005
        normalized = ((desired-center)/scale).astype(np.float32)
        generated = normalized.astype(np.float64)*scale+center
        np.savez_compressed(path/'waveforms.npz', generated_ibi_seconds=generated,
                            generated_normalized=normalized, seed=np.array(20261103), window_size=np.array(3000))
        metrics = [plotter.paired_statistics(real[i:i+3000], generated[i:i+3000], arrays['scoring_mask'][i:i+3000])
                   for i in range(0, 9000, 3000)]
        aggregate = {key: float(np.mean([row[key] for row in metrics])) for key in
                     ('gaussian_kl_real_to_generated', 'mean_absolute_error_ms', 'variance_absolute_error_ms2')}
        write_json(path/'summary.json', dict(status='completed', configuration=config, epoch=config['id']+1,
            checkpoint_sha256=f'{config["id"]:064x}', seed=20261103,
            waveform_sha256=plotter.sha256(path/'waveforms.npz'), metrics=aggregate,
            sampling=dict(independent_windows=True, signal_all_unknown=True,
                          condition_columns_bitwise_preserved=True, clipping_applied=False)))
    write_json(root/'summary.json', dict(status='completed', configurations=configs, seed=20261103,
        reference_sha256=plotter.sha256(root/'reference.npz'), implementation_test_only=True))
    return root


class SweepPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = ROOT/'work/sweep_plot_tests'
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.base = Path(self.temp.name)
        self.result = synthetic_result(self.base/'result')

    def tearDown(self):
        if not self.base.resolve().is_relative_to(self.temp_root.resolve()):
            raise RuntimeError('Unsafe test cleanup path')
        self.temp.cleanup()

    def test_complete_pairing_and_stage_selection_preserve_all_values(self):
        data = plotter.load_result(self.result)
        self.assertEqual(data['stage_starts'], dict(W=0, Light=750, REM=3000, Deep=6000))
        self.assertEqual(len(data['generated']), 12)
        self.assertEqual(len(data['rows']), 12)
        self.assertEqual(len(data['window_rows']), 36)
        self.assertEqual(len(data['stage_rows']), 48)
        self.assertLess(data['ylim'][0], min(array.min() for array in data['generated'].values()))
        self.assertGreater(data['ylim'][1], max(array.max() for array in data['generated'].values()))
        self.assertTrue(all(array[0] < 0 for array in data['generated'].values()))
        self.assertFalse((self.result/'plots').exists())

    def test_gaussian_kl_uses_only_reference_mask_without_filtering_generated(self):
        real = np.array([0., 1., 3.])
        generated = np.array([999., -1., 1.])
        report = plotter.paired_statistics(real, generated, real > 0)
        self.assertAlmostEqual(report['gaussian_kl_real_to_generated'], 2.)
        self.assertEqual(report['generated_mean_ms'], 0.)
        self.assertEqual(report['mean_absolute_error_ms'], 2000.)
        self.assertEqual(report['observed_samples'], 2)
        equal = plotter.paired_statistics(real, real, real > 0)
        self.assertEqual(equal['gaussian_kl_real_to_generated'], 0.)

    def test_reject_incomplete_grid_wrong_seed_and_unsaved_reference(self):
        path = self.result/'summary.json'
        base = plotter.read_json(path)
        variants = [dict(base, configurations=base['configurations'][:-1]),
                    dict(base, seed=123), dict(base, reference_sha256='0'*64), dict(base, status='sampling')]
        duplicate = deepcopy(base)
        duplicate['configurations'][11]['id'] = 10
        variants.append(duplicate)
        for variant in variants:
            with self.subTest(variant=variant.keys()):
                write_json(path, variant)
                with self.assertRaises(ValueError):
                    plotter.render_comparison(self.result)
                self.assertFalse((self.result/'plots').exists())
        write_json(path, base)

    def test_final_config_bad_hash_or_metrics_rejected_before_any_output(self):
        path = self.result/'config_11/summary.json'
        base = plotter.read_json(path)
        cases = [dict(base, waveform_sha256='0'*64), dict(base, seed=123), dict(base, epoch=0),
                 dict(base, metrics=dict(base['metrics'], gaussian_kl_real_to_generated=999.)),
                 dict(base, sampling=dict(base['sampling'], independent_windows=False))]
        for variant in cases:
            with self.subTest(variant=variant.keys()):
                write_json(path, variant)
                with self.assertRaises(ValueError):
                    plotter.render_comparison(self.result)
                self.assertFalse((self.result/'plots').exists())
        write_json(path, base)

    def test_mismatched_condition_inputs_are_rejected(self):
        path = self.result/'config_7/inputs.npz'
        arrays = plotter.read_npz(path)
        arrays['conditions'][0, 4] += 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'reference/conditions/mask'):
            plotter.load_result(self.result)

    def test_bad_units_or_nonfinite_generated_are_rejected_even_with_updated_hash(self):
        directory = self.result/'config_2'
        path = directory/'waveforms.npz'
        base = plotter.read_npz(path)
        report = plotter.read_json(directory/'summary.json')
        for mode in ('milliseconds', 'nan'):
            arrays = {key: value.copy() for key, value in base.items()}
            if mode == 'milliseconds':
                arrays['generated_ibi_seconds'] *= 1000
            else:
                arrays['generated_ibi_seconds'][42] = np.nan
            np.savez_compressed(path, **arrays)
            write_json(directory/'summary.json', dict(report, waveform_sha256=plotter.sha256(path)))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plotter.load_result(self.result)

    def test_actual_detail_and_full_figures_keep_all_points_and_shared_limits(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = plotter.load_result(self.result)
        for details, panels, points in ((True, 4, 750), (False, 3, 3000)):
            fig = plotter._figure(data, data['configs'][0], details=details, synthetic_fixture=True)
            try:
                self.assertEqual(len(fig.axes), panels*2)
                self.assertIn('SYNTHETIC FIXTURE', fig._suptitle.get_text())
                for axis in fig.axes[::2]:
                    self.assertEqual(len(axis.lines[1].get_xdata()), points)
                    self.assertEqual(axis.get_ylim(), data['ylim'])
                self.assertLess(fig.axes[0].lines[1].get_ydata()[0], 0)
                self.assertGreater(len(fig.axes[0].collections[0].get_offsets()), 0)
            finally:
                plt.close(fig)

    def test_existing_output_directory_is_not_overwritten(self):
        output = self.result/'plots'
        output.mkdir()
        sentinel = output/'existing.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            plotter.render_comparison(self.result)
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
