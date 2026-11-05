"""Exercise worker batching using supplied waveforms. No model is executed."""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import validation_metrics
from scripts import compare_s092_best_minmax as worker


class BatchedWorkerTest(unittest.TestCase):
    def test_partial_batch_window_offsets_paired_mask_and_equal_window_weight(self):
        window = 3000
        starts = np.array([0, 3000, 7500, 10500, 15000])
        real = np.broadcast_to(np.linspace(.6, 1.4, window), (5, window)).copy()
        normalized = (real * np.array([.9, .8, 1., 1.1, 1.8])[:, None]).astype(np.float32)
        real[1, :1000] = 0
        real[-1, 2:] = 0
        mask = real > 0
        conditions = np.zeros((5, window, 9), dtype=np.float32)
        conditions[:, :, 4] = np.arange(5)[:, None]
        batch_ids = []

        def supplied_waveforms(model, inputs, schedule, args, device):
            indices = inputs[:, 0, 5].to(torch.int64).tolist()
            batch_ids.append(indices)
            self.assertTrue(bool((inputs[:, :, 0] == 0).all()))
            self.assertTrue(np.array_equal(inputs[:, :, 1:].numpy(), conditions[indices]))
            return torch.from_numpy(normalized[indices]), dict(
                windows=len(indices), mask_shape=[len(indices), window], denoiser_calls=200,
                independent_windows=True, condition_columns_bitwise_preserved=True)

        dummy_model = mock.Mock()
        dummy_model.to.return_value = dummy_model
        modules = dict(training_utils=SimpleNamespace(fetchModel=lambda *args: dummy_model),
            cuda_runtime=SimpleNamespace(assert_cuda_model=lambda model: None, cuda_environment=lambda d: {}),
            night_checkpoint=SimpleNamespace(load_checkpoint_state=lambda model, saved: {}),
            diffusion_schedule=SimpleNamespace(build_diffusion_config=lambda args: {}),
            validation_sampling=SimpleNamespace(sample_validation_windows=supplied_waveforms),
            validation_metrics=validation_metrics)
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(random.setstate, random.getstate())
        with tempfile.TemporaryDirectory(dir=worker.ROOT) as tmp, torch.random.fork_rng(devices=[]):
            root = Path(tmp)
            weights = root / 'fake.pth'
            weights.write_bytes(b'fixture; never deserialized')
            context = dict(checkpoint=str(weights), checkpoint_sha256=worker.sha256(weights),
                           best_manifest_sha256='a'*64, epoch=25)
            for batch, expected_ids in ((2, [[0, 1], [2, 3], [4]]), (None, [[0, 1, 2, 3, 4]])):
                with self.subTest(batch=batch):
                    output = root / str(batch)
                    output.mkdir()
                    batch_ids.clear()
                    coordinates = dict(start_samples=starts) if batch else {}
                    np.savez(output/'inputs.npz', reference_ibi_seconds=real.ravel(), scoring_mask=mask.ravel(),
                             conditions=conditions.reshape(-1, 9), sample_rate_hz=25, **coordinates)
                    worker.save_json(output/'preflight.json', dict(context, windows=5, start_second=0,
                        signal_scale_seconds=1., signal_center_seconds=0., scaler={},
                        inputs_sha256=worker.sha256(output/'inputs.npz')))
                    with (mock.patch.object(worker, 'frozen_imports', return_value=(modules, SimpleNamespace(window_size=window), {})),
                          mock.patch.object(worker, 'require_sampling_device', return_value=torch.device('cpu')),
                          mock.patch.object(torch, 'load', return_value={}),
                          mock.patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([42], dtype=torch.uint8))):
                        report = worker.worker_sample(context, output, 314, torch.get_num_threads(), batch_size=batch)
                    self.assertEqual(batch_ids, expected_ids)
                    self.assertEqual(report['sampling']['denoiser_calls'], 200*len(expected_ids))
                    self.assertTrue(report['sampling']['condition_columns_bitwise_preserved'])
                    actual_starts = starts if batch else np.arange(5)*window
                    rows = report['per_window_metrics']
                    self.assertEqual([r['start_sample'] for r in rows], actual_starts.tolist())
                    self.assertEqual([r['start_second'] for r in rows], (actual_starts/25).tolist())
                    self.assertEqual([r['observed_counts'] for r in rows], mask.sum(1).tolist())
                    with np.load(output/'waveforms.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['generated_ibi_seconds'], normalized.ravel())
                    ref_mean, gen_mean, ref_var, gen_var = [], [], [], []
                    for i, row in enumerate(rows):
                        x, y = real[i, mask[i]]*1000, normalized[i, mask[i]].astype(np.float64)*1000
                        ref_mean.append(x.mean())
                        gen_mean.append(y.mean())
                        ref_var.append(x.var())
                        gen_var.append(y.var())
                        self.assertAlmostEqual(row['mean_error_ms'], y.mean()-x.mean())
                        self.assertAlmostEqual(row['variance_error_ms2'], y.var()-x.var())
                    metrics = report['metrics']
                    for key, vals in dict(reference_mean_ms=ref_mean, generated_mean_ms=gen_mean,
                            reference_variance_ms2=ref_var, generated_variance_ms2=gen_var,
                            mean_error_ms=np.subtract(gen_mean, ref_mean),
                            variance_error_ms2=np.subtract(gen_var, ref_var),
                            mean_absolute_error_ms=np.abs(np.subtract(gen_mean, ref_mean)),
                            variance_absolute_error_ms2=np.abs(np.subtract(gen_var, ref_var))).items():
                        self.assertAlmostEqual(metrics[key], np.mean(vals))


if __name__ == '__main__':
    unittest.main()
