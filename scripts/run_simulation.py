"""Observe baseline optimizer updates and record sample-weighted mean loss.

The training entry point explicitly runs epoch validation with isolated RNG and
an independent evaluation model. This trace only reads completed update metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda_runtime import require_cuda, cuda_environment
from epoch_validation import weighted_loss


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--dataset', choices=('DREAMT', 'DREAMT_SIM'), default='DREAMT_SIM')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/simulated_64x60m')
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.log_every) < 1:
        parser.error('Positive batch size, epoch count and log interval required')
    device = require_cuda()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = ROOT / 'saved_models' / args.dataset / 'model.pth'
    if checkpoint_path.exists():
        raise FileExistsError('The original script would overwrite its checkpoint; preserve this run first')
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_upstream_boundary.py'),
                    '--output', str(output / 'source_boundary_before.json')], cwd=ROOT, check=True)
    import torch
    import numpy as np
    training = ROOT / 'training_wavestitch.py'
    filename = str(training)
    lines = training.read_text(encoding='utf-8').splitlines()
    step_line = next(i for i, line in enumerate(lines, 1) if line.strip() == 'optimizer.step()')
    loop_line = next(i for i, line in enumerate(lines, 1) if line.strip() == 'for epoch in range(args.epochs):')
    source_paths = [training, ROOT / 'data_utils.py', ROOT / 'training_utils.py',
                    ROOT / 'TSImputers/SSSDS4Imputer.py', ROOT / 'TSImputers/S4Model.py', ROOT / 'utils/util.py',
                    ROOT / 'watchsleep_loader.py', ROOT / 'dataset_config.json', ROOT / 'cuda_runtime.py',
                    ROOT / 'third_party/WatchSleepNet/data_setup.py', ROOT / 'epoch_validation.py',
                    ROOT / 'validation_metrics.py', ROOT / 'validation_sampling.py',
                    ROOT / 'scripts/evaluate_original_sampler.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    command = [filename, '-d', args.dataset, '-window_size', '3000', '-s4_lmax', '3000',
               '-stride', '1500', '-batch_size', str(args.batch_size), '-epochs', str(args.epochs),
               '-validation_output', str(output / 'validation')]
    manifest = dict(status='starting', python=sys.executable, python_version=sys.version,
                    torch=torch.__version__, numpy=np.__version__, cuda=torch.cuda.is_available(),
                    mkldnn_enabled=torch.backends.mkldnn.enabled, threads=torch.get_num_threads(),
                    process_id=os.getpid(), command=command, source_sha256=hashes,
                    device=str(device), cuda_environment=cuda_environment(device),
                    observer='sys.settrace reads current loss and completed updates only; no training hooks or replacements',
                    observer_sha256=sha(Path(__file__)), training_args=None,
                    checkpoint=str(checkpoint_path), data_synthetic=None)
    save_json(output / 'run_manifest.json', manifest)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    pending = None
    steps = 0
    rows = []
    configured = False
    old_trace, old_argv, old_cwd = sys.gettrace(), sys.argv, Path.cwd()
    old_path = sys.path[:]
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'epoch', 'batch_samples', 'epsilon_mse', 'elapsed_seconds'])
        writer.writeheader()

        def local_trace(frame, event, arg):
            nonlocal pending, steps, configured
            if event == 'line':
                if pending is not None:
                    steps += 1
                    row = dict(step=steps, **pending, elapsed_seconds=round(time.perf_counter()-started, 3))
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    pending = None
                    progress = dict(status='training', **row, expected_updates=manifest['expected_updates'])
                    save_json(output / 'progress.json', progress)
                    if steps == 1 or steps % args.log_every == 0:
                        print(json.dumps(progress), flush=True)
                if frame.f_lineno == loop_line and not configured:
                    state = frame.f_locals
                    manifest.update(status='training', training_args=vars(state['args']).copy(),
                                    parameter_count=sum(p.numel() for p in state['model'].parameters()),
                                    train_windows=len(state['training_dataset']),
                                    batches_per_epoch=len(state['dataloader']),
                                    expected_updates=len(state['dataloader']) * args.epochs,
                                    data=state['preprocessor'].provenance(),
                                    data_identity=state['preprocessor'].data_identity(),
                                    model_input_columns=state['preprocessor'].columns.tolist())
                    manifest['data_synthetic'] = manifest['data']['data_synthetic']
                    save_json(output / 'run_manifest.json', manifest)
                    print(json.dumps({k:manifest[k] for k in ['status','parameter_count','train_windows','expected_updates']}), flush=True)
                    configured = True
                if frame.f_lineno == step_line:
                    state = frame.f_locals
                    value = float(state['loss'].detach())
                    if not math.isfinite(value):
                        raise FloatingPointError('Observed nonfinite loss; stopped without changing the algorithm')
                    pending = dict(epoch=int(state['epoch'])+1, batch_samples=int(state['batch'].shape[0]),
                                   epsilon_mse=value)
            return local_trace

        def trace(frame, event, arg):
            return local_trace if frame.f_code.co_filename == filename else None

        try:
            os.chdir(ROOT)
            sys.path.insert(0, str(ROOT))
            sys.argv = command
            sys.settrace(trace)
            training_state = runpy.run_path(filename, run_name='__main__')
        except BaseException:
            save_json(output / 'failure.json', dict(status='failed', completed_updates=steps,
                                                   traceback=traceback.format_exc()))
            raise
        finally:
            sys.settrace(old_trace)
            sys.argv = old_argv
            sys.path[:] = old_path
            os.chdir(old_cwd)
    assert pending is None and steps == manifest['expected_updates']
    validation = training_state['validator'].summary()
    del training_state  # Release the training model/optimizer before loading the checkpoint audit.
    assert validation['completed_epochs'] == args.epochs
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    torch.cuda.synchronize(device)
    state = torch.load(checkpoint_path, map_location=device)
    assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
    losses = [r['epsilon_mse'] for r in rows]
    report = dict(status='completed', device=str(device), cuda_environment=cuda_environment(device),
                  updates=steps, epochs=args.epochs, batch_size=args.batch_size,
                  train_windows=manifest['train_windows'], parameter_count=manifest['parameter_count'],
                  wall_seconds=time.perf_counter()-started, first_loss=losses[0], last_loss=losses[-1],
                  first_25_mean=weighted_loss(rows[:25]), last_25_mean=weighted_loss(rows[-25:]),
                  mean_loss=weighted_loss(rows), loss_aggregation='sample_weighted_mean',
                  validation=validation, all_losses_finite=True, checkpoint_all_finite=True,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                  source_unchanged=True, mkldnn_enabled=torch.backends.mkldnn.enabled,
                  synthetic_data=manifest['data_synthetic'], no_quality_or_convergence_claim=True)
    save_json(output / 'training_report.json', report)
    save_json(output / 'progress.json', dict(status='completed', updates=steps))
    manifest['validation'] = validation
    manifest['status'] = 'completed'
    manifest['checkpoint_sha256'] = report['checkpoint_sha256']
    save_json(output / 'run_manifest.json', manifest)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
