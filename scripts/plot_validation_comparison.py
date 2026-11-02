"""Plot exported independent validation windows without running a model.

Both inputs must describe exactly the same reference windows and scoring mask.
Plotting and file I/O use the host; this script performs no CPU inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_artifacts(directory):
    directory = Path(directory)
    recorded = json.loads((directory / 'artifacts.json').read_text(encoding='utf-8'))
    for filename, field in (('waveforms.npz', 'waveform_sha256'),
                            ('per_window_metrics.csv', 'metrics_sha256')):
        digest = hashlib.sha256()
        with (directory / filename).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if recorded.get(field) != digest.hexdigest():
            raise ValueError('Artifact hash does not match artifacts.json: ' + filename)
    with np.load(directory / 'waveforms.npz', allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    with (directory / 'per_window_metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(arrays['validation_index']):
        raise ValueError('Statistics and waveform counts differ')
    if [int(r['validation_index']) for r in rows] != arrays['validation_index'].tolist():
        raise ValueError('Statistics and waveform order differ')
    for field, convert in (('subject_id', str), ('filename', str),
                           ('start_sample', int), ('start_second', float)):
        if [convert(row[field]) for row in rows] != arrays[field].tolist():
            raise ValueError('Statistics and waveform identity differ: ' + field)
    reference = arrays['reference_ibi_seconds']
    if reference.ndim != 2 or min(reference.shape) < 1:
        raise ValueError('Expected nonempty [windows, samples] reference waveforms')
    for field in ('generated_ibi_seconds', 'scoring_mask', 'reference_observed_mask'):
        if arrays[field].shape != reference.shape:
            raise ValueError('Artifact shape does not match reference waveforms: ' + field)
    for field in ('scoring_mask', 'reference_observed_mask'):
        if arrays[field].dtype != np.bool_:
            raise ValueError('Artifact mask must be boolean: ' + field)
    observed = np.isfinite(reference) & (reference > 0)
    if not np.array_equal(arrays['reference_observed_mask'], observed):
        raise ValueError('Stored observed mask does not match finite positive reference positions')
    if not np.array_equal(arrays['scoring_mask'], observed):
        raise ValueError('This A/B comparison requires observed_only scoring in both artifacts')
    if (observed.sum(axis=1) < 2).any():
        raise ValueError('Every comparison window needs at least two observed reference samples')
    return arrays, rows


def verify_pair(a, b):
    keys = ('reference_ibi_seconds', 'scoring_mask', 'reference_observed_mask',
            'validation_index', 'subject_id', 'filename', 'start_sample', 'start_second',
            'sleep_stages', 'condition_columns', 'clean_conditions')
    for key in keys:
        same = (np.array_equal(a[key], b[key], equal_nan=True)
                if np.issubdtype(a[key].dtype, np.floating) else np.array_equal(a[key], b[key]))
        if not same:
            raise ValueError('Cannot compare different validation inputs: ' + key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preserve-dir', required=True, type=Path)
    parser.add_argument('--masked-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--indices', nargs='+', type=int, help='Fixed validation indices; default: up to 3 evenly spaced indices')
    args = parser.parse_args(argv)
    a, ar = read_artifacts(args.preserve_dir)
    b, br = read_artifacts(args.masked_dir)
    verify_pair(a, b)
    n, length = a['reference_ibi_seconds'].shape
    indices = args.indices if args.indices is not None else np.unique(np.linspace(0, n-1, min(3, n), dtype=int)).tolist()
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= n for i in indices):
        raise ValueError('indices must be distinct valid validation indices')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#2866a6', '#dc7431']
    # Sampling rate is fixed by the DREAMT model interface (25 Hz).
    seconds = np.arange(length) / 25.
    plots = []
    for i in indices:
        fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True,
                                  gridspec_kw={'height_ratios': [3.5, 1]})
        real = a['reference_ibi_seconds'][i] * 1000
        observed = a['reference_observed_mask'][i]
        axes[0].plot(seconds, np.where(observed, real, np.nan), color='#222222', lw=1.2, label='Real observed IBI', zorder=4)
        missing = ~observed
        axes[0].scatter(seconds[missing], real[missing], marker='x', color='#969696', s=9, label='Stored missing values', zorder=5)
        for arrays, label, color in ((a, 'Preserve zeros', colors[0]), (b, 'Masked missing values', colors[1])):
            axes[0].plot(seconds, arrays['generated_ibi_seconds'][i] * 1000, color=color, lw=1, alpha=.85, label=label)
        axes[0].axhline(0, color='#aaaaaa', lw=.6)
        axes[0].set_ylabel('IBI (ms)')
        axes[0].legend(loc='best', ncol=2, fontsize=9, frameon=False)
        start = a['start_second'][i]
        title = f"Validation window {i} | {a['subject_id'][i]} | NPZ offset {start:.0f}-{start + length/25:.0f} s"
        axes[0].set_title(title, loc='left', weight='bold', pad=12)
        stages = a['sleep_stages'][i]
        axes[1].step(np.arange(len(stages)+1)*30, np.r_[stages, stages[-1]], where='post', color='#553d78', lw=2)
        axes[1].set_yticks([0, 1, 2, 3], ['W', 'Light', 'REM', 'Deep'])
        axes[1].set_ylim(-.35, 3.35)
        axes[1].set_ylabel('Stage')
        axes[1].set_xlabel('Time within window (s)')
        axes[1].set_xlim(0, length/25)
        for ax in axes:
            ax.grid(alpha=.15)
        text = (f"Window KL: preserve={float(ar[i]['gaussian_kl_real_to_generated']):.4g}; masked={float(br[i]['gaussian_kl_real_to_generated']):.4g}. "
                'Both use the same observed reference positions.\n'
                'Independent conditional generation: curves need not match point by point. Stage is the shared input condition, not a generated prediction.')
        fig.text(.075, .018, text, fontsize=9, color='#555555')
        fig.subplots_adjust(left=.075, right=.985, top=.90, bottom=.20, hspace=.16)
        path = args.output_dir / f'window_{i:03d}.png'
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(str(path.resolve()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title, unit in zip(axes[:2], ['mean', 'variance'], ['Window means', 'Window variances'], ['ms', 'ms^2']):
        all_values = []
        for rows, label, color in ((ar, 'Preserve', colors[0]), (br, 'Masked', colors[1])):
            x = np.array([float(r[f'reference_{key}_{"ms" if key == "mean" else "ms2"}']) for r in rows])
            y = np.array([float(r[f'generated_{key}_{"ms" if key == "mean" else "ms2"}']) for r in rows])
            ax.scatter(x, y, s=20, alpha=.65, color=color, label=label)
            all_values.extend([x, y])
        lo, hi = min(float(v.min()) for v in all_values), max(float(v.max()) for v in all_values)
        if hi == lo:
            lo, hi = lo-1, hi+1
        ax.plot([lo, hi], [lo, hi], '--', color='#777777', lw=1)
        ax.set_xlabel(f'Real ({unit})')
        ax.set_ylabel(f'Generated ({unit})')
        ax.set_title(title, weight='bold')
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    means = [np.mean([float(r['kl_mean_term']) for r in rows]) for rows in (ar, br)]
    variances = [np.mean([float(r['kl_variance_term']) for r in rows]) for rows in (ar, br)]
    axes[2].bar(['Preserve', 'Masked'], means, label='Mean mismatch term', color='#75879b')
    axes[2].bar(['Preserve', 'Masked'], variances, bottom=means, label='Variance mismatch term', color='#dab27e')
    axes[2].set_title('KL decomposition', weight='bold')
    axes[2].set_ylabel('Mean window KL (nats)')
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Same {n} validation windows | raw population moments at observed reference positions', fontsize=13)
    fig.tight_layout(rect=[0, .02, 1, .93])
    path = args.output_dir / 'window_statistics.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots.append(str(path.resolve()))
    report = dict(status='completed', windows=n, selected_indices=indices, plots=plots,
                  pair_inputs_identical=True, model_executed=False,
                  limitation='Window Gaussian moments do not assess temporal fidelity. No smoothing or clipping applied.')
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
