"""Plot completed S092 best-checkpoint comparisons; never run a model.

The full segment and deterministic 30-second details preserve every generated
value. Stored reference zeros are displayed as missing markers. Independently
generated model windows are separated explicitly, never visually stitched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ARMS = ('minmax', 'minmax01')
STAGES = ('W', 'Light', 'REM', 'Deep')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _scalar(data, name):
    value = data[name]
    if value.shape != () or value.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be a numeric scalar')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def load_result(result_dir):
    """Validate all required artifacts before creating the output directory."""
    root = Path(result_dir).expanduser().resolve()
    required = [root/'reference.npz']
    for arm in ARMS:
        required.extend((root/arm/'waveforms.npz', root/arm/'summary.json'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Complete reference and both generation outputs are required: ' + ', '.join(missing))
    source_hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in required}
    reference = _npz(root/'reference.npz')
    real = reference['reference_ibi_seconds']
    mask = reference['scoring_mask']
    stages = reference['sleep_stages']
    fs = _scalar(reference, 'sample_rate_hz')
    start = _scalar(reference, 'start_second')
    if fs != 25 or start < 0:
        raise ValueError('Expected a nonnegative NPZ start time and 25 Hz sampling')
    window_value = _scalar(reference, 'window_size') if 'window_size' in reference else 3000
    if window_value != int(window_value):
        raise ValueError('Model window_size must be an integer')
    window_size = int(window_value)
    if window_size < 750 or window_size % 750:
        raise ValueError('Model window_size must contain complete 750-sample epochs')
    if (real.ndim != 1 or real.size < window_size or real.size % window_size
            or real.dtype.kind != 'f' or not np.isfinite(real).all() or np.any(real < 0)):
        raise ValueError('Reference must contain finite nonnegative complete model windows')
    if mask.dtype != np.bool_ or mask.shape != real.shape or not np.array_equal(mask, real > 0):
        raise ValueError('scoring_mask must identify exactly the original positive reference positions')
    if stages.shape != (real.size // 750,) or stages.dtype.kind not in 'iu' or np.any((stages < 0) | (stages > 3)):
        raise ValueError('sleep_stages must have one label per 750 samples: W=0, Light=1, REM=2, Deep=3')
    if (mask.reshape(-1, 750).sum(axis=1) < 2).any():
        raise ValueError('Each reference epoch must have at least two positive observations')
    generated, summaries = {}, {}
    for arm in ARMS:
        waveform = _npz(root/arm/'waveforms.npz')
        values = waveform['generated_ibi_seconds']
        if values.shape != real.shape or values.dtype.kind != 'f' or not np.isfinite(values).all():
            raise ValueError(f'{arm} must contain finite floating seconds matching the reference')
        # Negative generated values are valid diagnostic evidence and remain visible.
        generated[arm] = values.astype(np.float64, copy=False)
        summary = json.loads((root/arm/'summary.json').read_text(encoding='utf-8-sig'))
        if not isinstance(summary, dict):
            raise ValueError(f'{arm} summary must be a JSON object')
        if 'status' in summary and summary['status'] != 'completed':
            raise ValueError(f'{arm} generation is not completed')
        if type(summary.get('epoch')) is not int or summary['epoch'] < 1:
            raise ValueError(f'{arm} summary needs a positive checkpoint epoch')
        if type(summary.get('seed')) is not int or summary['seed'] < 0:
            raise ValueError(f'{arm} summary needs a nonnegative integer sampling seed')
        if 'seed' in waveform and _scalar(waveform, 'seed') != summary['seed']:
            raise ValueError(f'{arm} waveform seed differs from its summary')
        if ('waveform_sha256' in summary
                and summary['waveform_sha256'] != source_hashes[f'{arm}/waveforms.npz']):
            raise ValueError(f'{arm} waveform SHA256 differs from its summary')
        digest = summary.get('checkpoint_sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
            raise ValueError(f'{arm} summary needs checkpoint_sha256')
        summaries[arm] = summary
    if summaries['minmax']['seed'] != summaries['minmax01']['seed']:
        raise ValueError('The paired figures require the same sampling seed in both arms')
    if source_hashes != {path.relative_to(root).as_posix(): _sha256(path) for path in required}:
        raise RuntimeError('Input artifacts changed while being read')
    return dict(root=root, reference=real.astype(np.float64, copy=False), mask=mask,
                stages=stages, fs=fs, start=start, generated=generated,
                summaries=summaries, source_sha256=source_hashes, window_size=window_size,
                windows=real.size // window_size, duration_seconds=real.size / fs)


def _zoom_bounds(stages):
    """Choose solely from input labels, never waveform amplitude or model quality."""
    transitions = np.flatnonzero(stages[1:] != stages[:-1]) + 1
    if len(transitions):
        center = float(transitions[0] * 30)
        start = max(0., min(len(stages)*30. - 30., center - 15.))
        return start, start + 30., '30 seconds centered on the first input-stage transition'
    return 0., 30., 'first 30 seconds; no input-stage transition in the window'


def _draw(data, interval, *, detail, synthetic_fixture, stage_name=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    begin, end = interval
    fs, start = data['fs'], data['start']
    relative = np.arange(len(data['reference']), dtype=np.float64) / fs
    selected = (relative >= begin) & (relative < end)
    time = start + relative[selected]
    mask = data['mask'][selected]
    reference_ms = data['reference'][selected] * 1000.
    observed = np.where(mask, reference_ms, np.nan)
    fig, (axis, stage_axis) = plt.subplots(2, 1, figsize=(14.2, 7.3), sharex=True,
        gridspec_kw=dict(height_ratios=[4.2, 1.15], hspace=.12))
    fig.subplots_adjust(left=.075, right=.965, bottom=.17, top=.80)
    prefix = 'SYNTHETIC FIXTURE — ' if synthetic_fixture else ''
    title = 'S092 | Best-model IBI comparison'
    if detail:
        title += ' | ' + (stage_name + ' | ' if stage_name else '') + '30 s detail'
    fig.suptitle(prefix + title, x=.075, ha='left', y=.97, fontsize=17, fontweight='bold')
    summaries = data['summaries']
    subtitle = (f"NPZ time {start+begin:.0f}–{start+end:.0f} s  |  same sampling seed {summaries['minmax']['seed']}  |  "
                f"[-1, 1]: epoch {summaries['minmax']['epoch']}    [0, 1]: epoch {summaries['minmax01']['epoch']}")
    fig.text(.075, .92, subtitle, fontsize=11)
    axis.plot(time, observed, color='#222222', linewidth=1.25, alpha=.9,
              label='Real IBI (positive observations)', zorder=3)
    for arm, color, label in (('minmax', '#1672B8', 'Generated: minmax [-1, 1]'),
                              ('minmax01', '#D8691A', 'Generated: minmax [0, 1]')):
        blocks = (np.flatnonzero(selected) // data['window_size'])
        for number, block in enumerate(np.unique(blocks)):
            local = blocks == block
            axis.plot(time[local], data['generated'][arm][selected][local] * 1000., color=color,
                      linewidth=1.05, alpha=.88, label=label if number == 0 else None, zorder=2)
    axis.scatter(time[~mask], np.zeros(int((~mask).sum())), s=15, marker='x',
                 color='#777777', linewidths=.9, label='Stored zero: missing marker', zorder=5)
    if not mask.all() or any(np.any(values[selected] <= 0) for values in data['generated'].values()):
        axis.axhline(0, color='#999999', linewidth=.7, linestyle=':', zorder=1)
    axis.set_ylabel('IBI (ms)', fontsize=11)
    axis.grid(axis='both', color='#DEDEDE', linewidth=.55, alpha=.7)
    axis.margins(y=.10)
    axis.legend(loc='lower left', bbox_to_anchor=(0., 1.01), ncol=2,
                fontsize=10, frameon=False, borderaxespad=0., columnspacing=2.5)
    # Clinical display order differs from the encoded class IDs; labels are explicit.
    display_y = np.array([3, 1, 2, 0])[data['stages']]
    boundaries = start + np.arange(len(data['stages'])+1, dtype=np.float64) * 30.
    stage_axis.step(boundaries, np.r_[display_y, display_y[-1]], where='post',
                    linewidth=2.0, color='#5B4A8A')
    shades = ('#F2F5F9', '#E9F2FA', '#F1EAF7', '#E3EBF5')
    for index, stage in enumerate(data['stages']):
        stage_axis.axvspan(boundaries[index], boundaries[index+1], color=shades[int(stage)], zorder=0)
        if begin < index*30 < end:
            axis.axvline(boundaries[index], color='#BBBBBB', linewidth=.7, linestyle='--', zorder=0)
    for boundary in np.arange(1, data['windows']) * data['window_size'] / fs:
        if begin < boundary < end:
            for ax in (axis, stage_axis):
                ax.axvline(start+boundary, color='#8946A8', linewidth=1.4, linestyle=':', zorder=4)
    stage_axis.set_yticks([0, 1, 2, 3], ['Deep', 'Light', 'REM', 'W'])
    stage_axis.set_ylim(-.35, 3.35)
    stage_axis.set_ylabel('Input stage', fontsize=11)
    stage_axis.set_xlabel('Time from start of the stored NPZ record (mm:ss)', fontsize=11)
    stage_axis.set_xlim(start+begin, start+end)
    stage_axis.xaxis.set_major_locator(MultipleLocator(5 if detail else (30 if end-begin > 120 else 15)))
    stage_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{int(round(value))//60:02d}:{int(round(value))%60:02d}'))
    stage_axis.grid(axis='x', color='#CCCCCC', linewidth=.55, alpha=.65)
    for ax in (axis, stage_axis):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.text(.075, .078, f"{data['windows']} independent {data['window_size']/fs:g} s window(s); purple lines mark window boundaries. "
             'Sleep stages are input conditions, not predictions.', fontsize=10, color='#333333')
    fig.text(.075, .044, 'All samples retained; no clipping or smoothing. Reference zeros mark missing observations; '
             'generated negative values remain visible.', fontsize=10, color='#555555')
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true',
                        help='Label every figure as synthetic fixture QA, not real inference')
    cli = parser.parse_args(argv)
    data = load_result(cli.result_dir)
    plot_dir = data['root']/'plots'
    # Refuse even a partial existing plotting directory: never overwrite artifacts.
    plot_dir.mkdir(exist_ok=False)
    zoom_start, zoom_end, zoom_rule = _zoom_bounds(data['stages'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    products = []
    views = [('S092_best_minmax_comparison', (0., data['duration_seconds']), False, None),
             ('S092_best_minmax_zoom30s', (zoom_start, zoom_end), True, None)]
    stage_details = {}
    for stage, name in enumerate(STAGES):
        matches = np.flatnonzero(data['stages'] == stage)
        if len(matches):
            begin = float(matches[0] * 30)
            views.append((f'S092_best_minmax_stage_{name.lower()}', (begin, begin+30), True, name))
            stage_details[name] = dict(relative_start_seconds=begin, relative_end_seconds=begin+30,
                                      selection_rule='first original epoch of this stage in the selected segment')
    for stem, interval, detail, stage_name in views:
        fig = _draw(data, interval, detail=detail, synthetic_fixture=cli.synthetic_fixture, stage_name=stage_name)
        try:
            for extension in ('png', 'pdf'):
                path = plot_dir/(stem+'.'+extension)
                with path.open('xb') as stream:
                    fig.savefig(stream, format=extension, dpi=180, facecolor='white')
                products.append(dict(path=str(path), sha256=_sha256(path)))
        finally:
            plt.close(fig)
    manifest = dict(status='completed', synthetic_fixture=cli.synthetic_fixture,
        model_executed_by_plotter=False, source_sha256=data['source_sha256'],
        source_window=dict(start_second=data['start'], duration_seconds=data['duration_seconds'], sample_rate_hz=data['fs'],
                           samples=len(data['reference']), model_window_size=data['window_size'], windows=data['windows'],
                           sleep_stages=data['stages'].tolist(), stage_names=list(STAGES)),
        checkpoints={arm:{key:data['summaries'][arm][key] for key in ('epoch','checkpoint_sha256','seed')} for arm in ARMS},
        zoom=dict(relative_start_seconds=zoom_start, relative_end_seconds=zoom_end, selection_rule=zoom_rule),
        stage_details=stage_details,
        reference_missing_samples=int((~data['mask']).sum()),
        generated_nonpositive_samples={arm:int((value<=0).sum()) for arm,value in data['generated'].items()},
        plotting=dict(clipped=False, smoothed=False, decimated=False, reference_zeros_shown_as_missing_markers=True,
                      all_generated_values_retained=True), artifacts=products)
    with (plot_dir/'plot_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='completed', plots=str(plot_dir), synthetic_fixture=cli.synthetic_fixture)))
    return manifest


if __name__ == '__main__':
    main()
