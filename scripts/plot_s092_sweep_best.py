"""Plot selected completed, paired S092 best-checkpoint outputs without a model.

Four-stage details select the first original 30-second epoch of each stage.
Full-window views retain every generated value and separate independent draws.
All displayed IBI values are seconds; zeros and negative outputs are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

STAGES = ('W', 'Light', 'REM', 'Deep')
LEVELS = np.array([3, 1, 2, 0])
STAGE_COLORS = ('#f3d892', '#a3c8df', '#b7a1dc', '#84b6ab')
EPOCH_SAMPLES = 750


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return result


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def scalar(arrays, key):
    value = arrays[key]
    if value.shape != () or value.dtype.kind not in 'iuf' or not np.isfinite(value):
        raise ValueError(key + ' must be a finite numerical scalar')
    return float(value)


def configuration_rows(rows):
    """Validate a nonempty subset of the fixed twelve-job protocol."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require at least one selected configuration')
    if any(not isinstance(row, dict) or type(row.get('id')) is not int for row in rows):
        raise ValueError('Configuration IDs must be integers')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids).issubset(range(12)):
        raise ValueError('Require unique configuration IDs within 0 through 11')
    rows = sorted(rows, key=lambda row: row['id'])
    for row in rows:
        index = row['id']
        output_range = [-1., 1.] if index < 6 else [0., 1.]
        location = 'input' if index % 6 < 3 else 's4'
        probability = (0., .03, .05)[index % 3]
        expected = dict(normalization_range=output_range, dropout_location=location,
                        input_probability=probability if location == 'input' else 0.,
                        s4_probability=probability if location == 's4' else 0.)
        if not isinstance(row.get('name'), str) or not row['name']:
            raise ValueError('Configuration must have a name')
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('Configuration differs from the twelve-job grid: ' + str(index))
    return rows


def paired_statistics(reference, generated, mask):
    """Match training's within-window Gaussian KL(real||generated), floor 1 ms^2."""
    real, synth = reference[mask].astype(np.float64)*1000, generated[mask].astype(np.float64)*1000
    if real.size < 2:
        raise ValueError('Need at least two reference-observed values per scored interval')
    rm, gm, rv, gv = map(float, (real.mean(), synth.mean(), real.var(), synth.var()))
    vr, vg = max(rv, 1.), max(gv, 1.)
    change = (vr-vg)/vg
    if abs(change) <= 1e-4:
        variance_term = change*change*(.5+change*(-1/3+change*(.25-change/5)))
    else:
        variance_term = max(0., vr/vg-1.-math.log(vr/vg))
    kl = .5*(variance_term+(rm-gm)**2/vg)
    result = dict(gaussian_kl_real_to_generated=kl,
                  reference_mean_ms=rm, generated_mean_ms=gm,
                  mean_absolute_error_ms=abs(gm-rm),
                  reference_variance_ms2=rv, generated_variance_ms2=gv,
                  variance_absolute_error_ms2=abs(gv-rv), observed_samples=int(real.size))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite recomputed statistics')
    return result


def load_result(result_dir):
    """Validate all paired artifacts and hashes before creating output files."""
    root = Path(result_dir).expanduser().resolve()
    paths = [root/'summary.json', root/'reference.npz']
    summary = read_json(paths[0])
    if summary.get('status') != 'completed':
        raise ValueError('Only completed generation comparisons can be plotted')
    configs = configuration_rows(summary.get('configurations'))
    selected_ids = {row['id'] for row in configs}
    for key, expected in (('selected_configuration_ids', selected_ids),
                          ('excluded_configuration_ids', set(range(12))-selected_ids),
                          ('completed_configurations', selected_ids)):
        if key in summary:
            ids = summary[key]
            if (not isinstance(ids, list) or any(type(index) is not int for index in ids)
                    or len(ids) != len(set(ids)) or set(ids) != expected):
                raise ValueError(key + ' differs from the selected completed configurations')
    seed = summary.get('seed')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Require a shared unsigned integer sampling seed')
    for row in configs:
        paths.extend(root/f'config_{row["id"]}'/name for name in ('summary.json', 'inputs.npz', 'waveforms.npz'))
    hashes = {path.relative_to(root).as_posix(): sha256(path) for path in paths}
    if summary.get('reference_sha256') != hashes['reference.npz']:
        raise ValueError('Root reference SHA256 differs from the coordinator record')
    reference = read_npz(root/'reference.npz')
    real, mask, stages = (reference[key] for key in ('reference_ibi_seconds', 'scoring_mask', 'sleep_stages'))
    fs, start, window = (scalar(reference, key) for key in ('sample_rate_hz', 'start_second', 'window_size'))
    if fs != 25 or start < 0 or start % 30 or window < 750 or window != int(window) or window % 750:
        raise ValueError('Require 25 Hz, 30-second aligned time, and complete epoch windows')
    window = int(window)
    if (real.ndim != 1 or real.dtype.kind != 'f' or real.size < window or real.size % window
            or not np.isfinite(real).all() or np.any(real < 0)):
        raise ValueError('Reference must contain complete finite nonnegative model windows in seconds')
    if mask.dtype != np.bool_ or mask.shape != real.shape or not np.array_equal(mask, real > 0):
        raise ValueError('scoring_mask must identify exactly the positive reference values')
    if ((mask.reshape(-1, EPOCH_SAMPLES).sum(axis=1) < 2).any()
            or stages.shape != (real.size//EPOCH_SAMPLES,) or stages.dtype.kind not in 'iu'
            or set(stages.tolist()) != set(range(4))):
        raise ValueError('Require all four stages and observed values in every epoch')
    conditions = reference['conditions']
    if (conditions.shape != (real.size, 9) or conditions.dtype.kind != 'f'
            or not np.isfinite(conditions).all()
            or not np.array_equal(conditions[:, :4], np.eye(4)[np.repeat(stages, EPOCH_SAMPLES)])):
        raise ValueError('Reference condition channels must match the original stage labels')
    generated, reports, rows = {}, {}, []
    stage_rows, window_rows = [], []
    stage_starts = {name: int(np.flatnonzero(stages == index)[0])*EPOCH_SAMPLES
                    for index, name in enumerate(STAGES)}
    for config in configs:
        index = config['id']
        prefix = f'config_{index}'
        report = read_json(root/prefix/'summary.json')
        if (report.get('status') != 'completed' or report.get('configuration') != config
                or type(report.get('epoch')) is not int or report['epoch'] < 1
                or report.get('seed') != seed):
            raise ValueError(prefix + ': mismatched configuration, seed or checkpoint epoch')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', str(report.get('checkpoint_sha256', ''))):
            raise ValueError(prefix + ': missing checkpoint identity')
        if report.get('waveform_sha256') != hashes[prefix+'/waveforms.npz']:
            raise ValueError(prefix + ': waveform SHA256 differs')
        inputs = read_npz(root/prefix/'inputs.npz')
        if set(inputs) != set(reference) or any(not np.array_equal(value, inputs[key]) for key, value in reference.items()):
            raise ValueError(prefix + ': reference/conditions/mask differ from the paired inputs')
        arrays = read_npz(root/prefix/'waveforms.npz')
        values = arrays['generated_ibi_seconds']
        normalized = arrays['generated_normalized']
        if (values.dtype.kind != 'f' or values.shape != real.shape or not np.isfinite(values).all()
                or normalized.dtype.kind != 'f' or normalized.shape != real.shape or not np.isfinite(normalized).all()
                or scalar(arrays, 'seed') != seed or scalar(arrays, 'window_size') != window):
            raise ValueError(prefix + ': invalid waveform shape, finiteness, seed or window size')
        center, scale = (1., 1.) if config['normalization_range'][0] == -1. else (0., 2.)
        if not np.array_equal(normalized.astype(np.float64)*scale+center, values):
            raise ValueError(prefix + ': inverse normalization does not recover saved seconds')
        sampling = report.get('sampling', {})
        if (sampling.get('independent_windows') is not True
                or sampling.get('signal_all_unknown') is not True
                or sampling.get('condition_columns_bitwise_preserved') is not True
                or sampling.get('clipping_applied') is not False):
            raise ValueError(prefix + ': expected unmodified conditions and independent unknown-signal sampling')
        generated[index], reports[index] = values.astype(np.float64), report
        per_window = []
        for offset in range(0, real.size, window):
            section = slice(offset, offset+window)
            metrics = paired_statistics(real[section], values[section], mask[section])
            per_window.append(metrics)
            window_rows.append(dict(configuration_id=index, window_index=offset//window,
                                    start_second=start+offset/fs, **metrics))
        aggregate = {key: float(np.mean([item[key] for item in per_window])) for key in
                     ('gaussian_kl_real_to_generated', 'mean_absolute_error_ms', 'variance_absolute_error_ms2')}
        for key, value in aggregate.items():
            if not np.isclose(report.get('metrics', {}).get(key, np.nan), value, rtol=1e-8, atol=1e-8):
                raise ValueError(prefix + ': saved metrics differ from recomputed ' + key)
        rows.append(dict(configuration_id=index, configuration_name=config['name'], checkpoint_epoch=report['epoch'],
                         normalization_range=str(config['normalization_range']), dropout_location=config['dropout_location'],
                         input_probability=config['input_probability'], s4_probability=config['s4_probability'],
                         **aggregate, generated_nonpositive_fraction=float((values <= 0).mean())))
        for name, offset in stage_starts.items():
            section = slice(offset, offset+EPOCH_SAMPLES)
            stage_rows.append(dict(configuration_id=index, stage=name, start_second=start+offset/fs,
                                   duration_seconds=30., **paired_statistics(real[section], values[section], mask[section])))
    if hashes != {path.relative_to(root).as_posix(): sha256(path) for path in paths}:
        raise RuntimeError('Input artifacts changed while being read')
    extrema = [float(real.min()), float(real.max()), 0.]
    for values in generated.values():
        extrema.extend((float(values.min()), float(values.max())))
    low, high = min(extrema), max(extrema)
    margin = max(.05, (high-low)*.05)
    return dict(root=root, summary=summary, configs=configs, real=real.astype(np.float64), mask=mask,
                stages=stages, generated=generated, reports=reports, fs=fs, start=start, window_size=window,
                seed=seed, stage_starts=stage_starts, ylim=(low-margin, high+margin), source_sha256=hashes,
                rows=rows, stage_rows=stage_rows, window_rows=window_rows)


def _clock(seconds):
    value = int(seconds)
    return f'{value//3600:02d}:{value//60%60:02d}:{value%60:02d}'


def _panel(top, bottom, data, config, offset, samples, label_y):
    section = slice(offset, offset+samples)
    time = np.arange(samples)/data['fs']
    mask, real = data['mask'][section], data['real'][section]
    generated = data['generated'][config['id']][section]
    color = '#2b6fbd' if config['normalization_range'][0] == -1 else '#e47632'
    top.plot(time, np.where(mask, real, np.nan), color='#20262b', lw=1.2, label='Real IBI (observed)', zorder=3)
    top.plot(time, generated, color=color, lw=1.05, label='Generated IBI', zorder=4)
    top.scatter(time[~mask], real[~mask], color='#858c93', marker='x', s=10, linewidths=.65,
                label='Stored missing value (0)', zorder=5)
    top.axhline(0., color='#a0a6ab', lw=.7)
    top.set_ylim(*data['ylim'])
    top.grid(axis='y', color='#b2b7bc', alpha=.23, lw=.7)
    top.tick_params(labelbottom=False)
    duration = samples/data['fs']
    for boundary in np.arange(30., duration, 30.):
        top.axvline(boundary, color='#b2b7bc', lw=.7, ls='--', alpha=.55)
    stages = data['stages'][offset//750:(offset+samples)//750]
    edges = np.arange(len(stages)+1)*30.
    for index, stage in enumerate(stages):
        bottom.axvspan(edges[index], edges[index+1], color=STAGE_COLORS[stage], alpha=.38, lw=0)
        bottom.text((edges[index]+edges[index+1])/2, 3.95, STAGES[stage], ha='center', va='center', fontsize=10)
    bottom.stairs(LEVELS[stages], edges, baseline=None, color='#334155', lw=1.9)
    bottom.set_ylim(-.5, 4.45)
    bottom.set_yticks([0, 1, 2, 3], ['Deep', 'Light', 'REM', 'W'])
    bottom.set_xticks(np.arange(0, duration+.001, 5 if duration == 30 else 30))
    bottom.set_xlabel('Time within displayed segment (s)', labelpad=6)
    bottom.grid(axis='x', alpha=.14)
    for axis in (top, bottom):
        axis.set_xlim(0, duration)
    if label_y:
        top.set_ylabel('IBI (s)')
        bottom.set_ylabel('Sleep stage\n(input label)')
    else:
        top.tick_params(labelleft=False)
        bottom.tick_params(labelleft=False)


def _figure(data, config, *, details, synthetic_fixture):
    import matplotlib.pyplot as plt
    if details:
        intervals = [(name+' | 30 s detail', offset, 750) for name, offset in data['stage_starts'].items()]
    else:
        intervals = [(f'Window {i+1} | independent', offset, data['window_size'])
                     for i, offset in enumerate(range(0, len(data['real']), data['window_size']))]
    fig = plt.figure(figsize=(max(14, 5.5*len(intervals)), 8))
    grid = fig.add_gridspec(2, len(intervals), height_ratios=[3.7, 1.2], hspace=.13, wspace=.18)
    first = None
    for column, (name, offset, samples) in enumerate(intervals):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column], sharex=top)
        _panel(top, bottom, data, config, offset, samples, label_y=column == 0)
        start = data['start']+offset/data['fs']
        top.set_title(f'{name}\nNPZ {_clock(start)} - {_clock(start+samples/data["fs"])}', fontsize=11.5, pad=13)
        first = first or top
    prefix = 'SYNTHETIC FIXTURE | ' if synthetic_fixture else ''
    fig.suptitle(f'{prefix}S092 | config_{config["id"]} | Real and generated IBI', fontsize=19, y=.98)
    fig.text(.5, .925, f'Best epoch {data["reports"][config["id"]]["epoch"]} | Range {config["normalization_range"]} | '
             f'Input dropout {config["input_probability"]:g}; S4 dropout {config["s4_probability"]:g} | '
             f'Paired seed {data["seed"]}', ha='center', fontsize=12, color='#475569')
    fig.legend(*first.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .895),
               ncol=3, frameon=False, fontsize=11)
    line = ('First original 30-second epoch of each stage; selection does not use generated quality.' if details else
            'Complete saved segment; model windows are generated independently, without overlap or continuation.')
    fig.text(.055, .045, line+'\nAll values retained without smoothing or clipping. Gray crosses: missing zeros. '
             'Stages are input labels, not predictions.', fontsize=10.5, color='#475569', linespacing=1.55)
    fig.subplots_adjust(left=.055, right=.985, top=.775, bottom=.19)
    return fig


def _save_table(output, stem, rows):
    (output/(stem+'.json')).write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/(stem+'.csv')).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_comparison(result_dir, output_dir=None, synthetic_fixture=False):
    data = load_result(result_dir)
    output = Path(output_dir).expanduser().resolve() if output_dir else data['root']/'plots'
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'axes.titleweight': 'bold',
                         'svg.fonttype': 'none', 'path.simplify': False}):
        for config in data['configs']:
            for details in (True, False):
                fig = _figure(data, config, details=details, synthetic_fixture=synthetic_fixture)
                stem = f'config_{config["id"]}_'+('four_stages_seconds' if details else 'all_windows_seconds')
                try:
                    for extension in (('png', 'svg') if details else ('png',)):
                        path = output/(stem+'.'+extension)
                        fig.savefig(path, dpi=180)
                        files.append(path)
                finally:
                    plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        positions = np.arange(len(data['configs']))
        colors = ['#2b6fbd' if row['normalization_range'][0] == -1 else '#e47632'
                  for row in data['configs']]
        for axis, key, label in zip(axes, ('gaussian_kl_real_to_generated', 'mean_absolute_error_ms'),
                                   ('Mean window Gaussian KL', 'Window mean absolute error (ms)')):
            values = [row[key] for row in data['rows']]
            axis.bar(positions, values, color=colors, width=.7)
            axis.set_ylabel(label)
            axis.grid(axis='y', alpha=.2)
        labels = [f'config_{row["id"]}\n{row["dropout_location"]} '
                  f'{max(row["input_probability"], row["s4_probability"]):g}' for row in data['configs']]
        axes[-1].set_xticks(positions, labels)
        axes[-1].tick_params(axis='x', labelsize=9)
        prefix = 'SYNTHETIC FIXTURE | ' if synthetic_fixture else ''
        fig.suptitle(prefix+'S092 | Current best checkpoints on the same saved segment', fontsize=16)
        fig.text(.5, .92, 'Blue: [-1, 1]; orange: [0, 1]. Lower is better. Same seed and reference-observed scoring positions.',
                 ha='center', fontsize=10)
        fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.12, hspace=.22)
        for extension in ('png', 'svg'):
            path = output/('S092_sweep_metric_comparison.'+extension)
            fig.savefig(path, dpi=180)
            files.append(path)
        plt.close(fig)
    ranked = sorted(data['rows'], key=lambda row: (row['gaussian_kl_real_to_generated'], row['configuration_id']))
    _save_table(output, 'configuration_ranking', [dict(rank=i+1, **row) for i, row in enumerate(ranked)])
    _save_table(output, 'stage_excerpt_metrics', data['stage_rows'])
    _save_table(output, 'per_window_metrics', data['window_rows'])
    manifest = dict(status='completed', synthetic_fixture=synthetic_fixture, model_executed=False,
        source_sha256=data['source_sha256'], seed=data['seed'], configuration_count=len(data['configs']),
        selected_configuration_ids=[row['id'] for row in data['configs']],
        excluded_configuration_ids=sorted(set(range(12))-{row['id'] for row in data['configs']}),
        source_segment=dict(start_second=data['start'], samples=len(data['real']), sample_rate_hz=data['fs'],
                            duration_seconds=len(data['real'])/data['fs'], window_size=data['window_size']),
        stage_selection={name: dict(relative_start_second=offset/data['fs'], duration_seconds=30.)
                         for name, offset in data['stage_starts'].items()},
        stage_selection_rule='First original epoch of each stage; independent of generated quality',
        metric='Equal-weight mean over full model windows of KL(real||generated), univariate Gaussian fits; variance floor 1 ms^2',
        stage_metrics_scope='Separate first 30-second stage excerpts, not the full-window checkpoint-selection metric',
        scoring='Same positive-reference positions for both signals; generated negatives retained',
        checkpoint_epochs={str(index): report['epoch'] for index, report in data['reports'].items()},
        plotted_units='seconds', common_ylim_seconds=list(data['ylim']),
        smoothing=False, clipping=False, decimation=False, real_zeros_retained_as_missing_markers=True,
        stages_are_predictions=False, windows_generated_independently=True,
        artifacts=[dict(file=path.name, sha256=sha256(path)) for path in files])
    (output/'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args(argv)
    report = render_comparison(args.result_dir, args.output_dir, args.synthetic_fixture)
    print(json.dumps(dict(status=report['status'], configuration_count=report['configuration_count'],
                          plots=str(args.output_dir or args.result_dir/'plots'))))


if __name__ == '__main__':
    main()
