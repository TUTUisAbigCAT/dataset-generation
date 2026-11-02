"""Select disjoint original-condition windows and score their real 30 s stages.

No reference IBI is used as a model input. Selection uses labels, continuity,
and reference validity only; no generation is performed here. Tensor scoring
stays on the caller's device until small final summaries are serialized.
Production callers enforce CUDA; CPU is supported for numerical unit tests.
"""
from __future__ import annotations

from numbers import Integral
import math

import numpy as np
import torch

from validation_metrics import window_statistics
from watchsleep_loader import EPOCH_SAMPLES, STAGE_NAMES, TARGET_FS


METRIC_FIELDS = (
    'generated_mean_ms', 'reference_mean_ms', 'generated_variance_ms2',
    'reference_variance_ms2', 'mean_error_ms', 'mean_absolute_error_ms',
    'variance_error_ms2', 'variance_absolute_error_ms2',
    'gaussian_kl_real_to_generated', 'kl_mean_term', 'kl_variance_term',
    'generated_nonpositive_fraction',
)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def select_stage_windows(prepared, *, window_epochs, windows_per_stage=8, seed=42):
    """Seeded stage round-robin sampling from a fixed non-overlapping grid.

    The target counts windows containing a stage, not pure-stage windows. A
    mixed window counts toward every stage it contains and is selected once.
    The final stage counts can exceed the target. Neither IBI magnitudes nor
    generated outputs influence ranking. Every selected 30 s epoch has at
    least two positive, finite reference observations for downstream scoring.
    """
    k = _integer(window_epochs, 'window_epochs')
    target = _integer(windows_per_stage, 'windows_per_stage')
    seed = _integer(seed, 'seed', 0)
    length = k * EPOCH_SAMPLES
    conditions = np.asarray(prepared['conditions'])
    reference = np.asarray(prepared['reference_seconds'])
    if reference.ndim != 1 or reference.dtype.kind != 'f':
        raise ValueError('reference_seconds must be a floating-point vector')
    if conditions.shape != (len(reference), 9) or conditions.dtype.kind != 'f':
        raise ValueError('conditions must be floating-point [samples, 9]')
    record = prepared['record']
    epochs = len(reference) // EPOCH_SAMPLES
    stages = np.asarray(record['stage_epochs'])
    valid = np.asarray(record['valid_epochs'])
    adjacent = np.asarray(record['adjacent'])
    if stages.shape != (epochs,) or stages.dtype.kind not in 'iu':
        raise ValueError('record.stage_epochs must contain one integer per complete epoch')
    if valid.shape != (epochs,) or valid.dtype != np.bool_:
        raise ValueError('record.valid_epochs must be a bool vector')
    if adjacent.shape != (max(epochs-1, 0),) or adjacent.dtype != np.bool_:
        raise ValueError('record.adjacent must be a bool vector between epochs')
    if 'stage_epochs_four' in prepared and not np.array_equal(stages, prepared['stage_epochs_four']):
        raise ValueError('Prepared stage labels disagree with the source record')
    epoch_reference = reference[:epochs * EPOCH_SAMPLES].reshape(epochs, EPOCH_SAMPLES)
    observed = np.isfinite(epoch_reference) & (epoch_reference > 0)
    observed_counts = observed.sum(axis=1)
    signal_valid = np.isfinite(epoch_reference).all(axis=1) & (epoch_reference >= 0).all(axis=1)
    candidates, excluded = [], []
    for start_epoch in range(0, epochs-k+1, k):
        end_epoch = start_epoch+k
        start, end = start_epoch*EPOCH_SAMPLES, end_epoch*EPOCH_SAMPLES
        labels = stages[start_epoch:end_epoch]
        reasons = []
        if not valid[start_epoch:end_epoch].all() or not signal_valid[start_epoch:end_epoch].all():
            reasons.append('invalid_reference_or_source_epoch')
        if np.any((labels < 0) | (labels >= len(STAGE_NAMES))):
            reasons.append('unknown_stage')
        if np.any(observed_counts[start_epoch:end_epoch] < 2):
            reasons.append('fewer_than_two_observed_samples_in_epoch')
        if not adjacent[start_epoch:end_epoch-1].all():
            reasons.append('source_time_gap_inside_window')
        block = conditions[start:end]
        if not np.isfinite(block).all():
            reasons.append('nonfinite_conditions')
        elif 'unknown_stage' not in reasons:
            expected = np.repeat(np.eye(4, dtype=np.float32)[labels], EPOCH_SAMPLES, axis=0)
            if not np.array_equal(block[:, :4], expected):
                reasons.append('stage_conditions_disagree_with_labels')
        if reasons:
            excluded.append(dict(start_sample=start, start_epoch=start_epoch, reasons=reasons))
        else:
            candidates.append(dict(start_sample=start, start_epoch=start_epoch,
                stage_epochs=labels.tolist(), observed_counts=observed_counts[start_epoch:end_epoch].tolist()))
    shuffled = np.random.default_rng(seed).permutation(len(candidates)).tolist()
    by_stage = [[idx for idx in shuffled if stage in candidates[idx]['stage_epochs']] for stage in range(4)]
    chosen, stage_counts = set(), [0]*4
    while True:
        added = False
        for stage in range(4):
            if stage_counts[stage] >= target:
                continue
            index = next((idx for idx in by_stage[stage] if idx not in chosen), None)
            if index is None:
                continue
            chosen.add(index)
            for present in set(candidates[index]['stage_epochs']):
                stage_counts[present] += 1
            added = True
        if not added:
            break
    windows = [dict(window_index=i, **candidates[index])
               for i, index in enumerate(sorted(chosen, key=lambda idx: candidates[idx]['start_sample']))]
    stage_report = {}
    for stage, name in enumerate(STAGE_NAMES):
        count = sum(stage in row['stage_epochs'] for row in windows)
        stage_report[name] = dict(stage_id=stage, stage_name=name,
            candidate_windows=len(by_stage[stage]), selected_windows=count,
            selected_epochs=sum(row['stage_epochs'].count(stage) for row in windows),
            target_windows=target, target_met=count >= target,
            status='missing' if not count else ('available' if count >= target else 'insufficient_windows'))
    return dict(schema='original_condition_stage_selection_v1', seed=seed,
        window_epochs=k, window_size=length, epoch_samples=EPOCH_SAMPLES,
        sample_rate_hz=TARGET_FS, fixed_grid_origin_sample=0, candidate_stride_samples=length,
        selection_rule='Seeded round-robin over stage presence; mixed windows allowed; no window selected twice',
        ranking_uses_reference_amplitudes=False, stage_conditions_overridden=False,
        candidate_windows=len(reference)//length, eligible_windows=len(candidates),
        selected_windows=len(windows), start_samples=[row['start_sample'] for row in windows],
        windows=windows, stages=stage_report, excluded_windows=excluded,
        incomplete_window_tail_samples=len(reference) % length,
        incomplete_window_tail_complete_epochs=(len(reference) % length)//EPOCH_SAMPLES,
        source_discarded_tail_samples=int(record.get('discarded_tail_samples', 0)))


def _summarize_rows(rows, *, stage_id=None, stage_name=None):
    output = dict(status='available' if rows else 'missing',
        windows=len({row['window_index'] for row in rows}), epochs=len(rows),
        observed_samples=sum(row['observed_counts'] for row in rows),
        generated_samples_all=sum(row['generated_samples_all'] for row in rows),
        generated_nonpositive_samples_all=sum(row['generated_nonpositive_samples_all'] for row in rows))
    if stage_id is not None:
        output.update(stage_id=stage_id, stage_name=stage_name)
    output.update({key: math.fsum(row[key] for row in rows)/len(rows) if rows else None
                   for key in METRIC_FIELDS})
    return output


@torch.no_grad()
def score_stage_draw(generated_seconds, reference_seconds, stage_epochs_four, *,
                     start_samples, seed, kl_variance_floor_ms2=1.0):
    """Score disjoint original 30 s epochs; reference-positive positions only.

    Signals are [windows, K*750] floating tensors on one device, stages are
    [windows, K] int64 on that same device. All generated positions must be
    finite; negatives remain in the metrics and the all-position diagnostic.
    Each stage's moments/errors/KL are averaged equally over its epochs.
    """
    seed = _integer(seed, 'seed', 0)
    for name, tensor in (('generated_seconds', generated_seconds), ('reference_seconds', reference_seconds)):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 2:
            raise ValueError(f'{name} must be a [windows, samples] floating tensor')
    if generated_seconds.shape != reference_seconds.shape or generated_seconds.device != reference_seconds.device:
        raise ValueError('Generated and reference signals must have the same shape and device')
    count, length = generated_seconds.shape
    if count < 1 or length < EPOCH_SAMPLES or length % EPOCH_SAMPLES:
        raise ValueError('Scoring requires complete 750-sample epochs in nonempty windows')
    k = length//EPOCH_SAMPLES
    if (not torch.is_tensor(stage_epochs_four) or stage_epochs_four.dtype != torch.int64
            or stage_epochs_four.shape != (count,k) or stage_epochs_four.device != generated_seconds.device):
        raise ValueError('stage_epochs_four must be int64 [windows, K] on the signal device')
    if not bool(((stage_epochs_four >= 0) & (stage_epochs_four < 4)).all()):
        raise ValueError('Every scored epoch must have one of the four original stage labels')
    if len(start_samples) != count:
        raise ValueError('start_samples must contain one source start per window')
    starts = [_integer(value, 'start_sample', 0) for value in start_samples]
    if any(value % EPOCH_SAMPLES for value in starts):
        raise ValueError('Window starts must be aligned with original 30-second epochs')
    ordered = sorted(starts)
    if any(right < left+length for left,right in zip(ordered, ordered[1:])):
        raise ValueError('Overlapping source windows would score original epochs more than once')
    generated = generated_seconds.reshape(-1,EPOCH_SAMPLES)
    reference = reference_seconds.reshape(-1,EPOCH_SAMPLES)
    observed_mask = torch.isfinite(reference) & (reference > 0)
    statistics = window_statistics(generated, reference, observed_mask=observed_mask,
                                  kl_variance_floor_ms2=kl_variance_floor_ms2)
    # Only finished, small statistics cross the device boundary for JSON/CSV.
    columns = {name:tensor.tolist() for name,tensor in statistics.items()}
    nonpositive = (generated <= 0).sum(dim=1).tolist()
    labels = stage_epochs_four.reshape(-1).tolist()
    rows = []
    for index,stage in enumerate(labels):
        window, local_epoch = divmod(index,k)
        start = starts[window]+local_epoch*EPOCH_SAMPLES
        row = dict(seed=seed, window_index=window, epoch_in_window=local_epoch,
            source_epoch_index=start//EPOCH_SAMPLES, start_sample=start,
            start_second=start/TARGET_FS, stage_id=stage, stage_name=STAGE_NAMES[stage],
            generated_nonpositive_samples_all=nonpositive[index], generated_samples_all=EPOCH_SAMPLES,
            generated_nonpositive_fraction=nonpositive[index]/EPOCH_SAMPLES)
        row.update({name:values[index] for name,values in columns.items()})
        rows.append(row)
    return dict(seed=seed, window_epochs=k, windows=count, per_epoch=rows,
        stages={name:_summarize_rows([row for row in rows if row['stage_id']==stage],
                                    stage_id=stage,stage_name=name) for stage,name in enumerate(STAGE_NAMES)},
        overall=_summarize_rows(rows),
        protocol=dict(epoch_samples=EPOCH_SAMPLES, sample_rate_hz=TARGET_FS,
            reference_mask='finite_reference_and_reference_greater_than_zero',
            variance='population; correction=0', kl_direction='reference || generated',
            kl_variance_floor_ms2=float(kl_variance_floor_ms2),
            aggregation='Equal weight per original 30-second epoch within each stage',
            generated_nonpositive_denominator='All generated positions, including unscored reference-missing positions',
            temporal_structure_assessed=False, reference_used_as_model_input=False))


def summarize_stage_draws(draw_reports):
    """Across-seed population mean/std; no confidence-interval interpretation."""
    if not draw_reports:
        raise ValueError('At least one scored draw is required')
    seeds = [report['seed'] for report in draw_reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated seeds are not separate sampling draws')
    def identity(report):
        return [(row['source_epoch_index'],row['stage_id'],row['observed_counts'],
                 row['reference_mean_ms'],row['reference_variance_ms2']) for row in report['per_epoch']]
    baseline = identity(draw_reports[0])
    for report in draw_reports[1:]:
        if identity(report) != baseline or report['protocol'] != draw_reports[0]['protocol']:
            raise ValueError('Across-seed summaries require the same source epochs, references and scoring protocol')
    stages = {}
    for stage,name in enumerate(STAGE_NAMES):
        first = draw_reports[0]['stages'][name]
        row = {key:first[key] for key in ('stage_id','stage_name','status','windows','epochs','observed_samples')}
        row.update(draws=len(draw_reports), metrics={})
        for key in METRIC_FIELDS:
            values = [report['stages'][name][key] for report in draw_reports]
            if first['status']=='missing':
                row['metrics'][key] = dict(mean=None,std=None)
            else:
                row['metrics'][key] = dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=0)))
        stages[name] = row
    return dict(draws=len(draw_reports), seeds=seeds, stages=stages,
        uncertainty='Between-seed population standard deviation (ddof=0); not a confidence interval',
        independent_subjects=1, temporal_structure_assessed=False)
