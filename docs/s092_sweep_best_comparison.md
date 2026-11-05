# S092 current-best comparison for selected sweep configurations

The submission script compares `config_0` through `config_8` from the all88/S092 sweep using
their current `run/training/best_kl_model.pth`. Training need not have finished:
`run_manifest.json` supplies the frozen model/data settings and
`validation/summary.json` supplies the best epoch, raw KL and weight SHA256.
The final `best_kl_manifest.json` is not required. No final-epoch fallback is
allowed. Every selected configuration must already have a valid best checkpoint.
The default submission uses the nine completed configurations.

## Submit on the cluster

From the repository root:

```bash
git pull --ff-only
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_s092_sweep_best.sh
```

The script already targets
`/home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_1228119`,
and allocates one GPU and four CPU threads for up to 16 hours. The nine inference
workers run sequentially on that GPU in fresh processes; this does not submit
or modify the 12 training jobs. The BLIS startup policy is applied before
NumPy/PyTorch imports. Model inference requires CUDA, with no CPU fallback.

To evaluate only config 1, run `sbatch submit_s092_sweep_best.sh 1`.
To evaluate the three retrained models, use
`sbatch submit_s092_sweep_best.sh --sweep-root /path/to/the/retry_sweep 9 10 11`.
The Python entry point accepts `--config-ids 0 1 2 3 4 5 6 7 8`; omitting that
option retains the original all-12 behavior. Missing or invalid selected runs
still stop the job: exclusion is explicit, never an automatic skip on failure.

Best means the best available when the inference job starts on a compute node,
not the time `sbatch` was called. Each run is captured separately, and each
snapshot records its UTC time and best epoch. They may have different completed
training epochs. Compare these results as current checkpoints, not as a claim
that all models received identical numbers of updates.

## Exact comparison

- The Slurm entry point passes `--all-windows --window-stride-seconds 30`.
  A 120-second window starts at 0, 30, 60 seconds and so on. Adjacent windows
  share 90 seconds of recorded conditions. Each receives its own diffusion
  sample. The audited S092 record has 887 eligible windows at 25 Hz. The last
  starts at 26,580 seconds and reaches the end of the 26,700-second record.
- Source labels, positive IBI masks and the original validation window list
  are checked against the S092 NPZ. Checkpoint selection used the original
  222 nonoverlapping windows. For that inference selection, set
  `WINDOW_STRIDE_SECONDS=120` in the submission script. Supported steps are
  30, 60, 90 and 120 seconds.
- Use S092's recorded AHI/BMI/AGE/GENDER and each run's frozen metadata scaler.
- Use sampling seed **20261103** and check identical CUDA RNG states before and
  after sampling across all selected workers.
- The sampler receives stage and participant conditions and generates each
  window's entire IBI channel. Real IBI supplies the paired evaluation target.
- Restore each model's own S4 dropout setting when constructing it, then call
  `eval()`; training input dropout is not applied during inference.
- Restore the saved normalization, source implementation and diffusion
  schedule. Do not fit any scaler or modify the saved waveforms.

All 88 records, including S092, trained these models. This measures fitting on
S092, not held-out generalization. The two-arm S092 script targets the earlier
66/11/11 runs and shares the checkpoint-loading and sampling worker.

## Live-checkpoint consistency

The launcher reads the validation summary and weight bytes, verifies the weight
SHA256, and rereads the metadata to reject a concurrent update. It retries a
bounded number of times. It copies matching weights to a new inference output
directory and writes an inference snapshot manifest. Workers restore every
parameter and persistent buffer from that copy, including the saved S4
`B/P/w/Q` eigenbasis tensors. Each worker checks every restored tensor against
the checkpoint before sampling and records the result under `checkpoint_load`
in its `config_N/summary.json`. Training files
and source snapshots are never edited. Missing best weights, inconsistent
metadata, changed source/data hashes or unexpected settings stop the job.

## Results

The Slurm log prints a new output directory under
`runs/S092_all_windows_<UTC timestamp>_<job ID>/`:

- `per_window_metrics.csv` contains a row per model and source window, with
  the source start time, observed sample count, paired real/generated means
  (ms) and population variances (ms²), and their signed and absolute differences.
  Signed differences use generated minus reference. The same positive-reference
  positions are scored on both sides.
- `all_window_summary.csv` contains one row per model. Each `avg_*` field is
  the arithmetic mean of its per-window column. Source samples in an overlap
  contribute to each window that contains them. The report records source
  coverage separately from the sum of sampled window lengths. In particular,
  `avg_generated_variance_ms2` averages the within-window population variances.
  `avg_mean_absolute_error_ms` and `avg_variance_absolute_error_ms2` preserve
  the size of errors when different windows have opposite signs.

- `summary.json`: selected configurations, explicit selected/excluded IDs,
  captured epochs/hashes, paired checks, progress and completion status.
- `reference.npz`: the common real IBI, observed mask and original conditions.
- `config_N/checkpoint_snapshot/`: the copied best weights and
  `best_snapshot_manifest.json` identifying their epoch and provenance.
- `config_N/preflight.json`, `sample.log`, `summary.json`: source checks,
  runtime details and Gaussian KL/mean/variance statistics.
- `config_N/waveforms.npz`: generated normalized IBI and IBI in seconds.

For the earlier six-minute comparison and its four-stage plots, invoke the
Python entry point with `--start-second 18900 --windows 3` instead of
`--all-windows`. Full-window statistics go directly into the output root.

For data/configuration checks without model execution, use a fresh directory:

```bash
python -B scripts/compare_s092_sweep_best.py \
  --sweep-root /home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_1228119 \
  --config-ids 0 1 2 3 4 5 6 7 8 \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/S092_sweep_best_check \
  --all-windows --check-only
```

This also captures best snapshots; it does not perform inference or generate
waveforms. A normal Slurm submission uses a separate fresh output directory.
