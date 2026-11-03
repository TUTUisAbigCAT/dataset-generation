# S092 current-best comparison for all 12 configurations

This compares `config_0` through `config_11` from the all88/S092 sweep using
their current `run/training/best_kl_model.pth`. Training need not have finished:
`run_manifest.json` supplies the frozen model/data settings and
`validation/summary.json` supplies the best epoch, raw KL and weight SHA256.
The final `best_kl_manifest.json` is not required. No final-epoch fallback is
allowed. Every configuration must already have a valid best checkpoint.

## Submit on the cluster

From the repository root:

```bash
git pull --ff-only
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_s092_sweep_best.sh
```

The script already targets
`/home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_1228119`,
as shown in the supplied directory tree. Change only `SWEEP_ROOT` if the run
directory differs. Data and metadata paths are also directly in that script.
It allocates one GPU and four CPU threads for up to one hour. The 12 inference
workers run sequentially on that GPU in fresh processes; this does not submit
or modify the 12 training jobs. The BLIS startup policy is applied before
NumPy/PyTorch imports. Model inference requires CUDA, with no CPU fallback.

Best means the best available when the inference job starts on a compute node,
not the time `sbatch` was called. Each run is captured separately, and each
snapshot records its UTC time and best epoch. They may have different completed
training epochs. Compare these results as current checkpoints, not as a claim
that all models received identical numbers of updates.

## Exact comparison

- Use the same S092 NPZ segment: **18900--19260 seconds** (6 minutes).
- Generate three independent **120-second / 3000-point** windows at 25 Hz.
- Preserve the original stage sequence:
  `W Light Light Light | REM Light Light Light | Deep Light Light Deep`.
- Use S092's recorded AHI/BMI/AGE/GENDER and each run's frozen metadata scaler.
- Use sampling seed **20261103** and check identical CUDA RNG states before and
  after sampling across all 12 workers.
- Real IBI is used only for evaluation and plotting, not fed as a prefix or
  known signal. There is no overlap, inpainting or continuation in this test.
- Restore each model's own S4 dropout setting when constructing it, then call
  `eval()`; training input dropout is not applied during inference.
- Restore the saved normalization, source implementation and diffusion
  schedule. Do not fit any scaler or modify the saved waveforms.

All 88 records, including S092, trained these models. This measures fitting on
S092, not held-out generalization. The old two-arm S092 script remains unchanged
and is for the earlier 66/11/11 runs only.

## Live-checkpoint consistency

The launcher reads the validation summary and weight bytes, verifies the weight
SHA256, and rereads the metadata to reject a concurrent update. It retries a
bounded number of times. It copies matching weights to a new inference output
directory and writes an inference snapshot manifest. Workers load only that
copy, so subsequent best updates cannot alter this comparison. Training files
and source snapshots are never edited. Missing best weights, inconsistent
metadata, changed source/data hashes or unexpected settings stop the job.

## Results

The Slurm log prints a new output directory under
`runs/S092_sweep_best_<UTC timestamp>_<job ID>/`:

- `summary.json`: all 12 configurations, captured epochs/hashes, paired checks,
  progress and completion status.
- `reference.npz`: the common real IBI, observed mask and original conditions.
- `config_N/checkpoint_snapshot/`: the copied best weights and
  `best_snapshot_manifest.json` identifying their epoch and provenance.
- `config_N/preflight.json`, `sample.log`, `summary.json`: source checks,
  runtime details and three-window Gaussian KL/mean/variance statistics.
- `config_N/waveforms.npz`: generated normalized IBI and IBI in seconds.
- `plots/`: a four-stage figure and full six-minute figure for each configuration,
  plus comparison statistics. The four-stage panels show the first 30-second
  occurrence of W, Light, REM and Deep in this segment. They are selected using
  recorded labels, never by generated quality.

Reference zeros remain visible as missing-value markers and generated negative
values remain visible. Stages on the lower axes are the shared input labels,
not predicted stages. Main ranking uses the mean of the three paired 120-second
Gaussian KL values at the same positive-reference positions; 30-second stage
details are separately labeled.

For data/configuration checks without model execution, use a fresh directory:

```bash
python -B scripts/compare_s092_sweep_best.py \
  --sweep-root /home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_1228119 \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/S092_sweep_best_check \
  --start-second 18900 --windows 3 --check-only
```

This also captures best snapshots; it does not perform inference or generate
waveforms. A normal Slurm submission uses a separate fresh output directory.
