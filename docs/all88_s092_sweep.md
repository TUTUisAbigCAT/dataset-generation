# All 88 subjects train; S092 selects the checkpoint

This experiment uses every one of the 88 local DREAMT NPZ records for training,
including S092. There is no held-out validation or test set. After every epoch,
the model independently generates all eligible nonoverlapping two-minute windows
of S092. The **current epoch's mean per-window Gaussian KL(real || generated)**
selects `best_kl_model.pth`: smaller is better, the first epoch is eligible, and
equal values retain the earlier checkpoint. There is no moving average.

This evaluates fitting to a subject present in training. It is not an estimate
of performance on an unseen subject. S092's real IBI is used for scoring, not
as a known IBI prefix or an inpainting input. Its stages and participant metadata
are the generation conditions. Evaluation windows do not overlap or stitch.

## Twelve jobs

| Array IDs | Normalized range | Input dropout probabilities | S4 dropout probabilities |
|---|---|---|---|
| 0–2 | `[-1,1]` | 0, 0.03, 0.05 | 0 |
| 3–5 | `[-1,1]` | 0 | 0, 0.03, 0.05 |
| 6–8 | `[0,1]` | 0, 0.03, 0.05 | 0 |
| 9–11 | `[0,1]` | 0 | 0, 0.03, 0.05 |

The two zero-probability jobs within each range have identical computational
settings and the same seed. They are intentionally retained under input/S4
labels to give 12 requested jobs (10 unique parameter settings), not different
random-seed replicates. Their results may be identical on the same runtime.

`configs/all88_s092_sweep.json` contains the labeled grid.
`scripts/run_all88_s092_sweep.py` creates the complete per-task configuration
from the data paths in `dataset_config.json`; it does not load old checkpoints.

## Training and evaluation settings

- Fresh training for 30 epochs, batch size 8, training seed 42. Early termination
  is disabled so all 12 jobs have the same epoch budget.
- Four 30-second sleep epochs per model window, 25 Hz, 3,000 IBI samples.
  Training windows start every 30 seconds and stay within each subject.
- Exactly 88 NPZ files are required. Every subject must produce at least one
  legal training window. Existing `splits.json` is ignored in this explicit mode.
- Metadata scaling is refit over all 88 subjects. IBI uses fixed physical bounds
  `[0,2]` seconds: `z = IBI_seconds - 1` for `[-1,1]`, or
  `z = IBI_seconds / 2` for `[0,1]`. Inverse transforms are `z+1` and `2*z`.
- Stored raw zeros remain training targets. There is no new zero filling,
  interpolation, masking loss or clipping. Both reference and generated moments
  for evaluation use the same positions where reference IBI is positive.
- Input dropout masks individual noised IBI inputs to normalized zero, without
  rescaling. S4 dropout uses the existing eight S4 output Dropout2d modules.
  Both are disabled during evaluation. Native S4 dropout consumes the training
  RNG, so subsequent training noise draws need not match input-dropout jobs.
- S4 architecture, epsilon loss, Adam configuration and the corrected full
  200-step sampler are unchanged. Schedule is `linear_power`, terminal alpha-bar
  `1e-8`.
- Evaluation uses S092 only, with no 64-window cap. All complete, valid,
  nonoverlapping windows are fixed before training and reused with seed 20261020
  each epoch. An incomplete final window is not padded or scored. A window needs
  at least two observed reference points for its Gaussian moments.
- KL is computed separately within each two-minute reference/generated window
  pair and then averaged across the selected S092 windows. Best selection uses
  this epoch aggregate directly, not an average across epochs.

The default `dataset_config.json` now explicitly records:

```json
{
  "evaluation_protocol": {
    "mode": "all_subjects_train_single_subject_eval",
    "subject": "S092",
    "expected_subjects": 88
  },
  "early_stopping": {
    "enabled": false,
    "patience": 10,
    "selection_mode": "current",
    "smoothing_window": 1,
    "min_delta_kl": 0.0
  }
}
```

`smoothing_window: 1` is a compatibility setting; current mode bypasses the
moving-average calculation entirely. New `epoch_metrics.csv` files contain
`gaussian_kl_real_to_generated`, without smoothed KL or smoothing-ready columns.
Historical configurations without `evaluation_protocol` retain their previous
split behavior. Historical configurations without `selection_mode` retain their
smoothed selection behavior. Internal legacy `splits` metadata records all 88
under `train` with empty `val`/`test`; `evaluation_filenames` separately records
S092, and the protocol explicitly marks evaluation as in-sample.

## Submit

From the cluster repository root:

```bash
git pull --ff-only
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_all88_s092_sweep.sh
```

The script uses the current `.venv`, the existing modules/account, one GPU and
four CPU threads per task, and `--array=0-11%4` (at most four concurrent tasks).
The time limit is 10 hours per task. CUDA is required for model execution.
The earlier `submit_dropout_sweep.sh` is the separate historical 16-job setup;
use `submit_all88_s092_sweep.sh` for this protocol.

For configuration/data validation without training, select a new output path:

```bash
python -B scripts/run_all88_s092_sweep.py \
  --config dataset_config.json --configuration-id 0 --check-only \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/all88_s092_check
```

Changing the data location only requires updating `data_dir` and, if necessary,
`metadata_path` in `dataset_config.json`. The exact subject count and S092 name
are checked before training. Every task freezes its own source and configuration.

## Outputs

Results are under `runs/all88_s092_sweep_<array_job_id>/config_<id>/`:

- `summary.json`: protocol, configuration, subject/window counts, completion
  checks, final KL and best checkpoint.
- `config.json` and `preflight/preflight.json`: resolved configuration and exact
  data/window/scaler identities, including all 88 per-subject training counts.
- `training.log`: detailed subprocess progress or failure.
- `run/training/epoch_metrics.csv`: epoch loss, current KL and selection flags.
- `run/training/best_kl_model.pth` and `best_kl_manifest.json`: best current-KL
  weights and their matching frozen configuration/epoch/hash.
- `run/training/validation/epoch_XXXX/`: generated waveforms and per-window
  metrics from that epoch's S092 evaluation.
- `run/workspace/saved_models/DREAMT/model.pth`: final-epoch weights, which may
  differ from the selected best checkpoint.

Slurm logs are `logs/all88_s092_<array_job_id>_<id>.txt`. A failed or interrupted
task is not labeled completed. After training, the launcher verifies that the
best epoch is the first minimum of the raw KL column, its saved weight hash
matches, and the full requested update budget was completed.
