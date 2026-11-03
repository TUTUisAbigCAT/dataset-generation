# Dropout sweep: 16 configurations

The design is **2 normalization ranges × 2 separately tested dropout locations × 4 probabilities = 16 runs**. Each run enables one location and sets the other to zero. It does not combine nonzero input and S4 dropout in the same run.

The configuration grid is `configs/dropout_sweep.json`. Each array task selects one row and writes its complete data/training configuration to a new output directory.

| Array IDs | Clean IBI range | `training_input_dropout.probability` | `s4_dropout` |
|---|---|---|---|
| 0–3 | `[-1,1]` | 0.01, 0.03, 0.05, 0.1 | 0 |
| 4–7 | `[-1,1]` | 0 | 0.01, 0.03, 0.05, 0.1 |
| 8–11 | `[0,1]` | 0.01, 0.03, 0.05, 0.1 | 0 |
| 12–15 | `[0,1]` | 0 | 0.01, 0.03, 0.05, 0.1 |

## What each setting does

- **Input dropout:** after forward diffusion, randomly set individual values of the noised IBI channel `x_t` to normalized zero. Use a dedicated device RNG with seed 42. Do not rescale retained values. Keep sleep-stage/person conditions, epsilon targets, and the full-point MSE unchanged. Validation and inference do not apply this operation.
- **S4 dropout:** use the existing `s4_dropout` argument and the original `S4Layer.dropout` modules. Four residual blocks contain eight such modules. Each applies native `nn.Dropout2d` to `[batch, channels, time]` at the S4 output, before the local residual addition and LayerNorm. Its channel mask is shared over time, and retained activations are scaled by `1/(1-p)`. `model.eval()` disables it during validation/inference. The inner `S4` module's separate dropout setting is unchanged.

These are different interventions, so equal probabilities do not mean equivalent perturbations. Native S4 dropout uses the training RNG; input dropout uses its own RNG. Runs start with the same seed and validation seeds, but S4 dropout changes subsequent training random draws. This sweep uses one training seed per configuration.

## Shared experiment settings

- Start fresh; never resume the reference checkpoint or optimizer.
- 30 epochs, batch size 8, four CPU threads for loading/dispatch; model training and sampling require CUDA.
- Same source records, subject split, metadata, training windows and fixed 64 validation windows as the completed `minmax_20261103T003905Z_1223635` run.
- Four 30-second epochs per window: 3,000 points at 25 Hz, training stride 30 seconds.
- The baseline records 51,266 training windows, 6,409 updates per epoch, and 192,270 updates per run. The launcher checks the completed budget.
- Preserve stored raw zero values in training. Map `[0,2]` seconds to either `[-1,1]` (`z = IBI_seconds - 1`) or `[0,1]` (`z = IBI_seconds / 2`); inverse transforms are `z + 1` and `2*z`. Do not clip noised inputs or generated outputs.
- Keep the reference S4 architecture, diffusion schedule, 200 reverse steps, optimizer, and loss. Every epoch evaluates the same windows using the existing per-window Gaussian KL and observed-reference mask.
- Disable early termination to keep budgets equal. Retain the reference best-checkpoint selection rule. The final comparison CSV compares epoch 30; best-checkpoint metadata is saved separately.

Before training, the launcher checks reference source files, raw-data identity, transformations, splits, validation settings and requested budget. During training, it records the configured eight S4 dropout modules and input-dropout counters. Frozen workspaces include the S4 configuration helper so evaluation reloads the recorded probability.

## Submit on the cluster

Run from the repository root:

```bash
git pull --ff-only
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_dropout_sweep.sh
```

`submit_dropout_sweep.sh` requests one GPU per task and uses `--array=0-15%4`, so at most four configurations run concurrently. It uses the existing modules and `.venv`, with a 10-hour limit per task. The baseline path is written directly in the script:

```text
/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635
```

To rerun only one configuration, override the array selection, for example `sbatch --array=5 submit_dropout_sweep.sh`. A new Slurm job ID gives it a separate output directory.

For a data/configuration preflight without model execution, use a new output directory:

```bash
python -B scripts/run_dropout_sweep.py \
  --configuration-id 5 \
  --reference-run /home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635 \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/dropout_preflight_5 \
  --epochs 30 --batch-size 8 --threads 4 --check-only
```

## Outputs

Each task writes to `runs/dropout_sweep_<array_job_id>/config_<id>/`:

- `config.json`: complete selected configuration.
- `summary.json`: configuration, verification results, completion status, budget and final comparison.
- `training.log`: training subprocess output; inspect this if the Slurm task reports a subprocess failure.
- `comparison.csv`: baseline and selected run at the same final epoch.
- `run/training/epoch_metrics.csv`: per-epoch loss and validation KL.
- `run/training/validation/`: saved validation artifacts and best-checkpoint selection metadata.
- `run/training/best_kl_model.pth`: best selected checkpoint, when produced by the existing validator.
- `run/workspace/saved_models/DREAMT/model.pth`: final epoch-30 checkpoint.
- `run/workspace/`: frozen source and configuration used by this task.

Slurm output is `logs/dropout_sweep_<array_job_id>_<id>.txt`. A `checked` summary only confirms preflight; training is complete only when `status` is `completed` and `training_completed` is `true`.
