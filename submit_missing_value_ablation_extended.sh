#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-missing-ab30
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/missing_ab_extended_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

REFERENCE_RUN=/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_20261102T055527_103986Z
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local}"
REFERENCE_EPOCH=10
EXTENDED_EPOCHS=30
printf -v REFERENCE_EPOCH_DIR 'epoch_%04d' "$REFERENCE_EPOCH"
printf -v EXTENDED_EPOCH_DIR 'epoch_%04d' "$EXTENDED_EPOCHS"
mkdir "$OUTPUT_ROOT"

# First export both old final-epoch checkpoints on the same 64 validation windows.
# Any failure stops the job before new training; the reference directory is read only.
python scripts/export_ablation_validation.py \
  --reference-run "$REFERENCE_RUN" --output-dir "$OUTPUT_ROOT/epoch10_diagnostics" --threads 4
python scripts/plot_validation_comparison.py \
  --preserve-dir "$OUTPUT_ROOT/epoch10_diagnostics/preserve/validation/$REFERENCE_EPOCH_DIR" \
  --masked-dir "$OUTPUT_ROOT/epoch10_diagnostics/masked/validation/$REFERENCE_EPOCH_DIR" \
  --output-dir "$OUTPUT_ROOT/epoch10_plots"

# Fresh models and Adam optimizers, not checkpoint resume. Reuse each old arm's
# config; verify data/scaler/split/schedule/window/batch identity before training.
# Each arm: 51,266 windows / batch 8 -> 6,409 updates per epoch -> 192,270 total.
python scripts/run_missing_value_ablation.py \
  --reference-run "$REFERENCE_RUN" --output-dir "$OUTPUT_ROOT/training_30epochs" \
  --epochs "$EXTENDED_EPOCHS" --batch-size 8 --threads 4 --validation-max-windows 64 \
  --require-equal-updates --save-validation-artifacts
python scripts/plot_validation_comparison.py \
  --preserve-dir "$OUTPUT_ROOT/training_30epochs/preserve/training/validation/$EXTENDED_EPOCH_DIR" \
  --masked-dir "$OUTPUT_ROOT/training_30epochs/masked/training/validation/$EXTENDED_EPOCH_DIR" \
  --output-dir "$OUTPUT_ROOT/epoch30_plots"

printf 'Completed A/B diagnostics and fresh training. Outputs: %s\n' "$OUTPUT_ROOT"
