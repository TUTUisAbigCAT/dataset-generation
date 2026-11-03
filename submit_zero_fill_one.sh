#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-zero-fill-one
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/zero_fill_one_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

REFERENCE_RUN=/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/zero_fill_one_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"

# A fresh model and optimizer; the old preserve/masked runs are read only.
# The runner performs strict identity checks before starting CUDA training.
# Do not mkdir OUTPUT_ROOT here: the runner requires a new output directory.
srun python -u scripts/run_preprocessing_experiment.py \
  --experiment zero_fill_one \
  --reference-run "$REFERENCE_RUN" \
  --output-dir "$OUTPUT_ROOT" \
  --epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed fresh zero-to-one-second training. Outputs: %s\n' "$OUTPUT_ROOT"
