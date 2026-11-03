#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --array=0-15%4
#SBATCH --job-name=ibi-dropout-grid
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/dropout_sweep_%A_%a.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

REFERENCE_RUN=/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635
SWEEP_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/dropout_sweep_${SLURM_ARRAY_JOB_ID}"

# 2 normalization ranges x 2 separately active dropout locations x 4 probabilities.
# Every array task starts fresh with its own frozen workspace and output directory.
srun python -B -u scripts/run_dropout_sweep.py \
  --grid configs/dropout_sweep.json --configuration-id "${SLURM_ARRAY_TASK_ID}" \
  --reference-run "$REFERENCE_RUN" \
  --output-dir "$SWEEP_ROOT/config_${SLURM_ARRAY_TASK_ID}" \
  --epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed dropout configuration %s. Outputs: %s\n' "${SLURM_ARRAY_TASK_ID}" "$SWEEP_ROOT/config_${SLURM_ARRAY_TASK_ID}"
