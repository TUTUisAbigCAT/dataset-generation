#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=s092-minmax-best
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/s092_best_minmax_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

MINMAX_RUN=/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635
MINMAX01_RUN=/home/yxtao/links/scratch/dataset-generation/runs/minmax01_20261103T055325Z_1225443
S092_NPZ=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/S092.npz
METADATA=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/participant_info.csv
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/S092_best_minmax_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"

# Shared 18900--19260s segment covers W, Light, REM and Deep.
# Three independent 120-second windows; no training or overlap continuation.
# Best E27 and E29 are taken from their own frozen run/workspace and scalers.
srun python -B -u scripts/compare_s092_best_minmax.py \
  --minmax-run "$MINMAX_RUN" --minmax01-run "$MINMAX01_RUN" \
  --npz "$S092_NPZ" --metadata "$METADATA" \
  --start-second 18900 --windows 3 --require-all-stages \
  --seed 20261103 --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --output-dir "$OUTPUT_ROOT"

printf 'Completed CUDA comparison. Results and plots: %s\n' "$OUTPUT_ROOT"
