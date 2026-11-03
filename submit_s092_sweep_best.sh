#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=s092-sweep-best
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/s092_sweep_best_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export PYTHONFAULTHANDLER=1

SWEEP_ROOT=/home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_1228119
S092_NPZ=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/S092.npz
METADATA=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/participant_info.csv
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/S092_sweep_best_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"

# Capture all 12 current best weights, including runs still training.
# The same 18900--19260s S092 conditions cover W, Light, REM and Deep.
# One GPU runs the 12 small inference jobs sequentially in isolated processes.
srun python -B -u scripts/compare_s092_sweep_best.py \
  --sweep-root "$SWEEP_ROOT" --npz "$S092_NPZ" --metadata "$METADATA" \
  --start-second 18900 --windows 3 --seed 20261103 \
  --threads "${SLURM_CPUS_PER_TASK:-4}" --output-dir "$OUTPUT_ROOT"

printf 'Completed all 12 best-checkpoint comparisons. Results: %s\n' "$OUTPUT_ROOT"
