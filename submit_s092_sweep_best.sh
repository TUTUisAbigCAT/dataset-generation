#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
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
if [[ ${1:-} == --sweep-root ]]; then
  if (( $# < 3 )); then echo "Expected --sweep-root PATH followed by config IDs" >&2; exit 2; fi
  SWEEP_ROOT=$2
  shift 2
fi
S092_NPZ=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/S092.npz
METADATA=/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE/participant_info.csv
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/S092_all_windows_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"
CONFIG_IDS=(0 1 2 3 4 5 6 7 8)
if (( $# )); then CONFIG_IDS=("$@"); fi
WINDOW_STRIDE_SECONDS=30

# Each model samples the same full set of S092 windows in chronological order.
srun python -B -u scripts/compare_s092_sweep_best.py \
  --sweep-root "$SWEEP_ROOT" --npz "$S092_NPZ" --metadata "$METADATA" \
  --config-ids "${CONFIG_IDS[@]}" \
  --all-windows --window-stride-seconds "$WINDOW_STRIDE_SECONDS" --sampling-batch-size 8 --seed 20261103 \
  --threads "${SLURM_CPUS_PER_TASK:-4}" --output-dir "$OUTPUT_ROOT"

printf 'Window-level statistics: %s/per_window_metrics.csv\nWindow averages: %s/all_window_summary.csv\n' "$OUTPUT_ROOT" "$OUTPUT_ROOT"
