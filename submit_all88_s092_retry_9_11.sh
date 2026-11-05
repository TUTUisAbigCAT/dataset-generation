#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --array=9-11
#SBATCH --job-name=all88-retry
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/all88_retry_%A_%a.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the dataset-generation repository}"
case "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}" in
  9|10|11) ;;
  *) echo "Expected configuration 9, 10 or 11" >&2; exit 2 ;;
esac

RUNS=/home/yxtao/links/scratch/dataset-generation/runs
CONFIG="$RUNS/all88_s092_sweep_1228119/config_0/run/workspace/dataset_config.json"
OUTPUT="$RUNS/all88_s092_retry_1228119_${SLURM_ARRAY_JOB_ID}/config_${SLURM_ARRAY_TASK_ID}"
if [[ ! -r "$CONFIG" ]]; then
  echo "Cannot read the original sweep configuration: $CONFIG" >&2
  exit 2
fi

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

# The driver applies the selected row's normalization and dropout to the frozen base.
printf 'Training config_%s for 50 epochs. Results: %s\n' "$SLURM_ARRAY_TASK_ID" "$OUTPUT"
srun python -B -u scripts/run_all88_s092_sweep.py \
  --config "$CONFIG" --grid configs/all88_s092_sweep.json \
  --configuration-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT" \
  --epochs 50 --batch-size 8 --threads "$OMP_NUM_THREADS"
