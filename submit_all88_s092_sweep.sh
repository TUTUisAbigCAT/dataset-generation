#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --array=0-11
#SBATCH --job-name=all88-s092
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/all88_s092_%A_%a.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

CONFIG=dataset_config.json
SWEEP_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/all88_s092_sweep_${SLURM_ARRAY_JOB_ID}"

# All 88 records train, including S092; select on current S092 KL each epoch.
# Each range has two intentionally repeated p=0 baseline jobs (input/S4 labels).
srun python -B -u scripts/run_all88_s092_sweep.py \
  --config "$CONFIG" --grid configs/all88_s092_sweep.json \
  --configuration-id "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "$SWEEP_ROOT/config_${SLURM_ARRAY_TASK_ID}" \
  --epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed configuration %s. Outputs: %s\n' "${SLURM_ARRAY_TASK_ID}" "$SWEEP_ROOT/config_${SLURM_ARRAY_TASK_ID}"
