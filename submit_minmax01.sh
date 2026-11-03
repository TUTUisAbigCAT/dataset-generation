#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-minmax01
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/minmax01_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

REFERENCE_RUN=/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/minmax01_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"

# One fresh run: retain raw zeros and change only the clean IBI output range.
# The runner requires a new output directory; do not create OUTPUT_ROOT here.
srun python -u scripts/run_minmax_experiment.py \
  --experiment minmax01 \
  --reference-run "$REFERENCE_RUN" \
  --output-dir "$OUTPUT_ROOT" \
  --epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed fresh minmax [0,1] training. Outputs: %s\n' "$OUTPUT_ROOT"
