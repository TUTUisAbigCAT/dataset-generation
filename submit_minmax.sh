#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-minmax
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/minmax_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

REFERENCE_RUN=/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/minmax_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"

# Independent experiment: keep stored zeros, change only clean-signal scaling.
# Fresh model and optimizer. The old preserve/masked runs are read only.
# Do not mkdir OUTPUT_ROOT here: the runner requires a new output directory.
srun python -u scripts/run_preprocessing_experiment.py \
  --experiment minmax \
  --reference-run "$REFERENCE_RUN" \
  --output-dir "$OUTPUT_ROOT" \
  --epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed fresh minmax training. Outputs: %s\n' "$OUTPUT_ROOT"
