#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-stage-S092
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/stage_eval_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

ABLATION_RUN=/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs
OUTPUT_ROOT="/home/yxtao/links/scratch/dataset-generation/runs/S092_stage_eval_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local}"

# Evaluate the two final epoch-30 models. No training, best-model selection,
# reference-IBI observations, known prefix, or whole-night overlap continuation.
# The evaluation CLI creates a new output directory and refuses to overwrite it.
python scripts/evaluate_stage_generation.py \
  --ablation-run "$ABLATION_RUN" --subject S092 --expected-epochs 30 \
  --repeats 8 --seed 20261102 --selection-seed 42 --windows-per-stage 8 \
  --batch-size 8 --threads 4 --output-dir "$OUTPUT_ROOT"

# Plot saved results only; this command does not sample again.
python scripts/plot_stage_evaluation.py \
  --input-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/plots"

printf 'Completed S092 stage evaluation. Outputs: %s\n' "$OUTPUT_ROOT"
