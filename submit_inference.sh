#!/bin/bash
#SBATCH --job-name=dreamt-s092
#SBATCH --account=def-plago
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/inference_s092_%j.txt

# Run sbatch from the project root; create the log directory before submission.
# Usage: sbatch submit_inference.sh /absolute/path/to/best.pth [absolute_output_dir] [absolute_config_json]
set -eo pipefail

if (( $# < 1 || $# > 3 )); then
    printf 'Usage: sbatch submit_inference.sh /absolute/path/to/best.pth [absolute_output_dir] [absolute_config_json]\n' >&2
    exit 2
fi

CHECKPOINT="$1"
if [[ "$CHECKPOINT" != /* || ! -f "$CHECKPOINT" || ! -r "$CHECKPOINT" ]]; then
    printf 'Checkpoint must be an existing readable absolute file path: %s\n' "$CHECKPOINT" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
if [[ ! -f scripts/generate_night.py || ! -f .venv/bin/activate ]]; then
    printf 'Submit from the project root containing scripts/generate_night.py and .venv/bin/activate.\n' >&2
    exit 2
fi

CONFIG_PATH="${3:-$PWD/dataset_config.json}"
if [[ "$CONFIG_PATH" != /* || ! -f "$CONFIG_PATH" || ! -r "$CONFIG_PATH" ]]; then
    printf 'Config must be an existing readable absolute file path: %s\n' "$CONFIG_PATH" >&2
    exit 2
fi

OUTPUT_DIR="${2:-/home/yxtao/links/scratch/dataset-generation/runs/S092_generated_${SLURM_JOB_ID:?Missing Slurm job ID}}"
if [[ "$OUTPUT_DIR" != /* || -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    printf 'Output must be an absolute path to a new directory: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi

module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

printf 'S092 inference | job=%s | checkpoint=%s | config=%s | output=%s\n' "${SLURM_JOB_ID}" "$CHECKPOINT" "$CONFIG_PATH" "$OUTPUT_DIR"
exec srun --ntasks=1 python -u scripts/generate_night.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_PATH" \
    --subject S092 \
    --seed 42 \
    --threads "${SLURM_CPUS_PER_TASK:-4}" \
    --stride-epochs 1 \
    --windows-per-group 4 \
    --output-dir "$OUTPUT_DIR"
