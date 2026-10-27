#!/bin/bash
#SBATCH --job-name=dreamt-diagnose
#SBATCH --account=def-plago
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --chdir=/project/def-plago/yxtao/dataset-generation
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/diagnostics_s092_%j.txt

set -e

module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

srun python -u scripts/diagnose_generation.py \
    --checkpoint /home/yxtao/links/scratch/dataset-generation/runs/dreamt_20261026T234653_562786Z/training/best_kl_model.pth \
    --training-manifest /home/yxtao/links/scratch/dataset-generation/runs/dreamt_20261026T234653_562786Z/training/run_manifest.json \
    --validation-summary /home/yxtao/links/scratch/dataset-generation/runs/dreamt_20261026T234653_562786Z/training/validation/summary.json \
    --inference-summary /home/yxtao/links/scratch/dataset-generation/runs/S092_generated_1179820/summary.json \
    --config dataset_config.json \
    --subject S092 \
    --repeats 8 \
    --comparison-repeats 3 \
    --seed 42 \
    --threads 4 \
    --stride-epochs 1 \
    --windows-per-group 4 \
    --output-dir "/home/yxtao/links/scratch/dataset-generation/runs/S092_diagnostics_${SLURM_JOB_ID}"
