#!/bin/bash
#SBATCH --job-name=dreamt-s092
#SBATCH --account=def-plago
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --chdir=/project/def-plago/yxtao/dataset-generation
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/inference_s092_%j.txt

set -e

module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

srun python -u scripts/generate_night.py \
    --checkpoint /home/yxtao/links/scratch/dataset-generation/runs/dreamt_20261026T234653_562786Z/training/best_kl_model.pth \
    --config dataset_config.json \
    --subject S092 \
    --seed 42 \
    --threads 4 \
    --stride-epochs 1 \
    --windows-per-group 4 \
    --output-dir "/home/yxtao/links/scratch/dataset-generation/runs/S092_generated_${SLURM_JOB_ID}"
