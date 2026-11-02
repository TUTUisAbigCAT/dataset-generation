#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ibi-missing-ab
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/missing_ab_%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
source ./.venv/bin/activate

# Both arms: identical data/split/conditions/schedule, observed-only validation,
# 10 fixed epochs, at most 64 identical validation windows per epoch.
python scripts/run_missing_value_ablation.py --config dataset_config.json --epochs 10 --batch-size 8 --threads 4 --validation-max-windows 64
