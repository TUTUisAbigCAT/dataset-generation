#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=4         # CPU 数量
#SBATCH --gpus-per-node=1         # GPU 数量 (去掉这一行表示不用 GPU)
#SBATCH --time=10:00:00           # 最大运行时间 (时:分:秒)
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/openmp_output_%j.txt


module load StdEnv/2023  gcc/12.3
module load opencv/4.9.0
module load python/3.11

# 2. 激活虚拟环境 (Critical)
source ./.venv/bin/activate
# 3. 运行训练代码
# pip install --no-index scikit-learn numpy pandas torch
python scripts/run_dreamt.py --epochs 50 --batch-size 8 --threads 4
