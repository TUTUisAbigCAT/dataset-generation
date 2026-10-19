#!/bin/bash
#SBATCH --account=def-plago       # 换成你的账户名 (def-xxxx)
#SBATCH --cpus-per-task=4         # CPU 数量
#SBATCH --mem=60G                 # 内存
#SBATCH --gpus=h100:1              # GPU 数量 (去掉这一行表示不用 GPU)
#SBATCH --time=24:00:00            # 最大运行时间 (时:分:秒)
# 2. 激活虚拟环境 (Critical)
source ./venv/bin/activate

# 3. 运行训练代码
pip install --no-index scikit-learn numpy pandas torch
python scripts/run_dreamt.py --epochs 10 --batch-size 8 --threads 4
