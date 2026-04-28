#!/bin/bash
#SBATCH --job-name=ablation3c_asym_init
#SBATCH --output=/home/3199937/slurm_logs/ablation3c_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation3c_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3199937/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

# Asymmetric routing init experiment — fixed at psl=4 (confirmed optimum from ablation3/3b).
#
# Instead of initialising all post_lambdas to 1 (symmetric), we start from the
# expected specialised solution:
#   attn → lane0 = 1,  attn → lane1 = 0   (attention stays in its own lane)
#   mlp  → lane0 = 0,  mlp  → lane1 = 1   (MLP stays in its own lane)
#
# Two runs for direct comparison:
#   PARALLEL_ASYM_INIT=0  — symmetric init   (intra-ablation baseline, reproduces abl3 psl=4)
#   PARALLEL_ASYM_INIT=1  — asymmetric init  (the experiment)

RUN_ID=ablation3c_asym_init \
PARALLEL_START_LAYER_VALUES=4 \
PARALLEL_ASYM_INIT=0 \
SEEDS=1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py

RUN_ID=ablation3c_asym_init \
PARALLEL_START_LAYER_VALUES=4 \
PARALLEL_ASYM_INIT=1 \
SEEDS=1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py
