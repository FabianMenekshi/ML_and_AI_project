#!/bin/bash
#SBATCH --job-name=ablation4_psl4_multiseed
#SBATCH --output=/home/3199937/slurm_logs/ablation4_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation4_%j.err
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

# Multi-seed validation of the best parallel_start_layer from ablation 3.
# psl=4 was the clear winner (+0.0052 bpb vs sequential, +0.0069 vs baseline).
# Running all 3 seeds to confirm the improvement is real and not noise.

RUN_ID=ablation4_psl4_multiseed \
PARALLEL_START_LAYER_VALUES=4 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py
