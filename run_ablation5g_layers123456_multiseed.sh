#!/bin/bash
#SBATCH --job-name=ablation5g_layers123456_multiseed
#SBATCH --output=/home/3199937/slurm_logs/ablation5g_layers123456_multiseed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5g_layers123456_multiseed_%j.err
#SBATCH --time=04:00:00
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

# Multi-seed validation of the 6-layer depth recurrence config [1, 2, 3, 4, 5, 6].
#
# State going in:
#   abl5f probe [1, 2, 3, 4, 5, 6] target=both seed=1337 → 1.2999 bpb (single seed)
#   abl5f multiseed [2, 3, 4, 5]   target=both 3 seeds   → 1.3022 ± 0.0029 (4-layer winner)
#
# The 6-layer single-seed advantage over 4-layer is only 0.0011 bpb, well within
# the 4-layer std of 0.0029 — could be noise. Multi-seed is required to decide
# whether the 6-layer config should replace the 4-layer winner for downstream
# composition with parallel residuals.
#
# Seed 1337 already done in 5f — only running the missing seeds 42 and 123.
# Same seed grid as abl4 (parallel residuals) and abl5f (4-layer DR) for clean
# apples-to-apples comparison.
#
# Each run ~57 min (6 recurred layers); two seeds back-to-back ≈ 1h 55m total.

RUN_ID=ablation5g_layers123456_multiseed \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="1,2,3,4,5,6" \
RECUR_TIMES=1 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
