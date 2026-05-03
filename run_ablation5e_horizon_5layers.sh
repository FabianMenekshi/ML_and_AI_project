#!/bin/bash
#SBATCH --job-name=ablation5e_horizon_5layers
#SBATCH --output=/home/3199937/slurm_logs/ablation5e_horizon_5layers_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5e_horizon_5layers_%j.err
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

# Ablation 5e — push the recurrence horizon from 4 → 5 layers.
#
# Anchor: 5d-B [2, 3, 4, 5] target=both → 1.3010 bpb (current best).
# 5d showed encoder-side extension (adding layer 2) helped much more than
# decoder-side extension (adding layer 6). This sweep tests whether the
# encoder-side trend continues at 5 layers, and whether combining both
# extensions adds anything.
#
# Option A: [1, 2, 3, 4, 5]  – one more encoder-side step.
#                              Does the trend continue, or does layer 1 saturate?
#
# Option B: [2, 3, 4, 5, 6]  – combine encoder-side win (layer 2) with decoder-side
#                              extension (layer 6). Tells us whether the two extensions
#                              add or whether the encoder side dominates standalone.
#
# Both reuse train_gpt_ablation5b_recur_target.py with target=both, recur_times=1.
# Each run takes ~55 minutes (5 recurred layers, +3 min/layer over 38-min baseline).

# -----------------------------
# Option A — layers [1, 2, 3, 4, 5] (encoder-side extension)
# -----------------------------
RUN_ID=ablation5e_layers12345_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="1,2,3,4,5" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py

# -----------------------------
# Option B — layers [2, 3, 4, 5, 6] (combined extension)
# -----------------------------
RUN_ID=ablation5e_layers23456_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="2,3,4,5,6" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
