#!/bin/bash
#SBATCH --job-name=ablation5d_horizon_sweep
#SBATCH --output=/home/3199937/slurm_logs/ablation5d_horizon_sweep_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5d_horizon_sweep_%j.err
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

# Ablation 5d — sweep the recurrence horizon along two axes.
#
# Anchor: abl5c-B [3, 4, 5] target=both → 1.3029 bpb (current best).
# Going from 1→2→3 recurred layers each helped (~0.0025 / 0.0019 per step).
# This sweep tests two natural extensions:
#
# Option A: [3, 4, 5, 6]  – extend the horizon by one more layer toward the deep decoder.
#                           Answers: does 4 layers continue the trend, saturate, or regress?
#
# Option B: [2, 3, 4, 5]  – extend the horizon by one more layer toward the encoder side.
#                           Answers: is the win about straddling the U-Net hinge from the
#                           decoder side, or does encoder-side recurrence also help?
#
# Both reuse train_gpt_ablation5b_recur_target.py with target=both and recur_times=1.
# Sequential python invocations so each pins its own RECUR_LAYERS.

# -----------------------------
# Option A — layers [3, 4, 5, 6]
# -----------------------------
RUN_ID=ablation5d_layers3456_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="3,4,5,6" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py

# -----------------------------
# Option B — layers [2, 3, 4, 5]
# -----------------------------
RUN_ID=ablation5d_layers2345_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="2,3,4,5" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
