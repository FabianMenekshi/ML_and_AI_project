#!/bin/bash
#SBATCH --job-name=ablation5f_validate_and_6layer
#SBATCH --output=/home/3199937/slurm_logs/ablation5f_validate_and_6layer_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5f_validate_and_6layer_%j.err
#SBATCH --time=06:00:00
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

# Ablation 5f — two things in one job:
#
# 1) PROBE: 6-layer depth recurrence on [1, 2, 3, 4, 5, 6].
#    Why this window: it is the natural union of the two 5e configs
#      5e-A [1, 2, 3, 4, 5]   (encoder-side ext.)  → 1.3005
#      5e-B [2, 3, 4, 5, 6]   (combined ext.)      → 1.3006
#    so [1, 2, 3, 4, 5, 6] tests whether both extensions add when combined.
#    Expected runtime ~80 min (5 layers = 65 min, +3 min/layer).
#
# 2) VALIDATE: 3-seed validation of the locked-in 4-layer winner [2, 3, 4, 5].
#    Seeds 42 and 123 — seed 1337 is already done in 5d-B (1.3010).
#    Same seed grid as abl4 (parallel residuals multi-seed) for apples-to-apples.
#    Expected runtime ~51 min/run × 2 = ~1h 45m.
#
# Total wall-clock estimate: ~3h 25m (well within the 6h SBATCH allocation).

# -----------------------------
# STEP 1 — 6-layer probe: [1, 2, 3, 4, 5, 6] target=both, seed 1337
# -----------------------------
RUN_ID=ablation5f_layers123456_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="1,2,3,4,5,6" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py

# -----------------------------
# STEP 2 — 3-seed validation of [2, 3, 4, 5] target=both on seeds 42 and 123
# -----------------------------
RUN_ID=ablation5f_layers2345_multiseed \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="2,3,4,5" \
RECUR_TIMES=1 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
