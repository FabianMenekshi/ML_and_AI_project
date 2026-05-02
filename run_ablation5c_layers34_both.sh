#!/bin/bash
#SBATCH --job-name=ablation5c_layers34_both
#SBATCH --output=/home/3199937/slurm_logs/ablation5c_layers34_both_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5c_layers34_both_%j.err
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

# Ablation 5c — combine the two winners from ablation 5 and 5b.
#
# From abl5: best layer set was [3, 4] (1.3048 bpb)  vs  [4, 5] (1.3066 bpb).
# From abl5b: best target was "both" without resid_mix re-applied (1.3052 bpb on [4, 5]),
#             beating whole-block recurrence (with resid_mix) by ~0.0014 bpb.
#
# Option A: layers [3, 4] with target=both    (combine layer winner + target winner)
# Option B: layers [3, 4, 5] with target=both (extend to 3 layers — matches top
#                                              leaderboard entries that recur 3 layers)
#
# Both reuse train_gpt_ablation5b_recur_target.py (same script as ablation 5b).
# Two sequential python invocations so we can pin a different RECUR_LAYERS per option.

# -----------------------------
# Option A — layers [3, 4]
# -----------------------------
RUN_ID=ablation5c_layers34_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="3,4" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py

# -----------------------------
# Option B — layers [3, 4, 5]
# -----------------------------
RUN_ID=ablation5c_layers345_both \
RECUR_TARGET_VALUES="both" \
RECUR_LAYERS="3,4,5" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
