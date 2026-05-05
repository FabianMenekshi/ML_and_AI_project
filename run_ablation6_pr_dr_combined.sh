#!/bin/bash
#SBATCH --job-name=ablation6_pr_dr_combined
#SBATCH --output=/home/3199937/slurm_logs/ablation6_pr_dr_combined_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation6_pr_dr_combined_%j.err
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

# Ablation 6 — Parallel Residuals × Depth Recurrence (combined).
#
# Composes the two multi-seed-validated standalone winners:
#   - PR: psl=4, sym init       → 1.3057 ± 0.0016 (abl4, 3 seeds)
#   - DR: [2,3,4,5] target=both → 1.3022 ± 0.0029 (abl5f, 3 seeds)
#
# Expected outcomes:
#   linear (gains add)        : ~1.298  bpb
#   70% efficiency (interact) : ~1.300  bpb
#   sub-additive (overlap)    : ~1.302  bpb
#   super-additive (synergy)  : <1.298  bpb
#
# Strategy: single-seed exploratory run first (this script), then if it lands
# below ~1.302 (i.e. better than DR alone), launch the multi-seed validation.
# Estimated runtime: parallel + 4 recurred layers ≈ 60-65 min/run.

# -----------------------------
# STEP 1 — single-seed exploratory: psl=4 + recur=[2,3,4,5] target=both, seed 1337
# -----------------------------
RUN_ID=ablation6_pr_dr_combined_explore \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation6_pr_dr_combined.py
