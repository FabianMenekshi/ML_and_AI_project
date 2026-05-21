#!/bin/bash
#SBATCH --job-name=ablation7d_w8_multiseed
#SBATCH --output=/home/3199937/slurm_logs/ablation7d_w8_multiseed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation7d_w8_multiseed_%j.err
#SBATCH --time=03:00:00
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

# Ablation 7d — multi-seed validation of GATE_WIDTH=8.
#
# Why this run:
#   abl7c found w=8 at seed 1337 = 1.3040, below w=12's 3-seed mean (1.3058 ± 0.0008)
#   by 0.0018 bpb — ~2× the std, borderline-suggestive that w=8 may genuinely beat
#   the leaderboard's w=12 at this scale. Need seeds 42 and 123 to decide:
#     - if 3-seed w=8 mean is meaningfully below 1.3058 → new winner, leaderboard
#       choice is not scale-optimal for our 9-layer 512-dim model;
#     - if it regresses to ~1.3058 → seed-1337 was lucky, settle on w=12.
#
# Same protocol as abl7b / every other multi-seed validation in the project:
#   ITERATIONS=5000, WARMDOWN_ITERS=750, TRAIN_BATCH_TOKENS=131072, VAL_LOSS_EVERY=500.
# Runtime: 2 × ~50 min ≈ 1h45m. SBATCH time budgeted at 3h for margin.
#
# RUN_ID choice: prefixed `ablation7c_attn_gate_proj_w8` so the new seed-42/123
# runs share a wandb regex with the existing seed-1337 run from abl7c — the notebook
# picks up the full 3-seed grid via ^ablation7c_attn_gate_proj_w8.

# Quick diagnostic so the error log tells us whether outbound HTTPS is the issue.
echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# WANDB_MODE=offline writes runs to ./wandb/offline-run-* and never phones home.
# After the job finishes, sync them from the login node with:
#   wandb sync wandb/offline-run-*
WANDB_MODE=offline \
RUN_ID=ablation7c_attn_gate_proj_w8_multiseed \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=8 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py
