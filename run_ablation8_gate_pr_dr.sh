#!/bin/bash
#SBATCH --job-name=ablation8_gate_pr_dr
#SBATCH --output=/home/3199937/slurm_logs/ablation8_gate_pr_dr_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation8_gate_pr_dr_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antonio.honsell@studbocconi.it

mkdir -p /home/3199937/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

# Ablation 8 — Attention Gate × Parallel Residuals × Depth Recurrence (triple stack).
#
# The headline experiment of the project. Composes the three multi-seed-validated
# standalone winners and tests whether three orthogonal mechanisms compose
# nearly-linearly the way PR × DR did (92.7% efficient at 3-seed precision).
#
# Per-mechanism standalone (3-seed) gains over baseline 1.3101 ± 0.0013:
#   - PR alone   (abl4)  : -0.0044  (1.3057 ± 0.0016)
#   - DR alone   (abl5f) : -0.0079  (1.3022 ± 0.0029)
#   - Gate w=8   (abl7d) : -0.0059  (1.3042 ± 0.0004)     <- new winner
#   - Gate w=12  (abl7b) : -0.0043  (1.3058 ± 0.0008)     <- leaderboard default
#   - PR × DR    (abl6)  : -0.0114  (1.2987 ± 0.0028)
#
# Headline-likelihood-weighted predictions for the triple stack:
#   - pure additive             : -0.0173  → 1.2928 (w=8) / 1.2944 (w=12)
#   - PR×DR's 92.7% efficiency  : ~-0.016  → 1.2940 (w=8) / 1.2956 (w=12)
#   - 70% efficient             : ~-0.012  → 1.2980 (w=8) / 1.2987 (w=12)
#   - no extra gate gain        : -0.0114  → 1.2987 (the abl6 reference)
#
# Two runs at seed 1337:
#   1. width=8  — abl7d multi-seed winner
#   2. width=12 — leaderboard default; sanity check that w=8 also wins in composition
#                 (the standalone gap was 2.58σ; in composition it could shrink)
#
# Runtime: ~70 min/run × 2 ≈ 2h30m. SBATCH time set to 5h for margin (the triple
# stack does more compute per step than any single component, and the abl6 inductor
# workaround is already inherited).
#
# Decision tree after this run:
#   - lowest of the two < 1.2970 → run abl8b (multi-seed validation, 2 more seeds)
#                                  on the winning width.
#   - both >= 1.2970 → stop. The gate doesn't compose; headline stays at PR × DR.

# Compute nodes on this cluster have been intermittently unable to reach api.wandb.ai
# (see abl7d_w8 failures). WANDB_MODE=offline writes runs to ./wandb/offline-run-*
# and never phones home. After the job finishes, sync from the LOGIN node with:
#   wandb sync wandb/offline-run-*
# Quick network diagnostic so the error log tells us if outbound HTTPS is the issue.
echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# =============================================================================
# RUN 1 — Triple stack with width=8 (abl7d standalone winner)
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8_gate_pr_dr_w8 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=8 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# =============================================================================
# RUN 2 — Triple stack with width=12 (leaderboard default)
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8_gate_pr_dr_w12 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py
