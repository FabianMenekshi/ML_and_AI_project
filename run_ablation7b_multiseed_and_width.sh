#!/bin/bash
#SBATCH --job-name=ablation7b_multiseed_and_width
#SBATCH --output=/home/3199937/slurm_logs/ablation7b_multiseed_and_width_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation7b_multiseed_and_width_%j.err
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

# Ablation 7b — finalise the attention-output-gate result.
#
# Two purposes (mirrors how abl6b combined a multi-seed validation with synergy probes):
#
# OPTION A — Multi-seed validate the abl7 winner (the headline experiment)
#   Config: GATE_ATTN_OUT=1, GATE_ATTN_SRC=proj, GATE_WIDTH=12
#   Seeds 42, 123 (seed 1337 already done in abl7 → 1.3052 bpb).
#   Yields the 3-seed mean ± std for the headline result, comparable to:
#     - PR alone (abl4) : 1.3057 ± 0.0016
#     - DR alone (abl5f): 1.3022 ± 0.0029
#     - PR × DR (abl6)  : 1.2987 ± 0.0028
#   Runtime: 2 × ~50 min ≈ 1h45m.
#
# OPTION B — Width fine-tuning (single-seed)
#   Same src=proj, but widths between the underperformer (w=6) and the leaderboard
#   default (w=12). Tests whether the sweet spot is genuinely at 12 or could be
#   smaller for cheaper.
#     - GATE_WIDTH=8  → ~70% of w=12's params
#     - GATE_WIDTH=10 → ~83% of w=12's params
#   Single seed (1337). Runtime: 2 × ~50 min ≈ 1h45m.
#
# Total wall-clock: ~3h30m on a single GPU. SBATCH time set to 6h for margin.
#
# NOTE: w=24 multi-seed deliberately skipped — single-seed gap to w=12 was 0.0002 bpb
# (well inside DR's 0.0029 noise floor). Following the abl5g "use the cheaper config
# when tied" convention. Revisit only if abl7b confirms the gate is real and we want
# to know whether wider buys anything beyond noise.

# =============================================================================
# OPTION A — multi-seed validation: GATE_WIDTH=12, GATE_ATTN_SRC=proj, seeds 42,123
# (seed 1337 already on wandb as ablation7_attn_gate_proj_w12_*; the notebook
#  picks it up alongside these two via the ^ablation7_attn_gate_proj_w12 regex.)
# =============================================================================
RUN_ID=ablation7_attn_gate_proj_w12_multiseed \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# OPTION B-1 — width fine-tuning: GATE_WIDTH=8, GATE_ATTN_SRC=proj, seed 1337
# =============================================================================
RUN_ID=ablation7c_attn_gate_proj_w8 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=8 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# OPTION B-2 — width fine-tuning: GATE_WIDTH=10, GATE_ATTN_SRC=proj, seed 1337
# =============================================================================
RUN_ID=ablation7c_attn_gate_proj_w10 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=10 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py
