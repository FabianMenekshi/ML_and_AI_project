#!/bin/bash
#SBATCH --job-name=ablation7_attn_gate
#SBATCH --output=/home/3199937/slurm_logs/ablation7_attn_gate_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation7_attn_gate_%j.err
#SBATCH --time=07:00:00
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

# Ablation 7 — Attention Output Gate (per-head, sigmoid-gated).
#
# Ported from openai/parameter-golf PR #1667 (SmearGate + AttentionOutputGate, 2026-04-16).
# Mechanism: y = y * 2*sigmoid(W @ src[..., :gate_width]) before the output projection.
#   - W is zero-initialised, so the gate is the identity at step 0 (bit-identical to baseline).
#   - src ∈ {proj (raw block input x), q (Q-projection output, pre-RoPE)}.
#   - gate_width ∈ {6, 12, 24} controls how many input channels condition each head's gate.
#
# Single seed (1337). 5 runs total:
#   1. disabled control                       — sanity check vs baseline (1.3101 ± 0.0013)
#   2. src=proj, width=12   (leaderboard cfg) — the PR #1667 default
#   3. src=q,    width=12                     — source comparison
#   4. src=proj, width=6                      — narrower gate
#   5. src=proj, width=24                     — wider gate
#
# Expected runtime: ~45-55 min/run × 5 = ~4h on a single GPU. SBATCH time set to 7h for margin.
#
# Follow-up plan (separate sbatch, after seeing results):
#   - abl7b: multi-seed (seeds 42, 123) on the winning config, mirroring how abl3 → abl4 went.

# =============================================================================
# RUN 1 — disabled control (gate OFF). Verifies the abl7 script matches baseline.
# =============================================================================
RUN_ID=ablation7_attn_gate_disabled \
GATE_ATTN_OUT=0 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# RUN 2 — leaderboard default: src=proj, width=12 (PR #1667)
# =============================================================================
RUN_ID=ablation7_attn_gate_proj_w12 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# RUN 3 — source comparison: src=q, width=12 (gate conditioned on Q projection)
# =============================================================================
RUN_ID=ablation7_attn_gate_q_w12 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=q \
GATE_WIDTH=12 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# RUN 4 — narrower gate: src=proj, width=6
# =============================================================================
RUN_ID=ablation7_attn_gate_proj_w6 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=6 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py

# =============================================================================
# RUN 5 — wider gate: src=proj, width=24
# =============================================================================
RUN_ID=ablation7_attn_gate_proj_w24 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=24 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation7_attn_gate.py
