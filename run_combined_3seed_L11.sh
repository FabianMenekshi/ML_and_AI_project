#!/bin/bash
#SBATCH --job-name=combined_3seed_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_3seed_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_3seed_L11_%j.err
#SBATCH --time=08:00:00
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

# COMBINED 3-seed run, NUM_LAYERS=11.
#
# Architecture (project triple-stack winner, all multi-seed-validated at 9 layers):
#   - PR : psl=4, sym init                              (abl4)
#   - DR : recur=[2,3,4,5] target=both                  (abl5f)
#   - Gate : src=proj, width=12                         (abl7b)
# Compression (friend's work):
#   - GPTQ + LQER, int6 matrices / int7 embeddings, LQER rank=4 top-3, group=32.
#
# Extra layers (10th + 11th): extend the parallel-no-recurrence region. PSL and
# recur_layers unchanged from the 9-layer headline; the experiment isolates
# "additional depth" as the only varying axis.
#
# Note: at 11 layers, num_encoder = 5, num_decoder = 6 (skip-connections are
# capped at min(encoder, decoder) = 5, so layer 10 is skip-free in the decoder).
# This is consistent with how the baseline GPT class handles asymmetric depth.
#
# Runtime: 3 × ~80 min ≈ 4h (slightly longer than L=10 due to extra layer's
# forward + GPTQ Hessian collection). SBATCH 8h for margin.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=combined_pr_dr_gate_gptqlqer_L11 \
NUM_LAYERS=11 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
USE_GPTQ_LQER=1 \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=7 \
LQER_RANK=4 \
LQER_TOP_K=3 \
QUANT_GROUP_SIZE=32 \
QUANT_ASYMMETRIC=1 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined.py

# Best-effort sync from the compute node; tolerate failure.
echo "=== attempting wandb sync from compute node ==="
wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync failed; run 'wandb sync wandb/offline-run-*' from the login node"
