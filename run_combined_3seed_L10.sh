#!/bin/bash
#SBATCH --job-name=combined_3seed_L10
#SBATCH --output=/home/3199937/slurm_logs/combined_3seed_L10_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_3seed_L10_%j.err
#SBATCH --time=07:00:00
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

# COMBINED 3-seed run, NUM_LAYERS=10.
#
# Architecture (project triple-stack winner, all multi-seed-validated at 9 layers):
#   - PR : psl=4, sym init                              (abl4)
#   - DR : recur=[2,3,4,5] target=both                  (abl5f)
#   - Gate : src=proj, width=12                         (abl7b — leaderboard width;
#                                                        wins in composition at 9L)
# Compression (friend's work):
#   - GPTQ + LQER, int6 for attn/MLP matrices, int7 for embeddings,
#     LQER rank=4 on top-3 most sensitive layers, group_size=32, asymmetric.
#
# Extra layer (10th): extends the parallel-no-recurrence region (psl=4 unchanged,
# recur_layers unchanged), keeping the experiment a clean "depth vs no-depth"
# comparison against the 9-layer headline (1.2927 ± 0.0007 INT8, val_bpb only —
# this run reports val_bpb under GPTQ+LQER which may differ slightly).
#
# All 3 seeds (42, 123, 1337) run in ONE python invocation so torch.compile is
# shared across them. Total runtime: 3 × ~75 min ≈ 3h45m (slightly more than
# 9-layer because GPTQ+LQER's Hessian collection adds ~3-5 min per seed).
# SBATCH 7h for margin.
#
# WANDB_MODE=offline (compute-node network history). Sync from login node after:
#   wandb sync wandb/offline-run-*

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=combined_pr_dr_gate_gptqlqer_L10 \
NUM_LAYERS=10 \
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
