#!/bin/bash
#SBATCH --job-name=ablation8b_multiseed
#SBATCH --output=/home/3199937/slurm_logs/ablation8b_multiseed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation8b_multiseed_%j.err
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

# Ablation 8b — Multi-seed validation of the triple stack (Gate × PR × DR).
#
# Validates the project's HEADLINE experiment at 3-seed precision. Single-seed
# results from abl8 (seed 1337):
#   - Triple w=8  : 1.2928  (linear pred 1.2919, efficiency 95%)
#   - Triple w=12 : 1.2923  (linear pred 1.2935, efficiency 107%)
# Both single-seed numbers beat PR × DR (1.2987 ± 0.0028) by ~2σ. This run adds
# seeds 42 and 123 for BOTH widths so we can decide:
#   - whether the headline robustly lands at ~1.293
#   - whether the single-seed width flip (w=12 slightly beating w=8 in composition,
#     opposite of standalone) holds at 3-seed precision or was noise
#   - whether w=12's 107% "super-additive" composition efficiency is real
#
# 4 runs total, bundled into 2 python invocations (1 per width, both seeds in
# one SEEDS=42,123 call so torch.compile is reused). Each invocation handles 2
# seeds at ~70 min each ≈ 2h25m; total ~5h. SBATCH 7h budget for margin.
#
# WANDB_MODE=offline because compute-node connectivity to api.wandb.ai has been
# intermittent (abl7d, abl8). Sync from the LOGIN node after the job finishes:
#   wandb sync wandb/offline-run-*
#
# RUN_IDs prefixed `ablation8_gate_pr_dr_w{8,12}_multiseed` so the notebook regex
# ^ablation8_gate_pr_dr picks up the existing seed-1337 runs AND these new ones
# together, giving the full 3-seed grid in one query.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# =============================================================================
# RUN 1 — Triple stack with width=8, seeds 42,123 (single python invocation)
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8_gate_pr_dr_w8_multiseed \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=8 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# =============================================================================
# RUN 2 — Triple stack with width=12, seeds 42,123 (single python invocation)
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8_gate_pr_dr_w12_multiseed \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# Best-effort sync from the compute node; tolerate failure (network may be down).
echo "=== attempting wandb sync from compute node ==="
wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync failed; run 'wandb sync wandb/offline-run-*' from the login node"
