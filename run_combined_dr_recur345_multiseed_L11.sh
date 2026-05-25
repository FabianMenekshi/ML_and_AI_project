#!/bin/bash
#SBATCH --job-name=combined_dr_recur345_multiseed_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_dr_recur345_multiseed_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_dr_recur345_multiseed_L11_%j.err
#SBATCH --time=04:00:00
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

# Multi-seed validation of recur=[3,4,5] at L=11 (the single-seed winner of the DR ablation).
#
# Background: in the L=11 DR ablation, recur=[3,4,5] (3-layer recurrence) won on both
# axes at single seed (bpb=1.2813, size=15.36 MB — cheapest of all 4 probes).
# But friend's 3-seed [2,3,4,5] mean was 1.2799 ± 0.0026 (better than [3,4,5]'s
# single-seed of 1.2813), so the seed-1337 ablation might just have been an unlucky
# seed for [2,3,4,5] and a lucky one for [3,4,5]. Multi-seed validates which is true.
#
# Seed 1337 for [3,4,5] is already on wandb from the DR ablation, so this script runs
# the missing 2 seeds (42 and 123) in one python invocation. Combined with the seed-1337
# run, we'll have a full 3-seed mean ± std comparable to friend's [2,3,4,5] baseline.
#
# RUN_ID prefix is "combined_dr_ablation_L11_recur3_4_5" so the wandb regex
# ^combined_dr_ablation_L11_recur3_4_5 picks up the seed-1337 run + these new
# seed-42/123 runs together (the full 3-seed grid in one query).
#
# Runtime: 2 × ~70 min ≈ 2h20m. SBATCH 4h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=combined_dr_ablation_L11_recur3_4_5_multiseed \
NUM_LAYERS=11 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
MATRIX_QUANT_BITS=6 \
RECUR_QUANT_BITS=8 \
EMBED_QUANT_MODE=8 \
COMPRESSION_METHOD=brotli \
BROTLI_QUALITY=11 \
QAT_ENABLED=1 \
QAT_START_FRACTION=0.25 \
QAT_EMBED_BITS=8 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb.py

# Quick best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
