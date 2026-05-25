#!/bin/bash
#SBATCH --job-name=combined_mixed_qat_L11_2345_freshseeds
#SBATCH --output=/home/3199937/slurm_logs/combined_mixed_qat_L11_2345_freshseeds_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_mixed_qat_L11_2345_freshseeds_%j.err
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

# Fresh-seed validation of [2,3,4,5] at L=11 — does the friend's lucky seed-123 (1.2773)
# generalise to new seeds?
#
# Friend's 3-seed [2,3,4,5] mean: 1.2799 ± 0.0026 (seeds 42/123/1337 = 1.2799/1.2773/1.2825)
# Our 3-seed [3,4,5] mean       : 1.2811 ± 0.0004 (seeds 42/123/1337)
# Excluding seed 123, friend's [2,3,4,5] mean would be 1.2812 — essentially tied with [3,4,5].
#
# Three NEW seeds (7, 99, 999) for [2,3,4,5] to characterise the true distribution:
#   - If new 3-seed mean lands at ~1.281 → seed 123 was an outlier; [3,4,5] wins on every axis.
#   - If new 3-seed mean lands at ~1.279 → [2,3,4,5] has a real bpb edge; trade artifact size for it.
#   - In between (~1.280) → still ambiguous, but more seeds tightens the picture.
#
# Same config as friend's good 3-seed runs (verified via wandb config metadata):
# train_gpt_combined_qat_emb.py + brotli + protected int8 embedding + QAT @ 25%.
#
# RUN_ID prefix is distinct (`..._freshseeds_Antonio`) so these don't get mixed with
# the friend's existing 3 runs on wandb. For analysis, pull both prefixes separately.
#
# Runtime: 3 × ~70 min ≈ 3h30m. SBATCH 5h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=combined_mixed_qat_L11_2345_freshseeds_Antonio \
NUM_LAYERS=11 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
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
SEEDS=7,99,999 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb.py

# Best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
