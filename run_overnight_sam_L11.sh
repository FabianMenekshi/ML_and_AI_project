#!/bin/bash
#SBATCH --job-name=overnight_sam_L11
#SBATCH --output=/home/3199937/slurm_logs/overnight_sam_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/overnight_sam_L11_%j.err
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

# Overnight experiment 3/3: Sharpness-Aware Minimization (SAM) during QAT, at L=11.
#
# Hypothesis:
#   SAM explicitly seeks flat minima by taking an inner gradient *ascent* step before
#   each optimizer update. The model is rewarded for having low loss in a neighbourhood
#   around the current weights, not just at a point. Flat minima are empirically
#   robust to quantisation rounding, so SAM-during-QAT should give the cleanest
#   theoretical motivation for a post-quant BPB improvement.
#
# Implementation:
#   step 0 .. 1249  : normal 1-pass training step (identical to baseline)
#   step 1250 ..    : SAM step:
#                       Phase 1: forward+backward at θ → get gradient g
#                       Phase 2: perturb θ ← θ + ρ · g / ||g||
#                       Phase 3: forward+backward at perturbed θ → get gradient g'
#                       Phase 4: restore θ ← θ - ρ · g / ||g||
#                       Phase 5: optimizer.step() using g'
#   Phase 3 replays the cached micro-batches from Phase 1 so the gradient is
#   computed against the same data points.
#
# Cost: SAM doubles the forward+backward count during the QAT phase. QAT is 75%
#       of training, so total work is ≈ 5000 + 3750 = 8750 step-equivalents,
#       i.e. about 75% slower than baseline.
#
# Architecture: same L=11 final stack.
# Baseline for head-to-head: the 5-seed L=11 mean of 1.2812 ± 0.0007 BPB at 15.36 MB.
# Target effect: -0.002 to -0.005 BPB (largest expected gain of the three overnight runs).
#
# 2 seeds (not 3) due to the higher per-run cost (~2 hours each).
# Runtime: 2 × ~120 min ≈ 4h. SBATCH 7h budget for margin.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=overnight_sam_L11_recur345 \
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
SAM_RHO=0.05 \
SAM_ADAPTIVE=0 \
SEEDS=42,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_sam.py

# Best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
