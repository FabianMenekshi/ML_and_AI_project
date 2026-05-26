#!/bin/bash
#SBATCH --job-name=overnight_sophia_L11
#SBATCH --output=/home/3199937/slurm_logs/overnight_sophia_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/overnight_sophia_L11_%j.err
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

# Overnight experiment 2/3: SophiaG replaces Muon for matrix params during QAT, at L=11.
#
# Hypothesis:
#   The QAT phase is where weights need to find a "quant-robust" location in the loss
#   landscape — flat enough that rounding noise from int6/int8 quantisation doesn't
#   degrade the loss. SophiaG (Liu et al. 2023) uses an element-wise clip of m/h
#   (first-moment / diagonal-Hessian proxy) which provably promotes flatter minima
#   compared to first-order optimisers like Muon.
#
# Implementation:
#   This variant uses g^2 directly as the diagonal Fisher proxy (Bartlett's identity
#   says diagonal Fisher ∝ GNB Hessian estimate). This keeps the training loop
#   identical to baseline — no extra forward/backward passes needed.
#
# Mechanism timeline within a single run:
#   step 0 .. 1249  : Muon optimises matrix params (identical to baseline)
#   step 1250 ..    : QAT activates AND Muon → SophiaG swap happens simultaneously
#                     for the matrix-param group only. Adam still drives embeddings,
#                     scalars, and lm_head.
#
# Architecture: same L=11 final stack as the baseline experiment.
# Baseline for head-to-head: the 5-seed L=11 mean of 1.2812 ± 0.0007 BPB at 15.36 MB.
# Target effect: -0.001 to -0.004 BPB.
# Risk: SophiaG with default hyperparameters could destabilise post-QAT learning
#       (the m and h buffers start at zero, and the lr is different from Muon's).
#
# 3 seeds in one python invocation to share torch.compile across seeds.
# Runtime: 3 × ~70 min ≈ 3h30m. SBATCH 5h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=overnight_sophia_L11_recur345 \
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
SOPHIA_LR=3e-4 \
SOPHIA_RHO=0.04 \
SOPHIA_BETA1=0.965 \
SOPHIA_BETA2=0.99 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_sophia.py

# Best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
