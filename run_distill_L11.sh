#!/bin/bash
#SBATCH --job-name=distill_L11
#SBATCH --output=/home/3199937/slurm_logs/distill_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/distill_L11_%j.err
#SBATCH --time=06:00:00
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

# Post-training self-distillation on top of the L=11 final stack.
#
# Idea: between the end of standard training and the quantisation step, run a
# short self-distillation phase where the just-trained model serves as TEACHER
# (frozen, QAT off in its forward) and the STUDENT (a copy of the same weights,
# with QAT fake-quant active) is fine-tuned to match the teacher's logits via
# KL divergence. This pushes the student toward weights that, when quantised,
# still match the un-quantised teacher's predictions.
#
# The quantisation function is UNCHANGED — the only addition is the short
# distillation phase before quantisation.
#
# Configuration:
#   Architecture: identical to the L=11 final stack (PR psl=4 + DR [3,4,5] +
#                 Gate proj/w12).
#   Quantisation: identical (mixed INT6/INT8, QAT @ 25%, protected INT8 embedding).
#   Distillation hyperparameters (conservative starting point):
#     DISTILL_STEPS=200          (cheap — ~3-4 min per seed)
#     DISTILL_LR=1e-4            (small enough to refine, not overwrite)
#     DISTILL_TEMPERATURE=1.0    (standard; no softening of teacher logits)
#     DISTILL_OPTIMIZER=adamw
#
# Baseline (head-to-head): L=11 5-seed mean = 1.2812 ± 0.0006 BPB at 15.36 MB.
# Target: distilled BPB meaningfully lower than 1.2812. Realistically -0.001
#         to -0.003 BPB given paper precedent.
#
# 3 seeds in one Python invocation. Per-seed cost ≈ standard 70 min + ~4 min
# for distillation. Total ≈ 3 × 74 min ≈ 3h45m. SBATCH 6h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=distill_L11_recur345_steps200_lr1e4 \
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
DISTILL_ENABLED=1 \
DISTILL_STEPS=200 \
DISTILL_LR=1e-4 \
DISTILL_TEMPERATURE=1.0 \
DISTILL_OPTIMIZER=adamw \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_distill.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
