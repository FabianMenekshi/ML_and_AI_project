#!/bin/bash
#SBATCH --job-name=distill_L11_inbudget
#SBATCH --output=/home/3199937/slurm_logs/distill_L11_inbudget_%j.out
#SBATCH --error=/home/3199937/slurm_logs/distill_L11_inbudget_%j.err
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

# In-budget post-training self-distillation on the L=11 final stack.
#
# Hypothesis: the previous distillation experiment (5000 main + 200 distill = 5200
# total iters) exceeded our adapted ``5000-iteration'' budget by 4%. By reducing
# the main training to 4800 iterations and using the saved 200 iterations for
# distillation, the total stays at 5000 and the technique becomes fully in-budget.
#
# Schedule (auto-derived from ITERATIONS=4800 and WARMDOWN_ITERS=750):
#   - Pre-QAT phase   : iters 0--1199   (1200 iters,  was 1250)
#   - QAT flat-LR     : iters 1200--4049 (2850 iters, was 3000) <- the 200 cut comes from here
#   - Warmdown        : iters 4050--4799 (750 iters,  UNCHANGED)
#   - Distillation    : 200 extra steps post-training (AdamW lr=1e-4, T=1.0)
# Total budget: 5000 main + 0 = strictly in-budget under our adapted rule.
#
# The critical question: does distillation recover the 200 lost main-training iters?
# Three plausible outcomes:
#   (A) result lands at ~1.2793 BPB  -> distillation fully compensates; new headline
#   (B) result lands at ~1.2800-1.2805 BPB -> partial compensation; net positive
#   (C) result lands at ~1.2812 BPB  -> no net gain; cut iters were too valuable
#
# 3 seeds in one Python invocation to share torch.compile across seeds.
# Per-seed runtime ~74 min (70 main + 4 distill). Total ~3h45m. SBATCH 5h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=distill_L11_inbudget_main4800_distill200 \
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
ITERATIONS=4800 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_distill.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
