#!/bin/bash
#SBATCH --job-name=tttev_L11
#SBATCH --output=/home/3199937/slurm_logs/tttev_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/tttev_L11_%j.err
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

# Score-first TTT-eval on top of the L=11 final stack.
#
# This script does TWO evaluations per seed:
#   1. STANDARD post-quant eval  → reports final_val_bpb         (the baseline metric)
#   2. SCORE-FIRST TTT-EVAL      → reports final_val_bpb_tttev   (the new metric)
#
# TTT-eval mechanism (Sun 2020 / parameter-golf leaderboard style):
#   For each batch of validation tokens:
#     a. SCORE   — forward(x,y) under no_grad; this loss counts toward BPB.
#     b. UPDATE  — forward + backward + AdamW.step() on the same batch.
#     c. Move on (causally — the next batch is scored under the updated weights).
#
# The base 16 MB artifact is unchanged. The gradient updates are transient.
#
# Configuration:
#   Architecture: identical to the L=11 final stack (PR psl=4 + DR [3,4,5] + Gate proj/w12).
#   Quantisation: identical (mixed INT6/INT8, QAT @ 25%, protected INT8 embedding).
#   TTT-eval hyperparameters (conservative starting point):
#     TTT_EVAL_LR=1e-4         (small enough that updates don't destabilise quickly)
#     TTT_EVAL_STEPS=1         (one gradient step per validation batch)
#     TTT_EVAL_OPTIMIZER=adamw (defensive choice; SGD is an alternative)
#
# Baseline (head-to-head): L=11 5-seed mean = 1.2812 ± 0.0006 BPB at 15.36 MB.
# Target: tttev BPB meaningfully below the standard final_val_bpb on the same run.
#         Leaderboard top entries get ~0.2 BPB from TTT at larger scales.
#         At our scale, anything below -0.05 BPB is a real win.
#
# 3 seeds in one Python invocation to share torch.compile across seeds.
# Per-seed runtime ≈ standard 70 min + ~3 min for TTT-eval (~5 min if more steps).
# Total ≈ 3 × 75 min ≈ 3h45m. SBATCH 6h budget for margin.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=tttev_L11_recur345_lr1e4_steps1 \
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
TTT_EVAL_ENABLED=1 \
TTT_EVAL_LR=1e-4 \
TTT_EVAL_STEPS=1 \
TTT_EVAL_OPTIMIZER=adamw \
TTT_EVAL_RESET=1 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_tttev.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
