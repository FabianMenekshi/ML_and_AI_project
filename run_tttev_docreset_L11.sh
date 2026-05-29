#!/bin/bash
#SBATCH --job-name=tttev_docreset_L11
#SBATCH --output=/home/3199937/slurm_logs/tttev_docreset_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/tttev_docreset_L11_%j.err
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

# NOVEL ABLATION: per-document soft-reset for score-first TTT-eval.
#
# Hypothesis: FineWeb concatenates documents from very different distributions
# (Wikipedia, code, news, fiction, recipes, etc.). The standard "continuous-drift"
# TTT pattern used by every leaderboard entry lets adaptation to document A
# actively hurt scoring of document B. A soft-reset at SentencePiece document
# boundaries — blending the current adapted weights with the original frozen
# weights by α ∈ [0, 1] — could give each document its own clean adaptation
# budget.
#
# Sweep: TTT_EVAL_RESET_ALPHAS = "0.0,0.25,0.5,0.75,1.0"
#   α = 0.0   → no reset           (reproduces our standard tttev result, 1.2765 BPB)
#   α = 0.25  → mostly-keep        (gentle correction toward original at boundaries)
#   α = 0.5   → half-and-half      (Polyak blend at boundaries)
#   α = 0.75  → mostly-reset       (most prior-document adaptation is forgotten)
#   α = 1.0   → hard reset         (each document starts adaptation from scratch)
#
# Efficiency: training is identical across all α (depends only on seed); we run
# ONE training pass, then 5 TTT-eval passes back-to-back from the same fresh
# dequantised state. Per-α the optimizer is rebuilt from scratch and the model
# is restored to the post-quantisation roundtrip checkpoint.
#
# Single seed (1337). Within-effect noise of TTT-eval was ~0.0001 across our
# 3-seed tttev run, so multi-seed isn't where the budget should go — the alpha
# axis is what we're exploring.
#
# Per-α TTT-eval cost: ~17 min (same as our standard tttev). 5 αs ≈ 85 min.
# Plus training ~78 min. Total ~2h40m. SBATCH 4h budget for margin.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=tttev_docreset_L11_recur345_alphasweep \
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
TTT_EVAL_RESET_ALPHAS="0.0,0.25,0.5,0.75,1.0" \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb_tttev_docreset.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
