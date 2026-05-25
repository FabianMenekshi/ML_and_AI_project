#!/bin/bash
#SBATCH --job-name=combined_overnight_psl8_validation_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_overnight_psl8_validation_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_overnight_psl8_validation_L11_%j.err
#SBATCH --time=09:00:00
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

# Overnight bundle — psl=8 validation + 2D interaction probe + baseline strengthening.
#
# Three motivations:
#   (1) Multi-seed validate psl=8 (the single-seed PR ablation winner at L=11).
#   (2) Check if the optimal DR position shifts when psl moves from 4 → 8
#       (1D-then-1D ablation chains can miss joint optima).
#   (3) Add 2 more seeds to the psl=4+[3,4,5] baseline (currently 3 seeds at
#       1.2811 ± 0.0004) for a tighter head-to-head comparison.
#
# Total: 4 python invocations, 6 training runs, ~6h50m. SBATCH 9h budget for margin.
#
# Layout after this job finishes:
#   psl=8 + recur=[3,4,5]   : 3 seeds (1337 + new 42, 123) → multi-seed mean
#   psl=8 + recur=[3,4,5,6] : 1 seed (1337) → 2D probe vs psl=4+[3,4,5,6]
#   psl=8 + recur=[4,5,6]   : 1 seed (1337) → DR-shift hypothesis at psl=8
#   psl=4 + recur=[3,4,5]   : 5 seeds (existing 42/123/1337 + new 7/99) → tighter baseline
#
# RUN_ID conventions:
#   - psl=8 runs get unique prefixes so they're identifiable on wandb.
#   - psl=4 + [3,4,5] extra seeds share the existing ^combined_dr_ablation_L11_recur3_4_5
#     regex so the notebook automatically picks all 5 seeds together.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Shared protocol — same as friend's mixed-quant + QAT + brotli + protected-embedding stack
export WANDB_MODE=offline
export NUM_LAYERS=11
export PARALLEL_ASYM_INIT=0
export RECUR_TARGET=both
export RECUR_TIMES=1
export GATE_ATTN_OUT=1
export GATE_ATTN_SRC=proj
export GATE_WIDTH=12
export MATRIX_QUANT_BITS=6
export RECUR_QUANT_BITS=8
export EMBED_QUANT_MODE=8
export COMPRESSION_METHOD=brotli
export BROTLI_QUALITY=11
export QAT_ENABLED=1
export QAT_START_FRACTION=0.25
export QAT_EMBED_BITS=8
export ITERATIONS=5000
export MAX_WALLCLOCK_SECONDS=0
export WARMDOWN_ITERS=750
export TRAIN_BATCH_TOKENS=131072
export VAL_LOSS_EVERY=500

# =============================================================================
# 1) Multi-seed validate psl=8 + recur=[3,4,5] — the headline-deciding test
#    (~2h20m, 2 seeds in one python invocation to share torch.compile)
# =============================================================================
RUN_ID=combined_overnight_psl8_recur345_multiseed_L11 \
PARALLEL_START_LAYER=8 \
RECUR_LAYERS="3,4,5" \
SEEDS=42,123 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# 2) psl=8 + recur=[3,4,5,6] @ seed 1337 — does DR optimum shift at psl=8?
#    (compare against psl=4 + [3,4,5,6] which scored 1.2829 in the DR ablation)
# =============================================================================
RUN_ID=combined_overnight_psl8_recur3456_L11 \
PARALLEL_START_LAYER=8 \
RECUR_LAYERS="3,4,5,6" \
SEEDS=1337 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# 3) psl=8 + recur=[4,5,6] @ seed 1337 — alternative DR position at psl=8
#    (shifted 1 layer later — fully in the parallel-mode region)
# =============================================================================
RUN_ID=combined_overnight_psl8_recur456_L11 \
PARALLEL_START_LAYER=8 \
RECUR_LAYERS="4,5,6" \
SEEDS=1337 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# 4) psl=4 + recur=[3,4,5] @ seeds 7, 99 — tighten the baseline CI to 5 seeds total
#    (RUN_ID prefix matches existing ^combined_dr_ablation_L11_recur3_4_5 so the
#     notebook picks up all 5 seeds in one query)
# =============================================================================
RUN_ID=combined_dr_ablation_L11_recur3_4_5_extra_seeds_overnight \
PARALLEL_START_LAYER=4 \
RECUR_LAYERS="3,4,5" \
SEEDS=7,99 \
python3 train_gpt_combined_qat_emb.py

# Quick best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
