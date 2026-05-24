#!/bin/bash
#SBATCH --job-name=combined_dr_ablation_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_dr_ablation_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_dr_ablation_L11_%j.err
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

# DR (depth recurrence) ablation at L=11.
#
# The 9L winner was recur_layers=[2,3,4,5] target=both — chosen because it straddled
# the U-Net hinge (between layers 3 and 4 at 9L) with 2 encoder + 2 decoder layers
# (abl5d/5f, "straddle the hinge + extend encoder-side").
#
# At 11L the hinge moves: encoder=5 (0-4), decoder=6 (5-10), hinge between 4 and 5.
# The 9L config [2,3,4,5] now leans 3-enc + 1-dec relative to the new hinge — no longer
# straddles. This sweep tests four candidates for the new optimum:
#
#   recur=[2,3,4,5]     baseline — 9L winner, current default in friend's bash
#   recur=[3,4,5,6]     straddles the NEW hinge (2 enc + 2 dec around layers 4/5)
#   recur=[2,3,4,5,6]   encoder-extension at 11L (preserves 9L winning principle, 5 layers)
#   recur=[3,4,5]       lighter recurrence, matches leaderboard PR #1667's 3-layer DR
#
# Single seed (1337) — same protocol as every other "first sweep then validate" pattern
# in this project (abl3→abl4, abl7→abl7b). If one config clearly wins, multi-seed it.
#
# All four runs use the friend's mixed-quant + QAT + brotli + protected-embedding
# stack so the comparison is apples-to-apples vs run_combined_mixed_qat_L11_emb_Antonio.sh.
#
# Runtime: 4 × ~70 min ≈ 4h40m. SBATCH 7h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Shared protocol overrides (same as the verification run)
export WANDB_MODE=offline
export NUM_LAYERS=11
export PARALLEL_START_LAYER=4
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
export SEEDS=1337
export ITERATIONS=5000
export MAX_WALLCLOCK_SECONDS=0
export WARMDOWN_ITERS=750
export TRAIN_BATCH_TOKENS=131072
export VAL_LOSS_EVERY=500

# =============================================================================
# Probe A — recur=[2,3,4,5]  (9L winner, currently used at 11L; the "baseline" of this sweep)
# =============================================================================
RUN_ID=combined_dr_ablation_L11_recur2_3_4_5 \
RECUR_LAYERS="2,3,4,5" \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe B — recur=[3,4,5,6]  (straddles the NEW hinge at 11L: 2 enc + 2 dec around 4/5)
# =============================================================================
RUN_ID=combined_dr_ablation_L11_recur3_4_5_6 \
RECUR_LAYERS="3,4,5,6" \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe C — recur=[2,3,4,5,6]  (encoder-extension at 11L, preserves 9L principle, 5 layers)
# =============================================================================
RUN_ID=combined_dr_ablation_L11_recur2_3_4_5_6 \
RECUR_LAYERS="2,3,4,5,6" \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe D — recur=[3,4,5]  (lighter, matches leaderboard PR #1667 3-layer DR)
# =============================================================================
RUN_ID=combined_dr_ablation_L11_recur3_4_5 \
RECUR_LAYERS="3,4,5" \
python3 train_gpt_combined_qat_emb.py

# Quick best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
