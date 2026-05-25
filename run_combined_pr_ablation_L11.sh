#!/bin/bash
#SBATCH --job-name=combined_pr_ablation_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_pr_ablation_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_pr_ablation_L11_%j.err
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

# PR (parallel residuals) ablation at L=11.
#
# psl=4 was the 9L winner (abl3b, validated at 1.3057 ± 0.0016) — at 9L it meant
# "entire decoder parallel" (layers 4–8, 5 blocks). At 11L, psl=4 now means "last
# encoder layer + entire decoder parallel" (layers 4–10, 7 blocks) — a more
# aggressive operating point than 9L's, and a different semantic.
#
# Top leaderboard L=11 entries are far MORE conservative on PR than our default:
#   - PR #1855 uses psl=8 (only last 3 layers parallel)
#   - PR #1667 uses psl=7-8 (last 2-3 layers parallel)
#
# Sweep keeping DR=[3,4,5] fixed (the new L=11 DR winner from prior ablation):
#
#   psl=4: 7 parallel layers (4–10)   current default, baseline of this sweep
#   psl=5: 6 parallel layers (5–10)   "entire decoder parallel" — preserves the 9L semantic
#   psl=7: 4 parallel layers (7–10)   moderate leaderboard-leaning shift
#   psl=8: 3 parallel layers (8–10)   aggressive — matches PR #1855
#
# Single seed (1337) — same "sweep first, validate after" pattern as abl3→abl4 and
# combined_dr_ablation_L11. All other knobs match the friend's mixed-quant + QAT +
# brotli + protected-embedding stack.
#
# Runtime: 4 × ~70 min ≈ 4h40m. SBATCH 7h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Shared protocol — applies to all 4 probes
export WANDB_MODE=offline
export NUM_LAYERS=11
export PARALLEL_ASYM_INIT=0
export RECUR_LAYERS="3,4,5"
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
# Probe A — psl=4 (current default; 7 parallel layers; the "baseline" of this sweep)
# =============================================================================
RUN_ID=combined_pr_ablation_L11_psl4 \
PARALLEL_START_LAYER=4 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe B — psl=5 (entire decoder parallel; 6 parallel layers; preserves 9L semantic)
# =============================================================================
RUN_ID=combined_pr_ablation_L11_psl5 \
PARALLEL_START_LAYER=5 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe C — psl=7 (4 parallel layers; moderate leaderboard-leaning shift)
# =============================================================================
RUN_ID=combined_pr_ablation_L11_psl7 \
PARALLEL_START_LAYER=7 \
python3 train_gpt_combined_qat_emb.py

# =============================================================================
# Probe D — psl=8 (3 parallel layers; aggressive leaderboard direction — matches PR #1855)
# =============================================================================
RUN_ID=combined_pr_ablation_L11_psl8 \
PARALLEL_START_LAYER=8 \
python3 train_gpt_combined_qat_emb.py

# Quick best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
