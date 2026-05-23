#!/bin/bash
#SBATCH --job-name=ablation8_headline_3seed
#SBATCH --output=/home/3199937/slurm_logs/ablation8_headline_3seed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation8_headline_3seed_%j.err
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

# Ablation 8 — HEADLINE 3-SEED REPRODUCIBILITY RUN.
#
# Single-shot reproduction of the project headline:
#   Triple stack (Gate × PR × DR) at width=12, 3 seeds → 1.2927 ± 0.0007.
#
# All three seeds (42, 123, 1337) run in ONE python invocation so torch.compile
# is shared across them. Each seed trains the same model with the same protocol;
# the only thing that varies is the RNG seed.
#
# Configuration (every multi-seed-validated winner from the prior ablations):
#   - Parallel residuals (abl4):    psl=4, sym init           → 1.3057 ± 0.0016
#   - Depth recurrence  (abl5f):    recur=[2,3,4,5] both     → 1.3022 ± 0.0029
#   - Attention gate    (abl7b):    src=proj, width=12        → 1.3058 ± 0.0008
#   → Composed (abl8b, this script): Triple w=12              → 1.2927 ± 0.0007
#
# RUN_ID prefix is DISTINCT from abl8/abl8b (`ablation8_headline_3seed_w12`) so
# re-running this script does NOT pollute the existing wandb runs that the
# notebook references. If you want this script's results to feed the notebook
# instead, change the analysis regex in the abl8b section to also match
# `^ablation8_headline_3seed`, OR just inspect the runs directly on wandb.
#
# Runtime: 3 × ~70 min ≈ 3h30m. SBATCH 5h budget for margin.
# WANDB_MODE=offline (compute-node network has been intermittent — see abl7d/abl8 history).
# Sync from the LOGIN node after: wandb sync wandb/offline-run-*

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# =============================================================================
# Headline triple stack: Gate × PR × DR at width=12, 3 seeds in one invocation
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8_headline_3seed_w12 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# Best-effort sync from the compute node; tolerate failure.
echo "=== attempting wandb sync from compute node ==="
wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync failed; run 'wandb sync wandb/offline-run-*' from the login node"
