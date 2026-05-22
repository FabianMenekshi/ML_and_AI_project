#!/bin/bash
#SBATCH --job-name=ablation8c_synergy_probes
#SBATCH --output=/home/3199937/slurm_logs/ablation8c_synergy_probes_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation8c_synergy_probes_%j.err
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

# Ablation 8c — Synergy probes for the triple stack (Gate × PR × DR).
#
# Mirrors abl6b's probe design (which localised PR×DR's synergy as "distributed
# across all recurred layers"). With the triple-stack headline now established at
# 1.2927 ± 0.0007 (Triple w=12, abl8b), we ask: where does the gate's contribution
# to the triple-stack synergy actually live?
#
# The triple-stack layer composition is:
#   Layer 0,1   : sequential, NO recurrence
#   Layer 2,3   : sequential, with DR recurrence
#   Layer 4,5   : parallel (psl=4 starts here), with DR recurrence
#   Layer 6,7,8 : parallel, NO recurrence
#
# Three single-seed probes (seed 1337), all with width=12 (matches the abl8b
# headline), all keeping the full PR + DR stack:
#
#   Probe A — gate on parallel-mode layers only (4-8, 5 layers):
#     Does the gate compose better with PR's two-lane region?
#   Probe B — gate on sequential-mode layers only (0-3, 4 layers):
#     Does the synergy live entirely in the pre-PR encoder?
#   Probe C — gate on recurrence layers only (2-5, 4 layers):
#     Does the gate help specifically on layers that get the extra DR pass?
#
# Reference points (from abl8b multi-seed):
#   - Triple w=12 all 9 layers (abl8b): 1.2927 ± 0.0007  (the headline)
#   - PR × DR  no gate          (abl6): 1.2987 ± 0.0028
#
# Interpretation framework (single-seed, so 0.001-bpb-scale precision):
#   - If probe ≈ 1.2927 → that layer-set carries all the gate's synergy.
#   - If probe ≈ 1.2987 → that layer-set contributes nothing on top of PR×DR.
#   - In between → partial contribution, mirrors abl6b's "distributed" result.
#
# Runtime: 3 × ~70 min ≈ 3h30m. SBATCH 5h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# =============================================================================
# PROBE A — Gate only on parallel-mode layers (4-8). Tests PR-region synergy.
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8c_probe_parallel \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
GATE_LAYERS="4,5,6,7,8" \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# =============================================================================
# PROBE B — Gate only on sequential-mode layers (0-3). Tests pre-PR encoder synergy.
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8c_probe_sequential \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
GATE_LAYERS="0,1,2,3" \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation8_gate_pr_dr.py

# =============================================================================
# PROBE C — Gate only on DR-recurrence layers (2-5). Tests recurrence-coupling synergy.
# =============================================================================
WANDB_MODE=offline \
RUN_ID=ablation8c_probe_recurrence \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
GATE_LAYERS="2,3,4,5" \
SEEDS=1337 \
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
