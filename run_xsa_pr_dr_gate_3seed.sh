#!/bin/bash
#SBATCH --job-name=xsa_pr_dr_gate_3seed
#SBATCH --output=/home/3199937/slurm_logs/xsa_pr_dr_gate_3seed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/xsa_pr_dr_gate_3seed_%j.err
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

# XSA × PR × DR × Gate — 3-seed additivity test.
#
# Adds XSA (value-orthogonalization, parameter-free) on top of the abl8b project
# headline (PR psl=4 + DR recur=[2,3,4,5] target=both + Gate src=proj w=12 at INT8).
# All 3 seeds (42, 123, 1337) in one python invocation so torch.compile is reused.
#
# Reference: abl8b headline → 1.2927 ± 0.0007 bpb (the configuration this script
# enables when USE_XSA=0). USE_XSA=1 is the test.
#
# Predicted outcomes:
#   if XSA gain is fully additive (~the standalone effect from the friend's tests) →
#     ~1.286-1.290 bpb (back-of-envelope; actual standalone effect unknown to me)
#   if XSA composes 92%-95% efficient like other mechs in this project →
#     similar range, perhaps 1.290
#   if no extra gain (XSA's role overlaps with PR or Gate) →
#     ~1.2927 (no improvement on headline)
#   if it hurts → > 1.2927 (XSA conflicts with PR's two-lane logic or gate)
#
# Runtime: 3 × ~55 min ≈ 2h45m. XSA adds ~5% per-step cost (one extra normalize +
# matmul per attention call) so a bit longer than abl8b. SBATCH 5h for margin.
#
# WANDB_MODE=offline (compute-node network has been flaky in prior runs). Sync from
# the LOGIN node after the job finishes:
#   wandb sync wandb/offline-run-*

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=xsa_pr_dr_gate_3seed \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
USE_XSA=1 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_xsa_pr_dr_gate.py

# Best-effort sync from the compute node; tolerate failure.
echo "=== attempting wandb sync from compute node ==="
wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync failed; run 'wandb sync wandb/offline-run-*' from the login node"
