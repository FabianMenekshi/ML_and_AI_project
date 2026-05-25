#!/bin/bash
#SBATCH --job-name=combined_mixed_qat_L11_2345_seed123_verify
#SBATCH --output=/home/3199937/slurm_logs/combined_mixed_qat_L11_2345_seed123_verify_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_mixed_qat_L11_2345_seed123_verify_%j.err
#SBATCH --time=02:00:00
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

# Verify friend's seed-123 [2,3,4,5] run — they got bpb=1.2773 (unusually low; 1.6σ below
# the now-known 6-seed mean of 1.2809). This re-runs the EXACT same config under my account
# to check if I get the same number. Three possible outcomes:
#
#   (a) my seed-123 lands at ~1.277  → fully reproducible, the low value is genuinely
#                                       a property of seed 123 (favourable init/data order)
#   (b) my seed-123 lands at ~1.281  → friend's run had some non-determinism we're missing
#                                       (different library versions? different cluster node?)
#   (c) my seed-123 lands elsewhere  → unexpected, investigate
#
# All same env vars as friend's good runs (verified via wandb config). Single seed → 1 invocation.
# Distinct RUN_ID so this doesn't clash with the friend's existing wandb run.
#
# Runtime: ~70 min. SBATCH 2h budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=combined_mixed_qat_L11_2345_seed123_verify_Antonio \
NUM_LAYERS=11 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
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
SEEDS=123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat_emb.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
