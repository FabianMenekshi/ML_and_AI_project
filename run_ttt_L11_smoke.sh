#!/bin/bash
#SBATCH --job-name=ttt_L11_smoke
#SBATCH --output=/home/3199937/slurm_logs/ttt_L11_smoke_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ttt_L11_smoke_%j.err
#SBATCH --time=00:30:00
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

# TTT-Linear stability smoke test.
#
# Goal: verify that the truncated-BPTT fix (W.detach() after each chunked inner
# update) eliminates the gradient explosion observed in the previous full sweep.
#
# Previously, with the *untruncated* version, every single one of 6 sweep configs
# went NaN by step 200 (and 'chunk32' went NaN at step 2). The matrix_grad_norm
# blew up 50–6000× between step 1 and step 2 because the outer gradient flowed
# through n_chunks (16–32) chained matrix-multiply Jacobians.
#
# This smoke test runs the "center" config of the sweep for only ITERATIONS=100
# (enough to be PAST step 200's danger zone and into the QAT-active regime at
# step 25, but cheap enough to fail fast). What we want to see in the log:
#
#   ✓ matrix_grad_norm stays in a healthy band (roughly 0.05 to 10) throughout.
#   ✓ train_loss decreases monotonically-ish (~6.9 → ~4 over 100 steps).
#   ✓ No NaN anywhere.
#   ✓ ttt_inner_lr_mean and ttt_W_init_norm_mean are finite at end-of-run.
#
# If the smoke test passes (no NaN, healthy grad norms, decreasing loss), we can
# rerun the full sweep with confidence.
#
# Runtime: ~5–10 min including torch.compile warmup. SBATCH 30 min budget.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

WANDB_MODE=offline \
RUN_ID=ttt_L11_smoke_center_iters100 \
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
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=64 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=1.0 \
SEEDS=1337 \
ITERATIONS=100 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=15 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=5 \
python3 train_gpt_combined_qat_emb_ttt.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
