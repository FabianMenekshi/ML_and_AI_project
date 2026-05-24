#!/bin/bash
#SBATCH --job-name=split_tracks_attnres_3seed
#SBATCH --output=/home/3199937/slurm_logs/split_tracks_attnres_3seed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/split_tracks_attnres_3seed_%j.err
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

# split_tracks_AttnRes — 3-seed run with the project's standard ablation protocol.
#
# NOTE: this script uses SEED (singular) per invocation, not SEEDS=42,123,1337 like
# our other scripts. So we launch 3 separate python invocations, each with its own
# seed. RUN_ID is shared across them so a single ^split_tracks_attnres_3seed wandb
# regex picks up all three seeds for analysis.
#
# Architecture knobs default to whatever the friend baked into the script; only
# the training-budget and quantization-protocol knobs are overridden to match
# every other ablation in this project:
#   ITERATIONS=5000, WARMDOWN_ITERS=750, TRAIN_BATCH_TOKENS=131072,
#   MAX_WALLCLOCK_SECONDS=0, VAL_LOSS_EVERY=500.
# AttnRes-specific knob (DECODER_ATTNRES_NUM_BLOCKS=4) left at script default.
#
# Runtime: 3 × ~50 min ≈ 2h30m. SBATCH 5h for margin.
# WANDB_MODE=offline (compute-node network has been flaky in prior runs).
# Sync from the LOGIN node after: wandb sync wandb/offline-run-*

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Standard project-protocol overrides used by all 3 seeds
export WANDB_MODE=offline
export ITERATIONS=5000
export WARMDOWN_ITERS=750
export TRAIN_BATCH_TOKENS=131072
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500

# =============================================================================
# SEED 42
# =============================================================================
RUN_ID=split_tracks_attnres_3seed_seed42 \
SEED=42 \
python3 train_gpt_split_tracks_AttnRes_wandb.py

# =============================================================================
# SEED 123
# =============================================================================
RUN_ID=split_tracks_attnres_3seed_seed123 \
SEED=123 \
python3 train_gpt_split_tracks_AttnRes_wandb.py

# =============================================================================
# SEED 1337
# =============================================================================
RUN_ID=split_tracks_attnres_3seed_seed1337 \
SEED=1337 \
python3 train_gpt_split_tracks_AttnRes_wandb.py

# Best-effort sync from the compute node; tolerate failure.
echo "=== attempting wandb sync from compute node ==="
wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync failed; run 'wandb sync wandb/offline-run-*' from the login node"
