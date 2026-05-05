#!/bin/bash
#SBATCH --job-name=ablation6b_multiseed_and_probes
#SBATCH --output=/home/3199937/slurm_logs/ablation6b_multiseed_and_probes_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation6b_multiseed_and_probes_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3199937/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

# Ablation 6b — finalise PR×DR composition.
#
# Two purposes:
#
# OPTION A — Multi-seed validate abl6 (the headline experiment)
#   Config: psl=4 + recur_layers=[2,3,4,5] + target=both
#   Seeds 42, 123 (seed 1337 already done in abl6 → 1.2968 bpb).
#   Yields the 3-seed mean ± std needed for the headline result.
#   Runtime: 2 × ~64 min ≈ 2h 10m.
#
# OPTION B — Probe where the PR×DR synergy lives (single-seed micro-ablation)
#   Same psl=4, varies which layers get recurrence:
#     [4, 5]  → recur ONLY the parallel-mode layers (full overlap with PR)
#     [2, 3]  → recur ONLY the sequential layers (no overlap with PR)
#   If [4, 5] ≈ 1.2968, the synergy is in parallel-mode recurrence.
#   If [2, 3] ≈ 1.2968, the synergy is just "add capacity in different layers".
#   Runtime: 2 × ~55 min ≈ 1h 50m (2-layer recurrence is cheaper than 4-layer).
#
# Total wall-clock: ~4h on a single GPU. SBATCH time set to 6h for margin.

# =============================================================================
# OPTION A — multi-seed validation: psl=4 + recur=[2,3,4,5] target=both
# =============================================================================
RUN_ID=ablation6_pr_dr_combined_multiseed \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
SEEDS=42,123 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation6_pr_dr_combined.py

# =============================================================================
# OPTION B-1 — probe: recur ONLY the parallel-mode layers [4, 5] (full overlap with PR)
# =============================================================================
RUN_ID=ablation6_probe_overlap \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation6_pr_dr_combined.py

# =============================================================================
# OPTION B-2 — probe: recur ONLY the sequential layers [2, 3] (no overlap with PR)
# =============================================================================
RUN_ID=ablation6_probe_nooverlap \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation6_pr_dr_combined.py
