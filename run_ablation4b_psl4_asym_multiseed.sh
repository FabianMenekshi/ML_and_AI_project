#!/bin/bash
#SBATCH --job-name=ablation4b_psl4_asym_multiseed
#SBATCH --output=/home/3199937/slurm_logs/ablation4b_psl4_asym_multiseed_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation4b_psl4_asym_multiseed_%j.err
#SBATCH --time=12:00:00
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

# Multi-seed validation of psl=4 + asymmetric routing init (the abl3c winner).
#
# State going in:
#   abl3c psl=4 sym  init  seed=1337                     → 1.3046 bpb (single seed)
#   abl3c psl=4 asym init  seed=1337                     → 1.3042 bpb (single seed) ← winner
#   abl4  psl=4 sym  init  seeds={42, 123, 1337}         → 1.3057 ± 0.0015 (3 seeds)
#
# The single-seed asym→sym gap is only 0.0004 bpb, well below the seed-to-seed
# std of 0.0015. We need seeds 42 and 123 to compute mean ± std for asym init
# on the same 3-seed grid as abl4 and decide whether asym init genuinely helps.
#
# Seed 1337 is already done in abl3c — we only run the missing two seeds.

RUN_ID=ablation4b_psl4_asym_multiseed \
PARALLEL_START_LAYER_VALUES=4 \
PARALLEL_ASYM_INIT=1 \
SEEDS=42,123 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py
