#!/bin/bash
#SBATCH --job-name=ablation3_parallel_residuals
#SBATCH --output=/home/3199937/slurm_logs/ablation3_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation3_%j.err
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

# Sweep axis: where to start the parallel residual lanes.
#   -1 = disabled (sequential baseline, same architecture as train_gpt.py)
#    7 = last 2 layers parallel (mirrors the 2026-03-31 leaderboard entry, 1.1063 BPB)
#    4 = full decoder half parallel (layers 4-8 of 9)
#    0 = all layers parallel
#


RUN_ID=ablation3_parallel_residuals \
PARALLEL_START_LAYER_VALUES=-1,7,4,0 \
SEEDS=1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py
