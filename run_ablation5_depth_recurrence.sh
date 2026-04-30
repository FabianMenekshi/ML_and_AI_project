#!/bin/bash
#SBATCH --job-name=ablation5_depth_recurrence
#SBATCH --output=/home/3199937/slurm_logs/ablation5_depth_recurrence_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5_depth_recurrence_%j.err
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

# Sweep: which blocks to repeat (shared weights, zero extra params).
# Layers 0-3 are encoder, layers 4-8 are decoder.
# The U-Net hinge sits between layer 3 (last encoder) and layer 4 (first decoder).
# Values are semicolon-separated; commas separate layers within one config.
#   "4"     – repeat only the first decoder block
#   "4,5"   – repeat the first two decoder blocks (most common in leaderboard entries)
#   "3,4"   – repeat the last encoder + first decoder (straddle the hinge)
#   "5,6"   – repeat deeper decoder blocks
RUN_ID=ablation5_depth_recurrence \
RECUR_LAYERS_VALUES="4;4,5;3,4;5,6" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5_depth_recurrence.py
