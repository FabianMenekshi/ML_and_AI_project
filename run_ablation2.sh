#!/bin/bash
#SBATCH --job-name=ablation2_weight_decay
#SBATCH --output=/home/3199937/slurm_logs/ablation2_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation2_%j.err
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

RUN_ID=ablation2_weight_decay \
SCALAR_WD_VALUES=0.0,0.04 \
MUON_WD_VALUES=0.0,0.04 \
SEEDS=1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation2.py
