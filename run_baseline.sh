#!/bin/bash
#SBATCH --job-name=run_golf
#SBATCH --output=/home/3199937/slurm_logs/first_run_golf_%j.out
#SBATCH --error=/home/3199937/slurm_logs/first_run_golf_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3199937
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=2

mkdir -p /home/3199937/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

RUN_ID=golf_baseline_antonio \
ITERATIONS=15000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=1000 \
python3 train_gpt.py
