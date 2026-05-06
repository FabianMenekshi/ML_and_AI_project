#!/bin/bash
#SBATCH --job-name=run_golf
#SBATCH --output=/home/3245806/slurm_logs/first_run_golf_%j.out
#SBATCH --error=/home/3245806/slurm_logs/first_run_golf_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3245806
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3245806/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

RUN_ID=ablation_full_quantization_correct \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
QUANTIZE_ALL=1 \
MATRIX_QUANT_BITS=8 \
python3 ablation_baseline_full_quantization.py
