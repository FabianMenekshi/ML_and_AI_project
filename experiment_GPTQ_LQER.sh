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

RUN_ID=GPTQ_LQER_int6_int7_rank4_top3_group32_asym \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
USE_GPTQ_LQER=1 \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=7 \
LQER_RANK=4 \
LQER_TOP_K=3 \
QUANT_GROUP_SIZE=32 \
QUANT_ASYMMETRIC=1 \
python3 train_gpt_gptq_lqer.py
