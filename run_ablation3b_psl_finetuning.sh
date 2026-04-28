#!/bin/bash
#SBATCH --job-name=ablation3b_psl_finetuning
#SBATCH --output=/home/3199937/slurm_logs/ablation3b_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation3b_%j.err
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

# Fine-grained sweep around the psl=4 optimum found in ablation3.
# Sweeps psl in {3, 5, 6} (1 seed each) to pin down the exact
# encoder/decoder boundary that maximises final_val_bpb.
#
#   psl=3  last encoder layer + all decoder layers parallel (layers 3-8)
#   psl=5  decoder minus first layer parallel (layers 5-8)
#   psl=6  last 3 decoder layers parallel (layers 6-8)

RUN_ID=ablation3b_psl_finetuning \
PARALLEL_START_LAYER_VALUES=3,5,6 \
SEEDS=1337 \
ITERATIONS=5000 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_gpt_ablation3.py
