#!/bin/bash
#SBATCH --job-name=combined_mixed_qat_L11
#SBATCH --output=/home/3199937/slurm_logs/combined_mixed_qat_L11_%j.out
#SBATCH --error=/home/3199937/slurm_logs/combined_mixed_qat_L11_%j.err
#SBATCH --time=08:00:00
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

# COMBINED 3-seed run, NUM_LAYERS=11.
#
# Architecture: PR × DR × Gate (unchanged from train_gpt_combined.py)
# Quantization: Mixed int6/int8 naive + QAT at 25%
#   - Non-recurrent layers → int6
#   - Recurrent layers (2,3,4,5) → int8
#   - QAT enabled from step 1250 (25% of 5000)
#   - Embedding QAT at int8

WANDB_MODE=offline \
RUN_ID=combined_pr_dr_gate_mixed_qat_L11 \
NUM_LAYERS=11 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
MATRIX_QUANT_BITS=6 \
RECUR_QUANT_BITS=8 \
QAT_ENABLED=1 \
QAT_START_FRACTION=0.25 \
QAT_EMBED_BITS=8 \
SEEDS=42,123,1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_combined_qat.py

# Best-effort sync from the compute node; tolerate failure.