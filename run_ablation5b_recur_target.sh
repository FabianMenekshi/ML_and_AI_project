#!/bin/bash
#SBATCH --job-name=ablation5b_recur_target
#SBATCH --output=/home/3199937/slurm_logs/ablation5b_recur_target_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation5b_recur_target_%j.err
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

# Sweep: which sub-component of the block to repeat on the extra pass.
#   "attn" – only the attention sub-layer runs again (iterative routing)
#   "mlp"  – only the MLP sub-layer runs again (iterative feature building)
#   "both" – both attention and MLP run again (no resid_mix re-applied)
#
# RECUR_LAYERS: fixed layer set — set to the best config from ablation 5.
#   Layers 0-3 are encoder, layers 4-8 are decoder.
#   Default: "4,5" (first two decoder blocks, most common winning config).
#
# RECUR_TIMES: extra passes per sub-component (1 → sub-component runs twice total).
RUN_ID=ablation5b_recur_target \
RECUR_TARGET_VALUES="attn;mlp;both" \
RECUR_LAYERS="4,5" \
RECUR_TIMES=1 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation5b_recur_target.py
