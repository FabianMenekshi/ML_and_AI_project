#!/bin/bash
#SBATCH --job-name=ablation6c_recur_times2
#SBATCH --output=/home/3199937/slurm_logs/ablation6c_recur_times2_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ablation6c_recur_times2_%j.err
#SBATCH --time=03:00:00
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

# Ablation 6c — vertical axis of depth recurrence on the composed config.
#
# Anchor: abl6 (PR × DR) at recur_times=1 → 1.2987 ± 0.0028 (3 seeds).
# Same psl=4, same recur_layers=[2,3,4,5], same target=both — only the number of
# extra passes per recurred block changes.
#
# recur_times=1 → each recurred block runs 2 times total (the abl6 baseline).
# recur_times=2 → each recurred block runs 3 times total (this experiment).
#
# Cost: each recurred block does 1.5× the work of the abl6 baseline, so per-run
# wallclock is ~90 min (vs abl6's 64 min). The 3-hour SBATCH allocation is generous.
#
# Decision rule (single-seed probe, seed 1337):
#   bpb < 1.296 → meaningful gain, multi-seed validate
#   bpb in 1.296-1.299 → likely noise, multi-seed only if you want to be sure
#   bpb > 1.299 → vertical axis saturates; abl6 at recur_times=1 is the locked winner

RUN_ID=ablation6c_recur_times2 \
PARALLEL_START_LAYER=4 \
PARALLEL_ASYM_INIT=0 \
RECUR_LAYERS="2,3,4,5" \
RECUR_TARGET=both \
RECUR_TIMES=2 \
SEEDS=1337 \
ITERATIONS=5000 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=750 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
python3 train_gpt_ablation6_pr_dr_combined.py
