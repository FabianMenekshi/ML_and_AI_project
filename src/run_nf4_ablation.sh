#!/bin/bash
#SBATCH --job-name=nf4_ablation
#SBATCH --output=/home/3241043/slurm_logs/nf4_ablation_%j.out
#SBATCH --error=/home/3241043/slurm_logs/nf4_ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3241043
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3241043/slurm_logs
cd /mnt/beegfsstudents/home/3241043/projects/ML_and_AI_project

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate golf

python --version

export WANDB_DIR=/mnt/beegfsstudents/home/3241043/projects/ML_and_AI_project/wandb
mkdir -p "$WANDB_DIR"

run_one () {
  GRAN=$1
  CAL=$2
  SEED=$3

  RUN_ID=nf4_${GRAN}_${CAL} \
  SEEDS=$SEED \
  ITERATIONS=5000 \
  MAX_WALLCLOCK_SECONDS=0 \
  WARMDOWN_ITERS=750 \
  TRAIN_BATCH_TOKENS=131072 \
  VAL_LOSS_EVERY=500 \
  QUANTIZE_ALL=1 \
  QUANT_METHOD=nf4 \
  MATRIX_QUANT_BITS=4 \
  QUANT_GRANULARITY=$GRAN \
  QUANT_SYMMETRY=symmetric \
  QUANT_CALIBRATION=$CAL \
  python3 ablation_nf4.py
}

# 3 NF4 configs x 3 seeds = 9 runs.
for SEED in 1337 42 123; do
  run_one per_tensor uncalibrated $SEED
  run_one per_channel uncalibrated $SEED
  run_one per_channel percentile $SEED
done
