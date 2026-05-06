#!/bin/bash
#SBATCH --job-name=quant_design
#SBATCH --output=/home/3241043/slurm_logs/quant_design_%j.out
#SBATCH --error=/home/3241043/slurm_logs/quant_design_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=3241043
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3241043/slurm_logs
module load miniconda3

eval "$(conda shell.bash hook)"
conda activate golf
python --version

for BITS in 8 4; do
  for GRAN in per_tensor per_channel; do
    for SYM in symmetric asymmetric; do
      for CAL in uncalibrated percentile; do
        RUN_ID=quant_design_${BITS}_${GRAN}_${SYM}_${CAL} \
        ITERATIONS=5000 \
        MAX_WALLCLOCK_SECONDS=0 \
        WARMDOWN_ITERS=750 \
        TRAIN_BATCH_TOKENS=131072 \
        VAL_LOSS_EVERY=500 \
        QUANTIZE_ALL=1 \
        MATRIX_QUANT_BITS=$BITS \
        QUANT_GRANULARITY=$GRAN \
        QUANT_SYMMETRY=$SYM \
        QUANT_CALIBRATION=$CAL \
        python3 ablation_quant_design_study.py
      done
    done
  done
done