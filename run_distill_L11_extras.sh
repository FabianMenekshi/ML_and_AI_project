#!/bin/bash
#SBATCH --job-name=distill_L11_extras
#SBATCH --output=/home/3199937/slurm_logs/distill_L11_extras_%j.out
#SBATCH --error=/home/3199937/slurm_logs/distill_L11_extras_%j.err
#SBATCH --time=05:00:00
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

# Single-seed probes around the in-budget distillation baseline, exploring three
# distinct axes:
#
#   Probe A: 4600 + 400  (more distillation, less main training)
#     -> Is the optimal main/distill split closer to 4800/200 or 4600/400?
#        If the BPB at 4600+400 < 4800+200, distillation is under-allocated.
#
#   Probe B: 4800 + 200 with DISTILL_LR=3e-4  (more aggressive distillation LR)
#     -> Is our default LR=1e-4 too conservative? Higher LR could reach the
#        teacher faster but might destabilise. 3e-4 is the next standard step.
#
#   Probe C: 4800 + 200 with DISTILL_TEMPERATURE=2.0  (softer teacher logits)
#     -> Higher temperature spreads the teacher's probability mass across more
#        tokens, which often helps distillation by providing richer training
#        signal beyond just the argmax.
#
# Each probe uses a single seed (1337) for speed. Per-probe runtime ~74 min.
# Total: 3 probes * 74 min = ~3h45m. SBATCH 5h budget.
#
# Comparison points (no need to re-run):
#   - over-budget distillation: 1.2793 +- 0.0004 (3 seeds, 5000+200)
#   - in-budget distillation:   measured by run_distill_L11_inbudget.sh
#   - baseline (no distill):    1.2812 +- 0.0006 (5 seeds)

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Shared config across all 3 probes
export WANDB_MODE=offline
export NUM_LAYERS=11
export PARALLEL_START_LAYER=4
export PARALLEL_ASYM_INIT=0
export RECUR_LAYERS="3,4,5"
export RECUR_TARGET=both
export RECUR_TIMES=1
export GATE_ATTN_OUT=1
export GATE_ATTN_SRC=proj
export GATE_WIDTH=12
export MATRIX_QUANT_BITS=6
export RECUR_QUANT_BITS=8
export EMBED_QUANT_MODE=8
export COMPRESSION_METHOD=brotli
export BROTLI_QUALITY=11
export QAT_ENABLED=1
export QAT_START_FRACTION=0.25
export QAT_EMBED_BITS=8
export DISTILL_ENABLED=1
export DISTILL_OPTIMIZER=adamw
export SEEDS=1337
export MAX_WALLCLOCK_SECONDS=0
export WARMDOWN_ITERS=750
export TRAIN_BATCH_TOKENS=131072
export VAL_LOSS_EVERY=500

# =============================================================================
# Probe A - 4600 + 400 (more distillation, less main)
# =============================================================================
RUN_ID=distill_L11_probeA_main4600_distill400 \
ITERATIONS=4600 \
DISTILL_STEPS=400 \
DISTILL_LR=1e-4 \
DISTILL_TEMPERATURE=1.0 \
python3 train_gpt_combined_qat_emb_distill.py

# =============================================================================
# Probe B - 4800 + 200 with DISTILL_LR=3e-4 (more aggressive LR)
# =============================================================================
RUN_ID=distill_L11_probeB_main4800_distill200_lr3e4 \
ITERATIONS=4800 \
DISTILL_STEPS=200 \
DISTILL_LR=3e-4 \
DISTILL_TEMPERATURE=1.0 \
python3 train_gpt_combined_qat_emb_distill.py

# =============================================================================
# Probe C - 4800 + 200 with DISTILL_TEMPERATURE=2.0 (softer teacher)
# =============================================================================
RUN_ID=distill_L11_probeC_main4800_distill200_T2 \
ITERATIONS=4800 \
DISTILL_STEPS=200 \
DISTILL_LR=1e-4 \
DISTILL_TEMPERATURE=2.0 \
python3 train_gpt_combined_qat_emb_distill.py

# Best-effort sync with a 3-minute cap.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
