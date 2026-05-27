#!/bin/bash
#SBATCH --job-name=ttt_L11_sweep
#SBATCH --output=/home/3199937/slurm_logs/ttt_L11_sweep_%j.out
#SBATCH --error=/home/3199937/slurm_logs/ttt_L11_sweep_%j.err
#SBATCH --time=09:00:00
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

# TTT-Linear hyperparameter sweep at L=11 (single-seed 1337 per config).
#
# Strategy: 1 center config + 5 one-axis perturbations. Single-seed sweep first,
# then a separate multi-seed validation bash on the winner (to be written after this
# job finishes and we have the data to pick a winner).
#
# All configs share the L=11 final architecture (PR psl=4, DR [3,4,5], Gate proj/w12)
# + mixed int6/int8 quantisation + QAT at 25%. The only varying knobs are the four
# TTT hyperparameters.
#
# Sweep table:
#   #  tag           TTT_LAYERS  STATE_DIM  CHUNK  INNER_LR    motivation
#   1  center        1,2,3       64         64     1.0         my best-guess default
#   2  layers0123    0,1,2,3     64         64     1.0         does layer 0 want TTT too?
#   3  d32           1,2,3       32         64     1.0         narrower W (lower capacity, smaller artifact)
#   4  d96           1,2,3       96         64     1.0         wider W (more capacity)
#   5  chunk32       1,2,3       64         32     1.0         more inner-loop steps (32 chunks/seq vs 16)
#   6  lr01          1,2,3       64         64     0.1         smaller initial inner step
#
# All RUN_IDs share the `^ttt_L11_sweep_` regex prefix so they're queryable together.
# Baseline for comparison: the 5-seed L=11 mean of 1.2812 ± 0.0007 BPB at 15.36 MB.
#
# Per-config runtime: ~70–82 min depending on state_dim and chunk_size.
# Total estimated: ~7h30m. SBATCH 9h budget for margin.

echo "=== network sanity ==="
curl -sS -o /dev/null -w "wandb api HTTP %{http_code} (time %{time_total}s)\n" \
    --max-time 15 https://api.wandb.ai/ 2>&1 || echo "curl to api.wandb.ai FAILED (network unreachable)"
echo "=== end network sanity ==="

# Shared protocol — applies to all 6 probes.
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
export SEEDS=1337
export ITERATIONS=5000
export MAX_WALLCLOCK_SECONDS=0
export WARMDOWN_ITERS=750
export TRAIN_BATCH_TOKENS=131072
export VAL_LOSS_EVERY=500

# =============================================================================
# Probe 1 — center: my best-guess default (3 layers, d=64, chunk=64, inner_lr=1.0)
# =============================================================================
RUN_ID=ttt_L11_sweep_center \
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=64 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=1.0 \
python3 train_gpt_combined_qat_emb_ttt.py

# =============================================================================
# Probe 2 — layers0123: add the very first encoder block to the TTT set
# =============================================================================
RUN_ID=ttt_L11_sweep_layers0123 \
TTT_LAYERS="0,1,2,3" \
TTT_STATE_DIM=64 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=1.0 \
python3 train_gpt_combined_qat_emb_ttt.py

# =============================================================================
# Probe 3 — d32: narrower W (less capacity, smaller artifact)
# =============================================================================
RUN_ID=ttt_L11_sweep_d32 \
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=32 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=1.0 \
python3 train_gpt_combined_qat_emb_ttt.py

# =============================================================================
# Probe 4 — d96: wider W (more capacity at ~+1.2% params vs center)
# =============================================================================
RUN_ID=ttt_L11_sweep_d96 \
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=96 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=1.0 \
python3 train_gpt_combined_qat_emb_ttt.py

# =============================================================================
# Probe 5 — chunk32: smaller chunks → 32 inner-loop steps per seq (vs 16 at center)
# =============================================================================
RUN_ID=ttt_L11_sweep_chunk32 \
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=64 \
TTT_CHUNK_SIZE=32 \
TTT_INNER_LR_INIT=1.0 \
python3 train_gpt_combined_qat_emb_ttt.py

# =============================================================================
# Probe 6 — lr01: gentler initial inner step (scalar is learnable, init still matters)
# =============================================================================
RUN_ID=ttt_L11_sweep_lr01 \
TTT_LAYERS="1,2,3" \
TTT_STATE_DIM=64 \
TTT_CHUNK_SIZE=64 \
TTT_INNER_LR_INIT=0.1 \
python3 train_gpt_combined_qat_emb_ttt.py

# Best-effort sync with a 3-minute cap so a hung sync can't bleed out the slurm wallclock.
echo "=== attempting wandb sync from compute node (3-min timeout) ==="
timeout 180 wandb sync wandb/offline-run-* 2>&1 || \
    echo "compute-node sync timed out or failed; run 'wandb sync wandb/offline-run-*' from the login node"
