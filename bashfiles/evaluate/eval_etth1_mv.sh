#!/bin/bash
#SBATCH --job-name=etth1-mv-eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100

# Resolve repo root + interpreter + data locations dynamically so this
# script runs on any account. Override any FTS_* var in your environment.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -f "$_d/bashfiles/_paths.sh" ]; do _d="$(dirname "$_d")"; done
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"
## SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

# module load Anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"

# cd ${FTS_REPO_ROOT}
# conda activate ./.env

export PYTHONPATH="${FTS_REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/etth1_eval.py
ETTh1=${FTS_ETT_DIR}/ETT-small/ETTh1.csv

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: PATCH_SIZE must match the patch_size the checkpoint was trained
# with. The checkpoint path encodes that size (e.g. sincos_patch64).
# Edit CKPT and PATCH_SIZE together.
# ─────────────────────────────────────────────────────────────────────────────
STAGE=joint_phase1_extend_balanced
CKPT=${FTS_REPO_ROOT}/results/sincos_patch64/gemma_3_270m/OpenTSLMSPFixPos2D/${STAGE}/checkpoints/best_model.pt
PATCH_SIZE=64

HISTORY_LEN=512     # must be a multiple of PATCH_SIZE
PRED_LEN=128        # must be a multiple of PATCH_SIZE
EVAL_SPLIT=test     # train | val | test
MODE=both           # autoregressive | teacher_forced | both

OUT_DIR=./eval_results/${STAGE}/etth1_mv/H${HISTORY_LEN}_P${PRED_LEN}_p${PATCH_SIZE}_${EVAL_SPLIT}

python ${INFER} \
  --ckpt         ${CKPT} \
  --llm_id       ${LLM} \
  --etth1_csv    ${ETTh1} \
  --patch_size   ${PATCH_SIZE} \
  --history_len  ${HISTORY_LEN} \
  --pred_len     ${PRED_LEN} \
  --stride       1 \
  --eval_split   ${EVAL_SPLIT} \
  --mode         ${MODE} \
  --pe_type      sincos \
  --batch_size   8 \
  --plot_every   50 \
  --out_dir      ${OUT_DIR} \
  --device       cuda
