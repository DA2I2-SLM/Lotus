#!/bin/bash
# GIFT-Eval zero-shot evaluation — TS-only LoRA model (phase1_chronos_mse_lora_freq_mvpos).
# 5-task array covering all 97 official GIFT-Eval configs.
#
#SBATCH --job-name=gift_ts_only_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --array=0-4%5
#SBATCH --output=output_job/gift_ts_only_lora_%a.out
#SBATCH --error=output_job/gift_ts_only_lora_%a.err

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any FTS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set FTS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p ${FTS_REPO_ROOT}/output_job

TASK_ID=${SLURM_ARRAY_TASK_ID}

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export HF_HUB_OFFLINE=1
GEMMA_LOCAL="${FTS_GEMMA_LOCAL}"

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos/checkpoints/best_model.pt"
GIFT_DATA="${FTS_GIFT_DATA}"
OUT_DIR="${FTS_RESULTS_ROOT}/OpenTSLMSPFixPos2D_ts_only_lora"

echo "========================================================"
echo " GIFT-Eval — TS-only LoRA (phase1_chronos_mse_lora_freq_mvpos) — task ${TASK_ID}/5"
echo " ckpt : ${CKPT}"
echo " data : ${GIFT_DATA}"
echo " out  : ${OUT_DIR}"
echo " lora_r=4  lora_alpha=8"
echo "========================================================"

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/gift_eval_eval.py \
    --ckpt           "${CKPT}" \
    --llm_id         "${GEMMA_LOCAL}" \
    --gift_eval_data "${GIFT_DATA}" \
    --task_id        "${TASK_ID}" \
    --n_tasks        5 \
    --out_dir        "${OUT_DIR}" \
    --lora_r         4 \
    --lora_alpha     8 \
    --device         cuda
