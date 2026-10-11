#!/bin/bash
# Retry failed GIFT-Eval NaN configs with nanmean/nanstd fix.
# 32 configs across 5 tasks (~6-7 each).
#
#SBATCH --job-name=gift_retry
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --array=0-4%5
#SBATCH --output=output_job/gift_retry_%a.out
#SBATCH --error=output_job/gift_retry_%a.err

# Resolve repo root + interpreter + data locations dynamically so this
# script runs on any account. Override any FTS_* var in your environment.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -f "$_d/bashfiles/_paths.sh" ]; do _d="$(dirname "$_d")"; done
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
CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
GIFT_DATA="${FTS_GIFT_DATA}"
OUT_DIR="${FTS_RESULTS_ROOT}/OpenTSLMSPFixPos2D"

echo "========================================================"
echo " GIFT-Eval RETRY — OpenTSLMSPFixPos2D — task ${TASK_ID}/5"
echo " ckpt : ${CKPT}"
echo " out  : ${OUT_DIR}"
echo "========================================================"

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/gift_eval_retry.py \
    --ckpt           "${CKPT}" \
    --llm_id         "${GEMMA_LOCAL}" \
    --gift_eval_data "${GIFT_DATA}" \
    --task_id        "${TASK_ID}" \
    --n_tasks        5 \
    --out_dir        "${OUT_DIR}" \
    --prefix         "retry3_task"

echo "=== retry task ${TASK_ID} DONE ==="
