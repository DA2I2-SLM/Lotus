#!/bin/bash
# GIFT-Eval zero-shot evaluation of OpenTSLMSPFixPos2D.
# 5-task array; each task handles 33 of the 165 configs (55 datasets × 3 terms).
# Saves results to results/OpenTSLMSPFixPos2D/results_task{0-4}.csv
#
#SBATCH --job-name=gift_eval
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
#SBATCH --output=/scratch/s225250685/project/FoundationalTS/output_job/gift_eval_%a.out
#SBATCH --error=/scratch/s225250685/project/FoundationalTS/output_job/gift_eval_%a.err

set -euo pipefail
mkdir -p /scratch/s225250685/project/FoundationalTS/output_job

TASK_ID=${SLURM_ARRAY_TASK_ID}

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Use local gemma weights (no HF auth needed)
export HF_HUB_OFFLINE=1
GEMMA_LOCAL="/vast/s225250685/models/gemma-3-270m"

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
GIFT_DATA="/scratch/s225250685/project/gift_eval_data"
OUT_DIR="/vast/s225250685/project/FoundationalTS/results/OpenTSLMSPFixPos2D"

echo "========================================================"
echo " GIFT-Eval — OpenTSLMSPFixPos2D — task ${TASK_ID}/5"
echo " ckpt : ${CKPT}"
echo " data : ${GIFT_DATA}"
echo " out  : ${OUT_DIR}"
echo "========================================================"

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/gift_eval_eval.py \
    --ckpt           "${CKPT}" \
    --llm_id         "${GEMMA_LOCAL}" \
    --gift_eval_data "${GIFT_DATA}" \
    --task_id        "${TASK_ID}" \
    --n_tasks        5 \
    --out_dir        "${OUT_DIR}" \
    --device         cuda

echo "=== task ${TASK_ID} DONE ==="
