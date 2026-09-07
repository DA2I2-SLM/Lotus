#!/bin/bash
# Re-run weather eval for 1B model — split by ctx across 8 GPUs.
# 8 array tasks: 2 modes × 4 ctx values, each evaluating all H={96,192,336,720}.
#
# Task mapping:
#   0=zeroshot/ctx=512   1=zeroshot/ctx=1024   2=zeroshot/ctx=2048   3=zeroshot/ctx=3072
#   4=finetune/ctx=512   5=finetune/ctx=1024   6=finetune/ctx=2048   7=finetune/ctx=3072
#
#SBATCH --job-name=eval_1b_wx
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=output_job/eval_1b_wx_%a.out
#SBATCH --error=output_job/eval_1b_wx_%a.err
#SBATCH --constraint=gpu-v100

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export HF_TOKEN=<HF_TOKEN_REDACTED>

HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}_${SLURM_ARRAY_TASK_ID}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

PERM_HF="${HOME}/.cache/huggingface/hub"
for SNAP in "models--google--gemma-3-1b-pt" "models--google--gemma-3-1b"; do
  if [ -d "${PERM_HF}/${SNAP}" ]; then
    cp -r "${PERM_HF}/${SNAP}" "${HF_HUB_CACHE}/" 2>/dev/null || true
    echo "[cache] copied ${SNAP}"
  fi
done

# ── Task → (mode, ctx) mapping ────────────────────────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID}"
CTX_VALUES=(512 1024 2048 3072)

if [ "${TASK_ID}" -lt 4 ]; then
  MODE="zeroshot"
  CTX="${CTX_VALUES[$TASK_ID]}"
else
  MODE="finetune"
  CTX="${CTX_VALUES[$((TASK_ID - 4))]}"
fi

CKPT="results/sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

OUT_DIR="output_job/eval_1b_results"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/eval_1b_${MODE}_weather_ctx${CTX}.json"

echo "============================================================"
echo "Task ${TASK_ID}: mode=${MODE}  ctx=${CTX}  dataset=weather"
echo "Output: ${OUT_JSON}"
echo "============================================================"

COMMON_ARGS="
  --ckpt ${CKPT}
  --llm_id google/gemma-3-1b-pt
  --data_dir ${DATA_DIR}
  --datasets weather
  --eval_ctx ${CTX}
  --horizons 96 192 336 720
  --n_windows 500
  --seed 42
  --outer_bs 128
  --inner_bs 16
  --device cuda
  --out_json ${OUT_JSON}
"

if [ "${MODE}" = "zeroshot" ]; then
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} --zero_shot
else
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} \
    --ft_ctx 512 --ft_bs 4 --ft_accum 8 \
    --ft_lr 1e-4 --ft_min_lr 1e-5 --ft_stride 32 --ft_epochs 1
fi

echo "Done → ${OUT_JSON}"
