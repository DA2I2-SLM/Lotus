#!/bin/bash
# Evaluate the Gemma-3-1B LoRA r=8 joint checkpoint (job 3462055, epoch 2 snapshot)
# — zero-shot AND fine-tuned (1 epoch) — on ETTh1 / ETTh2 / weather.
#
# Grid and hyper-parameters are IDENTICAL to run_eval_1b_array.sh /
# run_eval_1b_weather_fast.sh so the numbers drop straight into the
# "1B Model" sheet of results/finetune_comparison.xlsx next to the
# non-LoRA 1B run. The only additions are --lora_r 8 --lora_alpha 16.
#
# ETT runs cover all 4 contexts in one task (7 channels, cheap).
# Weather is split per-context (21 channels, ~3x the work) — the same
# split that let the non-LoRA weather eval finish inside the walltime.
#
# Task mapping (SLURM_ARRAY_TASK_ID):
#    0 = zeroshot ETTh1 (ctx 512/1024/2048/3072)
#    1 = zeroshot ETTh2 (ctx 512/1024/2048/3072)
#    2 = finetune ETTh1 (ctx 512/1024/2048/3072)
#    3 = finetune ETTh2 (ctx 512/1024/2048/3072)
#  4-7 = zeroshot weather, ctx 512 / 1024 / 2048 / 3072
# 8-11 = finetune weather, ctx 512 / 1024 / 2048 / 3072
#
#SBATCH --job-name=eval_1b_r8v3
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%10
#SBATCH --output=output_job/eval_1b_r8v3_%a.out
#SBATCH --error=output_job/eval_1b_r8v3_%a.err
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

# Offline mode is REQUIRED here, not just an optimisation. This transformers
# version calls is_base_mistral() -> hub model_info() while building the
# tokenizer, even when the model is fully cached. With 10 concurrent tasks
# that trips HF's 1000-req/5-min API limit and every task dies with
# "429 Too Many Requests" surfaced as "Unable to load vocabulary from file".
# Everything needed is in the copied cache below, so force local resolution.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}_${SLURM_ARRAY_TASK_ID}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

# Copy Gemma-3-1B from the persistent cache so 10 concurrent tasks do not
# all hit the hub at once.
PERM_HF="${HOME}/.cache/huggingface/hub"
for SNAP in "models--google--gemma-3-1b-pt" "models--google--gemma-3-1b"; do
  if [ -d "${PERM_HF}/${SNAP}" ]; then
    cp -r "${PERM_HF}/${SNAP}" "${HF_HUB_CACHE}/" 2>/dev/null || true
    echo "[cache] copied ${SNAP}"
  fi
done

# RUN_TAG separates result sets from different checkpoints of the same run.
RUN_TAG="${RUN_TAG:-r8v2}"
OUT_DIR="output_job/eval_1b_${RUN_TAG}_results"

# ── Task → (mode, dataset, ctx list) mapping ─────────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID}"
CTX_VALUES=(512 1024 2048 3072)
ALL_CTX="512 1024 2048 3072"

if [ "${TASK_ID}" -lt 4 ]; then
  # ETT tasks: whole context sweep in one task
  MODES=("zeroshot" "zeroshot" "finetune" "finetune")
  DSETS=("ETTh1" "ETTh2" "ETTh1" "ETTh2")
  MODE="${MODES[$TASK_ID]}"
  DATASET="${DSETS[$TASK_ID]}"
  EVAL_CTX="${ALL_CTX}"
  INNER_BS=8
  OUT_JSON="${OUT_DIR}/eval_1b_${RUN_TAG}_${MODE}_${DATASET}.json"
else
  # Weather tasks: one context per task
  IDX=$((TASK_ID - 4))
  if [ "${IDX}" -lt 4 ]; then
    MODE="zeroshot"
    CTX="${CTX_VALUES[$IDX]}"
  else
    MODE="finetune"
    CTX="${CTX_VALUES[$((IDX - 4))]}"
  fi
  DATASET="weather"
  EVAL_CTX="${CTX}"
  INNER_BS=16
  OUT_JSON="${OUT_DIR}/eval_1b_${RUN_TAG}_${MODE}_weather_ctx${CTX}.json"
fi

# Snapshot of the live job-3462055 checkpoint. Copied out of the training dir
# on purpose: the run overwrites best_model.pt on every val improvement, and
# wipes the whole dir on resubmit unless KEEP_CKPT=1.
#
# Override per run, e.g.
#   CKPT=checkpoint_snapshots/lora_r8_1b_<date>.pt RUN_TAG=r8v2 \
#     sbatch bashfiles/evaluate/finetune/run_eval_1b_lora_r8_array.sh
CKPT="${CKPT:-checkpoint_snapshots/lora_r8_1b_20260808_v2.pt}"
DATA_DIR="${DATA_DIR:-/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw}"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Gemma-3-1B LoRA r=8 eval   task=${TASK_ID}"
echo "Mode     : ${MODE}"
echo "Dataset  : ${DATASET}"
echo "Eval ctx : ${EVAL_CTX}"
echo "Ckpt     : ${CKPT}"
echo "Output   : ${OUT_JSON}"
echo "============================================================"

COMMON_ARGS="
  --ckpt ${CKPT}
  --llm_id google/gemma-3-1b-pt
  --lora_r 8
  --lora_alpha 16
  --data_dir ${DATA_DIR}
  --datasets ${DATASET}
  --eval_ctx ${EVAL_CTX}
  --horizons 96 192 336 720
  --n_windows 500
  --seed 42
  --outer_bs 128
  --inner_bs ${INNER_BS}
  --device cuda
  --out_json ${OUT_JSON}
"

if [ "${MODE}" = "zeroshot" ]; then
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} --zero_shot
else
  # Fine-tune 1 epoch: enc/proj/dec + LoRA adapters trainable, LLM base frozen.
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} \
    --ft_ctx 512 \
    --ft_bs 4 \
    --ft_accum 8 \
    --ft_lr 1e-4 \
    --ft_min_lr 1e-5 \
    --ft_stride 32 \
    --ft_epochs 1
fi

echo "Done → ${OUT_JSON}"
