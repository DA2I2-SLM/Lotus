#!/bin/bash
# Evaluate Gemma-3-1B joint checkpoint — zero-shot AND fine-tuned (1 epoch).
# 6 array tasks: 2 modes × 3 datasets, all run in parallel on separate GPUs.
#
# Task mapping (SLURM_ARRAY_TASK_ID):
#   0=zero-shot/ETTh1   1=zero-shot/ETTh2   2=zero-shot/weather
#   3=finetune/ETTh1    4=finetune/ETTh2    5=finetune/weather
#
#SBATCH --job-name=eval_1b
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=output_job/eval_1b_%a.out
#SBATCH --error=output_job/eval_1b_%a.err
#SBATCH --constraint=gpu-v100

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
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}_${SLURM_ARRAY_TASK_ID}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

# Copy Gemma-3-1B from persistent cache
PERM_HF="${HOME}/.cache/huggingface/hub"
for SNAP in "models--google--gemma-3-1b-pt" "models--google--gemma-3-1b"; do
  if [ -d "${PERM_HF}/${SNAP}" ]; then
    cp -r "${PERM_HF}/${SNAP}" "${HF_HUB_CACHE}/" 2>/dev/null || true
    echo "[cache] copied ${SNAP}"
  fi
done

# ── Task → (mode, dataset) mapping ───────────────────────────────────────────
MODES=("zeroshot" "zeroshot" "zeroshot" "finetune" "finetune" "finetune")
DATASETS=("ETTh1" "ETTh2" "weather" "ETTh1" "ETTh2" "weather")

MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

CKPT="results/sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="${FTS_TSREASON_DIR}"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

OUT_DIR="output_job/eval_1b_results"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/eval_1b_${MODE}_${DATASET}.json"

echo "============================================================"
echo "Gemma-3-1B Eval  task=${SLURM_ARRAY_TASK_ID}"
echo "Mode    : ${MODE}"
echo "Dataset : ${DATASET}"
echo "Output  : ${OUT_JSON}"
echo "============================================================"

# Shared args
COMMON_ARGS="
  --ckpt ${CKPT}
  --llm_id google/gemma-3-1b-pt
  --data_dir ${DATA_DIR}
  --datasets ${DATASET}
  --eval_ctx 512 1024 2048 3072
  --horizons 96 192 336 720
  --n_windows 500
  --seed 42
  --outer_bs 128
  --inner_bs 8
  --device cuda
  --out_json ${OUT_JSON}
"

if [ "${MODE}" = "zeroshot" ]; then
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} --zero_shot
else
  # Fine-tune 1 epoch (LLM frozen, enc/dec/proj trainable)
  # Smaller batch for 1B backbone memory
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
