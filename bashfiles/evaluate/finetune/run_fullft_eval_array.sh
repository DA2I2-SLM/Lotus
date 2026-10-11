#!/bin/bash
# Full fine-tuning evaluation — 9 parallel tasks (3 models × 3 datasets).
#
# All LLM backbone weights are unfrozen and trained (full fine-tuning).
# Evaluation: fixed ctx=512 for all horizons H={96,192,336,720}.
# Each task handles 1 (model, dataset) pair independently on 1 GPU.
#
# Task mapping (SLURM_ARRAY_TASK_ID):
#   0=baseline/ETTh1   1=baseline/ETTh2   2=baseline/weather
#   3=ts_only/ETTh1    4=ts_only/ETTh2    5=ts_only/weather
#   6=ts_lang/ETTh1    7=ts_lang/ETTh2    8=ts_lang/weather
#
#SBATCH --job-name=fullft_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=output_job/fullft_eval_%a.out
#SBATCH --error=output_job/fullft_eval_%a.err
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

PERM_HF="${HOME}/.cache/huggingface/hub"
for SNAP in "models--google--gemma-3-270m" "models--google--gemma-3-pt-270m"; do
  if [ -d "${PERM_HF}/${SNAP}" ]; then
    cp -r "${PERM_HF}/${SNAP}" "${HF_HUB_CACHE}/" 2>/dev/null || true
    echo "[cache] copied ${SNAP}"
  fi
done

# ── Task → (model, dataset) mapping ──────────────────────────────────────────
MODELS=("baseline" "ts_only_lora" "ts_lang_lora")
DATASETS=("ETTh1" "ETTh2" "weather")

MODEL_TAG="${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}"
DATASET="${DATASETS[$((SLURM_ARRAY_TASK_ID % 3))]}"

CKPT_BASELINE="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_seqrope/checkpoints/best_model.pt"
CKPT_TS_ONLY="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos/checkpoints/best_model.pt"
CKPT_TS_LANG="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"

case "${MODEL_TAG}" in
  baseline)    CKPT="${CKPT_BASELINE}"; EXTRA="--no_mvpos" ;;
  ts_only_lora) CKPT="${CKPT_TS_ONLY}"; EXTRA="--lora_r 4 --lora_alpha 8" ;;
  ts_lang_lora) CKPT="${CKPT_TS_LANG}"; EXTRA="--lora_r 4 --lora_alpha 8" ;;
esac

OUT_DIR="output_job/fullft_results"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/ft_fullft_${MODEL_TAG}_${DATASET}.json"

DATA_DIR="/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

echo "============================================================"
echo "Full Fine-tuning Eval  task=${SLURM_ARRAY_TASK_ID}"
echo "Model   : ${MODEL_TAG}"
echo "Dataset : ${DATASET}"
echo "Ckpt    : ${CKPT}"
echo "Output  : ${OUT_JSON}"
echo "============================================================"

${PYTHON} ${EVAL_SCRIPT} \
  --ckpt "${CKPT}" \
  --llm_id "google/gemma-3-270m" \
  --data_dir "${DATA_DIR}" \
  --datasets "${DATASET}" \
  --ft_ctx 512 \
  --ft_bs 4 \
  --ft_accum 8 \
  --ft_lr 1e-4 \
  --ft_min_lr 1e-5 \
  --llm_lr 1e-5 \
  --ft_stride 32 \
  --ft_epochs 1 \
  --eval_ctx 512 \
  --horizons 96 192 336 720 \
  --n_windows 500 \
  --seed 42 \
  --outer_bs 256 \
  --inner_bs 16 \
  --device cuda \
  --out_json "${OUT_JSON}" \
  --full_ft \
  ${EXTRA}

echo "Done → ${OUT_JSON}"
