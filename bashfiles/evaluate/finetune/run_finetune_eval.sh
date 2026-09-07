#!/bin/bash
# Fine-tune + evaluate OpenTSLMSPFixPos2D on ETTh1/ETTh2/Weather (Time-MoE protocol).
#
# Runs 3 checkpoints × 3 datasets = 9 combinations sequentially on 1 GPU.
# Fine-tune: 1 epoch, AdamW lr=1e-4, ctx=512, stride=32.
# Evaluate : ctx=[512,1024,2048,3072], H=[96,192,336,720], n_windows=500.
#
#SBATCH --job-name=finetune_eval_timemoe
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/finetune_eval_timemoe.out
#SBATCH --error=output_job/finetune_eval_timemoe.err
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

# Shared scratch HF cache (avoid re-downloading Gemma weights)
HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

# Copy Gemma-3-270M from persistent cache to avoid repeated downloads
PERM_HF="${HOME}/.cache/huggingface/hub"
for SNAP in "models--google--gemma-3-270m" "models--google--gemma-3-pt-270m"; do
  if [ -d "${PERM_HF}/${SNAP}" ]; then
    cp -r "${PERM_HF}/${SNAP}" "${HF_HUB_CACHE}/" 2>/dev/null || true
    echo "[cache] copied ${SNAP}"
  fi
done

DATA_DIR="/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

CKPT_BASELINE="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_seqrope/checkpoints/best_model.pt"
CKPT_TS_ONLY="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos/checkpoints/best_model.pt"
CKPT_TS_LANG="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"

OUT_DIR="output_job/finetune_results"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Fine-tune + Evaluate  (Time-MoE in-distribution protocol)"
echo "Datasets : ETTh1, ETTh2, weather"
echo "Horizons : 96 192 336 720"
echo "Eval ctx : 512 1024 2048 3072"
echo "Data dir : ${DATA_DIR}"
echo "============================================================"

run_ft_eval() {
  local CKPT="$1"
  local TAG="$2"
  local EXTRA_ARGS="${3:-}"
  local DATASETS="${4:-ETTh1 ETTh2 weather}"

  echo ""
  echo ">>> Model: ${TAG}"
  echo "    CKPT : ${CKPT}"

  ${PYTHON} ${EVAL_SCRIPT} \
    --ckpt "${CKPT}" \
    --llm_id "google/gemma-3-270m" \
    --data_dir "${DATA_DIR}" \
    --datasets ${DATASETS} \
    --ft_ctx 512 \
    --ft_bs 8 \
    --ft_accum 4 \
    --ft_lr 1e-4 \
    --ft_min_lr 1e-5 \
    --ft_stride 32 \
    --ft_epochs 1 \
    --eval_ctx 512 1024 2048 3072 \
    --horizons 96 192 336 720 \
    --n_windows 500 \
    --seed 42 \
    --outer_bs 256 \
    --inner_bs 16 \
    --device cuda \
    --out_json "${OUT_DIR}/ft_results_${TAG}.json" \
    ${EXTRA_ARGS}

  echo "    Saved → ${OUT_DIR}/ft_results_${TAG}.json"
}

# ── 1. Baseline (no LoRA, seqrope, no mvpos) ──────────────────────────────
run_ft_eval "${CKPT_BASELINE}" "baseline" "--no_mvpos"

# ── 2. TS-only LoRA ────────────────────────────────────────────────────────
run_ft_eval "${CKPT_TS_ONLY}" "ts_only_lora" "--lora_r 4 --lora_alpha 8"

# ── 3. TS+Lang LoRA ────────────────────────────────────────────────────────
run_ft_eval "${CKPT_TS_LANG}" "ts_lang_lora" "--lora_r 4 --lora_alpha 8"

echo ""
echo "============================================================"
echo "All fine-tune+eval jobs complete."
echo "Results in ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
echo "============================================================"
