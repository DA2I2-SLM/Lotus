#!/bin/bash
# Fine-tune + evaluate TS+Lang LoRA checkpoint only (baseline + ts_only_lora already done).
#
#SBATCH --job-name=ft_eval_tslang
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/ft_eval_tslang.out
#SBATCH --error=output_job/ft_eval_tslang.err
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
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
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

DATA_DIR="/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

CKPT_TS_LANG="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"
OUT_DIR="output_job/finetune_results"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Fine-tune + Evaluate — TS+Lang LoRA only"
echo "Datasets : ETTh1, ETTh2, weather"
echo "Horizons : 96 192 336 720"
echo "Eval ctx : 512 1024 2048 3072"
echo "============================================================"

${PYTHON} ${EVAL_SCRIPT} \
  --ckpt "${CKPT_TS_LANG}" \
  --llm_id "google/gemma-3-270m" \
  --data_dir "${DATA_DIR}" \
  --datasets ETTh1 ETTh2 weather \
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
  --out_json "${OUT_DIR}/ft_results_ts_lang_lora.json" \
  --lora_r 4 --lora_alpha 8

echo "Saved → ${OUT_DIR}/ft_results_ts_lang_lora.json"
echo "============================================================"
