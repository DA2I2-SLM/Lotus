#!/bin/bash
#SBATCH --job-name=diagnose-phase1-found
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source activate
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

LLM_ID="${LLM_ID:-google/gemma-3-270m}"
CKPT="${CKPT:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
HISTORY_LEN="${HISTORY_LEN:-480}"
PATCH_SIZE="${PATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-1000}"
LR="${LR:-2e-4}"
SPLIT="${SPLIT:-train}"
WINDOW_POSITION="${WINDOW_POSITION:-tail}"
OUT_DIR="${OUT_DIR:-./eval_results/diagnose_phase1_foundational_overfit/patch${PATCH_SIZE}_n${NUM_SAMPLES}}"

CMD=(
  python /home/${USER_NAME}/FoundationalTS/evaluation/forecasting/diagnose_phase1_foundational_overfit.py
  --llm_id "${LLM_ID}"
  --device cuda
  --split "${SPLIT}"
  --num_samples "${NUM_SAMPLES}"
  --history_len "${HISTORY_LEN}"
  --patch_size "${PATCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --lr "${LR}"
  --window_position "${WINDOW_POSITION}"
  --out_dir "${OUT_DIR}"
  --pe_type sincos
)

if [ -n "${CKPT}" ]; then
  CMD+=(--ckpt "${CKPT}")
fi

echo "Launching foundational Phase-1 overfit diagnostic:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
