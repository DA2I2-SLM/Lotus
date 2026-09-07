#!/bin/bash
# Re-run weather eval for 1B model — zero-shot and fine-tune.
# 2 array tasks, 24h limit, inner_bs=16 for speed.
#
# Task 0 = zero-shot/weather
# Task 1 = finetune/weather
#
#SBATCH --job-name=eval_1b_weather
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=output_job/eval_1b_weather_%a.out
#SBATCH --error=output_job/eval_1b_weather_%a.err
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

CKPT="results/sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="/vast/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"
EVAL_SCRIPT="evaluation/forecasting/finetune_eval.py"
PYTHON="./.env/bin/python"

OUT_DIR="output_job/eval_1b_results"
mkdir -p "${OUT_DIR}"

COMMON_ARGS="
  --ckpt ${CKPT}
  --llm_id google/gemma-3-1b-pt
  --data_dir ${DATA_DIR}
  --datasets weather
  --eval_ctx 512 1024 2048 3072
  --horizons 96 192 336 720
  --n_windows 500
  --seed 42
  --outer_bs 128
  --inner_bs 16
  --device cuda
"

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  echo "Mode: zero-shot / weather"
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} \
    --out_json "${OUT_DIR}/eval_1b_zeroshot_weather.json" \
    --zero_shot
else
  echo "Mode: finetune / weather"
  ${PYTHON} ${EVAL_SCRIPT} ${COMMON_ARGS} \
    --out_json "${OUT_DIR}/eval_1b_finetune_weather.json" \
    --ft_ctx 512 --ft_bs 4 --ft_accum 8 \
    --ft_lr 1e-4 --ft_min_lr 1e-5 --ft_stride 32 --ft_epochs 1
fi
