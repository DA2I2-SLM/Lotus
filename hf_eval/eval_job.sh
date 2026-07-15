#!/bin/bash
#SBATCH --job-name=hfeval
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s

# Args: $1 = HuggingFace repo id, $2 = short tag (for out_dir)
REPO_ID=${1:?need repo_id}
TAG=${2:?need tag}

module load Anaconda3
source activate
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

OUT_DIR=/scratch/${USER_NAME}/FoundationalTS/hf_eval_results/${TAG}
mkdir -p "${OUT_DIR}"

echo "=== node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u hf_eval/run_eval.py \
  --repo_id "${REPO_ID}" \
  --tasks ${TASKS:-sleep ecg har} \
  --max_samples "${MAX_SAMPLES:-2000}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-400}" \
  --patch_size "${PATCH_SIZE_OVERRIDE:-0}" \
  --attn_impl "${ATTN_IMPL:-eager}" \
  --llm_dtype "${LLM_DTYPE:-bf16}" \
  --out_dir "${OUT_DIR}"

echo "=== eval_job finished for ${REPO_ID} ==="
