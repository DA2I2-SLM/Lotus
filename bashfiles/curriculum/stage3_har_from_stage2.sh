#!/bin/bash
#SBATCH --job-name=curr_S3_har
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

# Curriculum Stage 3 (HAR CoT) — warm-start from stage2_captioning best.
# LoRA r=16/alpha=32 will be auto-initialized fresh (stages 1+2 had no LoRA), matching paper.

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

export USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export RESULTS_BASE_DIR="/scratch/${USER_NAME}/FoundationalTS/results"

# Warm-start from stage2_captioning best.
export WARM_START_PRETRAINED_CKPT="/scratch/${USER_NAME}/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/stage2_captioning_from_pretrained_freq/checkpoints/best_model.pt"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"

NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29513}"

echo "Curriculum Stage 3 (stage3_cot) — HAR CoT, LoRA r=16/alpha=32 (fresh init)"
echo "Warm-start from: ${WARM_START_PRETRAINED_CKPT}"

MAX_RESTARTS=10
for try in $(seq 1 ${MAX_RESTARTS}); do
  echo "[outer-retry] attempt ${try}/${MAX_RESTARTS}"
  set +e
  srun --export=ALL \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    bash -lc '
      set -euo pipefail
      HOST_SHORT="$(hostname -s)"
      export LOCAL_SCRATCH_BASE="/tmp/${USER}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"
      export HF_HOME="${LOCAL_SCRATCH_BASE}"
      export HF_DATASETS_CACHE="${HF_HOME}/datasets"
      export HF_HUB_CACHE="${HF_HOME}/hub"
      export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
      unset TRANSFORMERS_CACHE || true
      export HF_DATASETS_DISABLE_PROGRESS_BARS=1
      mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
      if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
        cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
        chmod 600 "${HF_HOME}/token" || true
      fi
      find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

      torchrun \
        --nnodes='"${NNODES}"' \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
        --rdzv_id='"${SLURM_JOB_ID}"'_'"${try}"' \
        --node_rank="${SLURM_NODEID}" \
        /home/'"${USER_NAME}"'/FoundationalTS/curriculum_learning.py \
          --model OpenTSLMSPFixPos2D \
          --stages stage3_cot \
          --llm_id google/gemma-3-270m \
          --num_epochs 20 \
          --pe_type sincos \
          --patch_size 32 \
          --enc_dec_type mlp \
          --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
          --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
          --use_freq_embed \
          --batch_size 2 \
          --val_batch_size 1 \
          --max_tokens_per_batch 1200 \
          --grad_accum_steps 8 \
          --save_every_n_steps 800 \
          --gradient_checkpointing \
          --warm_start_from pretrained
    '
  EXIT_CODE=$?
  set -e
  echo "[outer-retry] srun exit=${EXIT_CODE}"
  if [ ${EXIT_CODE} -eq 0 ]; then break; fi
  if [ ${try} -ge ${MAX_RESTARTS} ]; then exit ${EXIT_CODE}; fi
  sleep 10
done
