#!/bin/bash
#SBATCH --job-name=p2_harsleep_txl
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

export USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

TAG="${1:-run}"

# ─────────────────────────────────────────────────────────────────────────────
# Warm-start from the TEXT-ONLY phase-1 LoRA checkpoint (TSQA+M4, CE-only, LoRA r4/a8).
# --warm_start_from pretrained loads WARM_START_PRETRAINED_CKPT directly. Because that
# ckpt has lora_enabled=True, _load_weights_from_stage enables LoRA at its saved rank
# (r4/a8) BEFORE loading, so phase-2 CONTINUES the phase-1 adapter (LoRA on).
# This differs from the sweep runs (raw/bal/sqrt3), which warm-start from
# joint_phase1_extend_balanced_lora_nofreq — here we isolate the starting point.
# ─────────────────────────────────────────────────────────────────────────────
export WARM_START_PRETRAINED_CKPT=/scratch/s223540177/FoundationalTS/results_tsqam4_lora/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4_nofreq/checkpoints/best_model.pt

export RESULTS_BASE_DIR="/scratch/s223540177/FoundationalTS/results_harsleep_txlora"

# HAR+Sleep-only phase-2 (no forecasting, no ChatTS)
export PHASE2_COT_ONLY=1
# (no SLEEP_UPSAMPLE -> natural sizes: HAR 68.5k, Sleep 7.4k; mirrors the unbalanced run)


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
MASTER_PORT="${MASTER_PORT:-29514}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"
echo "Warm start: --warm_start_from pretrained -> ${WARM_START_PRETRAINED_CKPT}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

# Outer retry loop: each iteration is a fresh srun -> bash -> torchrun
# launch with fresh c10d rendezvous. Python exits 42 on memory-leak restart;
# we re-launch the whole srun stack until python exits 0 (training done).
MAX_RESTARTS=30
for try in $(seq 1 ${MAX_RESTARTS}); do
  echo "[outer-retry] attempt ${try}/${MAX_RESTARTS}"
  set +e
  srun --export=ALL \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    bash -lc '
      set -euo pipefail

      echo "[$(hostname)] SLURM_NODEID=${SLURM_NODEID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      nvidia-smi -L || true

      HOST_SHORT="$(hostname -s)"
      export LOCAL_SCRATCH_BASE="/tmp/${USER}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"

      export HF_HOME="${LOCAL_SCRATCH_BASE}"
      export HF_DATASETS_CACHE="${HF_HOME}/datasets"
      export HF_HUB_CACHE="${HF_HOME}/hub"
      export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

      unset TRANSFORMERS_CACHE || true

      export HF_DATASETS_DISABLE_PROGRESS_BARS=1

      mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

      echo "[$(hostname)] HF_HOME=${HF_HOME}"
      echo "[$(hostname)] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
      echo "[$(hostname)] HF_HUB_CACHE=${HF_HUB_CACHE}"
      df -h /tmp || true

      if [ -n "${HF_TOKEN:-}" ]; then
        echo "[$(hostname)] Using HF_TOKEN from environment"
      else
        echo "[$(hostname)] HF_TOKEN not set, trying token file fallback"

        if [ -f "${HOME}/.cache/huggingface/token" ]; then
          cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
          chmod 600 "${HF_HOME}/token" || true
          echo "[$(hostname)] Copied token from ~/.cache/huggingface/token"
        elif [ -f "${HOME}/.huggingface/token" ]; then
          cp "${HOME}/.huggingface/token" "${HF_HOME}/token"
          chmod 600 "${HF_HOME}/token" || true
          echo "[$(hostname)] Copied token from ~/.huggingface/token"
        else
          echo "[$(hostname)] No saved HF token file found"
        fi
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
          --stages joint_phase2_downstream \
          --llm_id google/gemma-3-270m \
          --num_epochs 20 \
          --pe_type sincos \
          --patch_size 32 \
          --enc_dec_type mlp \
          --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
          --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
          --batch_size 2 \
          --val_batch_size 1 \
          --max_tokens_per_batch 1200 \
          --grad_accum_steps 8 \
          --mse_alpha 1.0 \
          --windows_per_col_train 0 \
          --dataloader_workers 0 \
          --save_every_n_steps 800 \
          --gradient_checkpointing \
          --warm_start_from pretrained
    '
  EXIT_CODE=$?
  set -e
  echo "[outer-retry] srun exit=${EXIT_CODE}"
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[outer-retry] training completed cleanly"
    break
  fi
  if [ ${try} -ge ${MAX_RESTARTS} ]; then
    echo "[outer-retry] hit max restarts; exiting ${EXIT_CODE}"
    exit ${EXIT_CODE}
  fi
  echo "[outer-retry] restarting after exit=${EXIT_CODE}"
  sleep 10
done
