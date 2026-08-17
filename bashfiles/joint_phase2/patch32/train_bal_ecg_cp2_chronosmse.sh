#!/bin/bash
#SBATCH --job-name=p2bal_cp2_chrmse
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
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
# Warm-start loads the pretrained checkpoint DIRECTLY via `--warm_start_from
# pretrained` (see curriculum_learning.py). The path it loads is
# PRETRAINED_WARM_START_CKPT there:
#   /scratch/$USER/FoundationalTS/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt
# To point at a different file, uncomment + edit the export below (no symlink,
# no results/<stage> convention path involved):
# export WARM_START_PRETRAINED_CKPT=/scratch/s223540177/FoundationalTS/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt
# ─────────────────────────────────────────────────────────────────────────────

export RESULTS_BASE_DIR="/scratch/s223540177/FoundationalTS/results_balecg_cp2_chronosmse"
export WARM_START_PRETRAINED_CKPT=/scratch/s223540177/FoundationalTS/pretrained_weights/phase1_chronos_mse_full_nofreq/checkpoints/best_model.pt

# HAR+Sleep-only phase-2 (no forecasting, no ChatTS)
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_COT_MIX=balanced   # HAR = Sleep = ECG (1:1:1)
export PHASE2_SELECT_BY_GEN=1   # new per-task generation-EM early stop


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
MASTER_PORT="${MASTER_PORT:-29511}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"
echo "Warm start: --warm_start_from pretrained (PRETRAINED_WARM_START_CKPT)"

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
  # Disable set -e around srun: srun returns non-zero when python exits 42
  # for the mem-restart workaround, and we want the loop to handle it, not
  # have the script abort.
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
        --nproc_per_node=4 \
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
          --batch_size 1 \
          --val_batch_size 1 \
          --max_tokens_per_batch 600 \
          --grad_accum_steps 16 \
          --mse_alpha 1.0 \
          --windows_per_col_train 0 \
          --dataloader_workers 0 \
          --save_every_n_steps 800 \
          --gradient_checkpointing \
          --warm_start_from pretrained
    '
  EXIT_CODE=$?
  set -e  # re-enable for safety after srun
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
