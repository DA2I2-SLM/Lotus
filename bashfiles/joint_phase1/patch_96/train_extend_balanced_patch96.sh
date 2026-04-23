#!/bin/bash
#SBATCH --job-name=joint_extended_balanced_patch96_8g
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/pretrain_%j.out
#SBATCH --error=output_job/pretrain_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
cd /scratch/s225250685/project/time-series-foundation-model/FoundationalTS
conda activate ./.env

TAG="${1:-run}"
export PYTHONPATH="/scratch/s225250685/project/time-series-foundation-model/FoundationalTS/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# BLOCKING_WAIT=1 causes silent infinite hangs on NCCL errors; keep it off
# so failures surface as proper Python exceptions with tracebacks.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"

NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((SLURM_JOB_ID % 10000 + 20000))}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

# srun: --gres must match #SBATCH --gres (gpu:2) and --nproc_per_node below
srun --export=ALL --kill-on-bad-exit=1 \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --cpu-bind=none \
  --gres=gpu:2 \
  bash -lc '
    set -euo pipefail

    # Per-node log files so each rank's stdout/stderr is independently visible.
    NODE_OUT="output_job/pretrain_${SLURM_JOB_ID}_$(hostname -s).out"
    NODE_ERR="output_job/pretrain_${SLURM_JOB_ID}_$(hostname -s).err"
    exec > >(tee -a "${NODE_OUT}") 2> >(tee -a "${NODE_ERR}" >&2)

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
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages joint_phase1_extend_balanced \
        --llm_id google/gemma-3-270m \
        --num_epochs 20 \
        --pe_type sincos \
        --patch_size 96 \
        --val_batch_size 4 \
        --grad_accum_steps 8
  '
