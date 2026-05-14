#!/bin/bash
#SBATCH --job-name=flamingo-mn
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]


set -euo pipefail
mkdir -p output_job

# -----------------------
# Conda env
# -----------------------
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

TAG="${1:-run}"

# -----------------------
# Runtime settings
# -----------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid loopback / virtual interfaces being selected
unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"

# -----------------------
# Distributed rendezvous
# -----------------------
NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"

# FORCE 1 GPU per node
GPUS_PER_NODE=1

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"

# -----------------------
# Launch (multi-node, 2 GPU/node)
# -----------------------
srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -lc '
    set -euo pipefail
    echo "[$(hostname)] SLURM_NODEID=${SLURM_NODEID} starting torchrun (2 GPU/node)"

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=2 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      /home/${USER_NAME}/FoundationalTS/curriculum_learning.py \
        --model OpenTSLMFlamingo \
        --stages foundation_stage1_chatts \
        --llm_id google/gemma-3-270m \
        --num_epochs 20 \
        --pe_type nope
  '
