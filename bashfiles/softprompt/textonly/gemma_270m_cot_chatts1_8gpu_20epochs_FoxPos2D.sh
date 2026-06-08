#!/bin/bash
#SBATCH --job-name=nope_chatts
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

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

# -----------------------
# Run tag (optional)
# -----------------------
TAG="${1:-run}"

# -----------------------
# Runtime settings
# -----------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# How many GPUs on this node?
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
fi

echo "Nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"

# -----------------------
# Launch
# -----------------------
# Single node: use --standalone so torchrun picks a free port (fixes EADDRINUSE)
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  curriculum_learning.py \
    --model PlainTextLM \
    --stages plaintext_lm \
    --llm_id google/gemma-3-270m \
    --num_epochs 20 \
    --pe_type plain_text \
    --is_pretraining False