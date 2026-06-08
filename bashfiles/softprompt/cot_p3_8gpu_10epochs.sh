#!/bin/bash
#SBATCH --job-name=tsqa
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
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
# Use whichever works on your cluster:
# source /etc/profile.d/conda.sh
source activate
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/OpenTSLM

# -----------------------
# Run tag (optional)
# -----------------------
TAG="${1:-run}"

# If you want the tag in filenames, redirect here (SBATCH output stays too):
# exec > "output_job/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TAG}.out" \
#   2> "output_job/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TAG}.err"

# -----------------------
# Distributed settings
# -----------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=29500

# How many GPUs per node?
# Works on many clusters; fallback to nvidia-smi count.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
fi

echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID} Tag: ${TAG}"

# -----------------------
# Launch (single node or multi node)
torchrun --nproc_per_node=8 curriculum_learning.py --model OpenTSLMSP --stages stage3_cot --llm_id OpenTSLM/llama-3.2-1b-m4-sp --num_epochs 10