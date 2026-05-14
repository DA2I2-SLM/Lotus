#!/bin/bash
#SBATCH --job-name=tsqa-eval
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# -----------------------
# Models to evaluate
# -----------------------
models=(
    "google/gemma-3-270m"
    # "google/gemma-3-1b-pt"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
)

EVAL_SCRIPT="/home/${USER_NAME}/FoundationalTS/evaluation/baseline/evaluate_tsqa.py"

echo "Job: ${SLURM_JOB_ID:-no_slurm}  Tag: ${TAG}"
echo "GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Models to evaluate: ${#models[@]}"

# -----------------------
# Evaluate each model directly (no srun needed for single-node sbatch)
# device_map=auto in Python will use all allocated GPUs automatically
# -----------------------
echo "Starting TSQA evaluations..."
for model in "${models[@]}"; do
    echo "─────────────────────────────────────────────────────────"
    echo "Evaluating: ${model}"
    echo "─────────────────────────────────────────────────────────"
    python "${EVAL_SCRIPT}" "${model}"
done

# echo "Starting HAR evaluations..."
# for model in "${models[@]}"; do
#     python /home/${USER_NAME}/FoundationalTS/evaluation/baseline/evaluate_har.py "${model}"
# done

# echo "Starting ChatTS evaluations..."
# for model in "${models[@]}"; do
#     python /home/${USER_NAME}/FoundationalTS/evaluation/baseline/evaluate_chatts.py "${model}"
# done

echo "All evaluations completed!  Job: ${SLURM_JOB_ID:-no_slurm}"