#!/bin/bash
#SBATCH --job-name=eval_opentslm_off_sleep
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

# Evaluate official OpenTSLM/gemma-3-270m-sleep-sp checkpoint on HAR + Sleep CoT.

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export TOKENIZERS_PARALLELISM=false

OUT_DIR="/scratch/${USER_NAME}/FoundationalTS/eval_results/opentslm_official_sleep_sp"

echo "Repo     : OpenTSLM/gemma-3-270m-sleep-sp"
echo "Out dir  : ${OUT_DIR}"

python evaluation/cot/eval_opentslm_official.py \
  --repo_id        OpenTSLM/gemma-3-270m-sleep-sp \
  --tasks          har sleep \
  --split          test \
  --max_samples    0 \
  --max_new_tokens 1024 \
  --batch_size     8 \
  --out_dir        "${OUT_DIR}"
