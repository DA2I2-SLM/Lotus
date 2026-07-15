#!/bin/bash
#SBATCH --job-name=eval_p2_cot
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

# Evaluate phase-2 best_model.pt on stage 3/4/5 CoT datasets
# (HAR + SleepEDF + ECG-QA) — classification accuracy + ROUGE-L.

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export TOKENIZERS_PARALLELISM=false

CKPT="/scratch/${USER_NAME}/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/stage3_cot_from_pretrained_nofreq/checkpoints/best_model.pt"
OUT_DIR="/scratch/${USER_NAME}/FoundationalTS/eval_results/phase2_cot"

echo "Checkpoint: ${CKPT}"
echo "Out dir   : ${OUT_DIR}"

# Pass --max_samples 0 for the full test split (slow, hours).
python evaluation/cot/eval_cot_phase2.py \
  --ckpt           "${CKPT}" \
  --llm_id         google/gemma-3-270m \
  --tasks          har \
  --split          test \
  --max_samples    0 \
  --max_new_tokens 1024 \
  --out_dir        "${OUT_DIR}"
