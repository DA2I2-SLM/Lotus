#!/bin/bash
#SBATCH --job-name=debug_one_window
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export TOKENIZERS_PARALLELISM=false

CKPT="/scratch/${USER_NAME}/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt"
CSV="/scratch/${USER_NAME}/TimeSeriesECM/dataset/ETTh1.csv"

python evaluation/forecasting/debug_one_window.py \
  --ckpt "${CKPT}" \
  --csv  "${CSV}" \
  --pred_len 96 \
  --hist_len 512 \
  --patch_size 32 \
  --window_idx 0 \
  --channel 0 \
  --show_n 30
