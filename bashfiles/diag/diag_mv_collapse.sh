#!/bin/bash
# Diagnose where per-channel info is lost in MV forecast for
# the phase1_chronos_mse_freq checkpoint. Prints pairwise cosine
# similarity across channels at each stage:
#   encoder → projector → LLM hidden → decoder.
#
#SBATCH --job-name=diag_mv_collapse
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x.out
#SBATCH --error=output_job/%x.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

ENV_PY="$(pwd)/.env/bin/python"
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq/checkpoints/best_model.pt
CACHE=/scratch/s225250685/project/FoundationalTS/_preprocessed_chronos_uv_synth_phase1_mse_freq

echo "CKPT=${CKPT}"
echo "CACHE=${CACHE}"

${ENV_PY} scripts/diag_mv_collapse.py \
  --ckpt          ${CKPT} \
  --cache_dir     ${CACHE} \
  --llm_id        google/gemma-3-270m \
  --enc_dec_type  transformer \
  --use_freq_embed \
  --patch_size    32
