#!/bin/bash
# Time-series eval (forecasting) for the MLP phase1_chronos_mse_lora run trained by
# bashfiles/joint_phase1/mlp/train_phase1_chronos_mse_nofreq_lora.sh.
#
# Mirrors eval_time_series_phase1_chronos_mse_nofreq.sh but loads the LoRA-adapted
# checkpoint (r=4, alpha=8) from phase1_chronos_mse_lora_nofreq/.
#
# Outputs:
#   eval_results/mlp_phase1_chronos_mse_nofreq_lora/time_series/chronos_uv/
#   eval_results/mlp_phase1_chronos_mse_nofreq_lora/time_series/synthetic/
#
#SBATCH --job-name=eval-ts-mlp-phase1-chronos-mse-nofreq-lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
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

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=/scratch/s225250685/project/time-series-foundation-model/dataset/ETT-small/ETTh1.csv

CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_nofreq/checkpoints/best_model.pt
STAGE=phase1_chronos_mse_lora
OUT_BASE=./eval_results/mlp_phase1_chronos_mse_nofreq_lora/time_series

echo "CHRONOS_PREPROCESSED_SPLITS_DIR=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "CKPT=${CKPT}"

for SRC in chronos_uv synthetic; do
  OUT=${OUT_BASE}/${SRC}
  echo "================================================================"
  echo "[eval_time_series] source=${SRC} -> ${OUT}"
  echo "================================================================"
  ${ENV_PY} ${INFER} \
    --ckpt              ${CKPT} \
    --llm_id            ${LLM} \
    --stage_name        ${STAGE} \
    --eval_format       phase1_pretrain \
    --etth1_csv         ${ETTh1} \
    --mode              autoregressive \
    --device            cuda \
    --batch_size        32 \
    --patch_size        32 \
    --history_len       576 \
    --pred_len          192 \
    --plot_every        50 \
    --enc_dec_type      mlp \
    --out_dir           ${OUT} \
    --forecast_backbone llm \
    --eval_split        test \
    --source_filter     ${SRC} \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim     256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim     256 \
    --lora_r            4 \
    --lora_alpha        8
done
