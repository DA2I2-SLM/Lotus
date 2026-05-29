#!/bin/bash
# Time-series eval (forecasting) for the TRANSFORMER phase1_chronos_mse run
# trained by bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_nofreq.sh.
#
# transformer is the default --enc_dec_type so no _mlp suffix in results_dir.
#
# Outputs:
#   eval_results/trans_phase1_chronos_mse_nofreq/time_series/chronos_uv/
#   eval_results/trans_phase1_chronos_mse_nofreq/time_series/synthetic/
#
#SBATCH --job-name=eval-ts-trans-phase1-chronos-mse-nofreq
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
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=/scratch/s225250685/project/time-series-foundation-model/dataset/ETT-small/ETTh1.csv

CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_nofreq/checkpoints/best_model.pt
STAGE=phase1_chronos_mse
OUT_BASE=./eval_results/trans_phase1_chronos_mse_nofreq/time_series

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
    --enc_dec_type      transformer \
    --out_dir           ${OUT} \
    --forecast_backbone llm \
    --eval_split        test \
    --source_filter     ${SRC}
done
