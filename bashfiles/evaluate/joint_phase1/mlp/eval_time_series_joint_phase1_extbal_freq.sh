#!/bin/bash
# Time-series eval (forecasting) for the MLP joint_phase1_extend_balanced + freq
# run trained by bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth_freq.sh.
#
# Outputs:
#   eval_results/mlp_joint_phase1_extbal_freq/time_series/chronos_uv/
#   eval_results/mlp_joint_phase1_extbal_freq/time_series/synthetic/
#
#SBATCH --job-name=eval-ts-mlp-joint-extbal-freq
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

# Resolve repo root + interpreter + data locations dynamically so this
# script runs on any account. Override any FTS_* var in your environment.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -f "$_d/bashfiles/_paths.sh" ]; do _d="$(dirname "$_d")"; done
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

ENV_PY="$(pwd)/.env/bin/python"
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=${FTS_ETT_DIR}/ETT-small/ETTh1.csv

CKPT=${FTS_RESULTS_ROOT}/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt
STAGE=joint_phase1_extend_balanced
OUT_BASE=./eval_results/mlp_joint_phase1_extbal_freq/time_series

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
    --use_freq_embed \
    --out_dir           ${OUT} \
    --forecast_backbone llm \
    --eval_split        test \
    --source_filter     ${SRC} \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim     256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim     256
done
