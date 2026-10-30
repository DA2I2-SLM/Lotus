#!/bin/bash
# Time-series eval (forecasting) for the TRANSFORMER phase1_chronos_mse + freq
# run trained by
# bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_freq.sh.
#
# Outputs:
#   eval_results/trans_phase1_chronos_mse_freq/time_series/chronos_uv/
#   eval_results/trans_phase1_chronos_mse_freq/time_series/synthetic/
#
#SBATCH --job-name=eval-ts-trans-phase1-chronos-mse-freq
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

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any LOTUS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${LOTUS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set LOTUS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job


cd "${LOTUS_REPO_ROOT}"

ENV_PY="${LOTUS_PYTHON}"
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=${LOTUS_ETT_DIR}/ETT-small/ETTh1.csv

CKPT=${LOTUS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq/checkpoints/best_model.pt
STAGE=phase1_chronos_mse
OUT_BASE=./eval_results/trans_phase1_chronos_mse_freq/time_series

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
    --use_freq_embed \
    --out_dir           ${OUT} \
    --forecast_backbone llm \
    --eval_split        test \
    --source_filter     ${SRC}
done
