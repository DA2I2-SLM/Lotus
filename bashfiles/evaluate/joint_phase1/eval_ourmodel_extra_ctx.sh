#!/bin/bash
# Our model benchmark with paper-default contexts 1024 and 3072.
# (ctx=512 and ctx=2048 already done in prior run.)
# Matches TimeMoE paper protocol: H=192→ctx=1024, H=720→ctx=3072, etc.
#
#SBATCH --job-name=our_model_extra_ctx
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/our_model_extra_ctx.out
#SBATCH --error=output_job/our_model_extra_ctx.err

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
mkdir -p ${LOTUS_REPO_ROOT}/output_job


cd "${LOTUS_REPO_ROOT}"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
GEMMA_LOCAL="${LOTUS_GEMMA_LOCAL}"

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="${LOTUS_TSREASON_DIR}"

echo ""
echo "=========================================================="
echo " Benchmark evaluation — OpenTSLMSPFixPos2D"
echo " ckpt     : ${CKPT}"
echo " contexts : 1024 3072  (512 and 2048 already done)"
echo " horizons : 96 192 336 720"
echo " datasets : ETTh1 ETTh2 weather"
echo "=========================================================="
echo ""

nvidia-smi -L || true

"${LOTUS_PYTHON}" evaluation/forecasting/our_model_benchmark_eval.py \
    --ckpt             "${CKPT}" \
    --llm_id           "${GEMMA_LOCAL}" \
    --data_dir         "${DATA_DIR}" \
    --datasets         ETTh1 ETTh2 weather \
    --context_lengths  1024 3072 \
    --horizons         96 192 336 720 \
    --n_eval_windows   500 \
    --seed             42 \
    --device           cuda \
    --outer_bs         256 \
    --inner_bs         16 \
    --out_json         output_job/benchmark_results_extra_ctx.json
