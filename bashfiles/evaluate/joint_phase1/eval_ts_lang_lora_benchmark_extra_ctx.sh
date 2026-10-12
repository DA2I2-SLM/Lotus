#!/bin/bash
# ETTh1/ETTh2/Weather benchmark eval — TS+Language LoRA, ctx=1024 and ctx=3072.
# Companion to eval_ts_lang_lora_benchmark.sh (which covers ctx=512/2048).
# Together they match the two-run setup used on the baseline model.
#
#SBATCH --job-name=eval_bm_ts_lang_lora_xc
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/eval_bm_ts_lang_lora_xc.out
#SBATCH --error=output_job/eval_bm_ts_lang_lora_xc.err

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any FTS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set FTS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
GEMMA_LOCAL="${FTS_GEMMA_LOCAL}"

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="${FTS_TSREASON_DIR}"

echo ""
echo "=========================================================="
echo " Benchmark eval (extra ctx) — TS+Language LoRA (joint_phase1_extend_balanced_lora_freq_mvpos)"
echo " ckpt     : ${CKPT}"
echo " contexts : 1024 3072  (512 and 2048 in companion job)"
echo " horizons : 96 192 336 720"
echo " datasets : ETTh1 ETTh2 weather"
echo " lora_r=4  lora_alpha=8"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/our_model_benchmark_eval.py \
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
    --lora_r           4 \
    --lora_alpha       8 \
    --out_json         output_job/benchmark_results_ts_lang_lora_extra_ctx.json
