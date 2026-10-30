#!/bin/bash
# ETTh1/ETTh2/Weather benchmark eval — TS+Language LoRA model (joint_phase1_extend_balanced_lora_freq_mvpos).
# Matches Time-MoE run_eval.py protocol (StandardScaler, channel-independent, ctx=512/2048).
#
#SBATCH --job-name=eval_bm_ts_lang_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/eval_bm_ts_lang_lora.out
#SBATCH --error=output_job/eval_bm_ts_lang_lora.err

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

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=0
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

PERM_HF="${HOME}/.cache/huggingface/hub"
for MODEL_SNAP in "models--google--gemma-3-270m"; do
  if [ -d "${PERM_HF}/${MODEL_SNAP}" ]; then
    cp -r "${PERM_HF}/${MODEL_SNAP}" "${HF_HUB_CACHE}/" || true
    echo "[setup] copied ${MODEL_SNAP} from permanent cache"
  fi
done

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="${LOTUS_TSREASON_DIR}"

echo ""
echo "=========================================================="
echo " Benchmark eval — TS+Language LoRA (joint_phase1_extend_balanced_lora_freq_mvpos)"
echo " ckpt     : ${CKPT}"
echo " data_dir : ${DATA_DIR}"
echo " datasets : ETTh1 ETTh2 weather  ctx: 512 2048  H: 96 192 336 720"
echo " lora_r=4  lora_alpha=8"
echo "=========================================================="
echo ""

nvidia-smi -L || true

"${LOTUS_PYTHON}" evaluation/forecasting/our_model_benchmark_eval.py \
    --ckpt "${CKPT}" \
    --llm_id "google/gemma-3-270m" \
    --data_dir "${DATA_DIR}" \
    --datasets ETTh1 ETTh2 weather \
    --context_lengths 512 2048 \
    --horizons 96 192 336 720 \
    --n_eval_windows 500 \
    --seed 42 \
    --device cuda \
    --outer_bs 256 \
    --inner_bs 16 \
    --lora_r 4 \
    --lora_alpha 8 \
    --out_json output_job/benchmark_results_ts_lang_lora.json
