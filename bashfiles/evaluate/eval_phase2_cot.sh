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

# Evaluate a Phase-2 (joint_phase2_downstream) best_model.pt on the CoT
# downstream tasks: HAR, SleepEDF, ECG-QA, PAMAP2 and TimeMQA
# (classification accuracy / exact match + ROUGE-L).
#
# Environment overrides (all optional):
#   EVAL_CKPT       checkpoint to evaluate
#                   (default: the 270M "ours" run under $LOTUS_RESULTS_ROOT_joint5_ours)
#   EVAL_TASKS      comma-separated task list   (default: har,sleep,ecg,pamap2,timemqa)
#   EVAL_OUT_DIR    output directory            (default: $LOTUS_RESULTS_ROOT/eval_phase2_cot/<ckpt tag>)
#   EVAL_MAX_SAMPLES 0 = full test split        (default: 0)
#   EVAL_LLM_ID     backbone id or local snapshot (default: $LOTUS_GEMMA_LOCAL or google/gemma-3-270m)
#   EVAL_EXTRA      extra flags for evaluation/cot/eval_cot_phase2.py, e.g. for
#                   the 1B transformer-encoder run:
#                   "--enc_dec_type transformer --trans_enc_dim 192 --trans_enc_num_layers 6
#                    --trans_enc_num_heads 8 --trans_enc_ff_dim 896 --trans_dec_dim 192
#                    --trans_dec_ff_dim 896 --pe_base_c 10 --lora_r 8 --lora_alpha 16"
#                   or the input-corruption diagnostics (paper Fig. 5), which
#                   corrupt only the array the encoder sees (caption text with the
#                   raw mean/std is unchanged); the mode is recorded in summary.json:
#                   "--corrupt shuffle" | "--corrupt noise" | "--corrupt zero"
#                   (+ "--corrupt_seed N", default 0)
#
#   EVAL_TASKS=har,sleep sbatch bashfiles/evaluate/eval_phase2_cot.sh
#   EVAL_TASKS=har,sleep EVAL_OUT_DIR=$LOTUS_RESULTS_ROOT/eval_phase2_cot/ours_shuffle \
#     EVAL_EXTRA="--corrupt shuffle" sbatch bashfiles/evaluate/eval_phase2_cot.sh

set -euo pipefail
mkdir -p output_job

cd "${LOTUS_REPO_ROOT}"

export TOKENIZERS_PARALLELISM=false

CKPT="${EVAL_CKPT:-${LOTUS_RESULTS_ROOT}_joint5_ours/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_freq/checkpoints/best_model.pt}"
TASKS="${EVAL_TASKS:-har,sleep,ecg,pamap2,timemqa}"
LLM_ID="${EVAL_LLM_ID:-${LOTUS_GEMMA_LOCAL:-google/gemma-3-270m}}"
MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
# default out dir is keyed by the stage dir name (…/<stage>/checkpoints/best_model.pt)
_stage="$(basename "$(dirname "$(dirname "${CKPT}")")")"
OUT_DIR="${EVAL_OUT_DIR:-${LOTUS_RESULTS_ROOT}/eval_phase2_cot/${_stage}}"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (set EVAL_CKPT)" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

echo "Checkpoint : ${CKPT}"
echo "Tasks      : ${TASKS}"
echo "LLM        : ${LLM_ID}"
echo "Out dir    : ${OUT_DIR}"

# The freq flag is inferred from the stage dir name (…_freq vs …_nofreq); set
# EVAL_FREQ_FLAG explicitly to override.
case "${CKPT}" in
  *_nofreq/*) _freq_default="" ;;
  *)          _freq_default="--use_freq_embed" ;;
esac
FREQ_FLAG="${EVAL_FREQ_FLAG-${_freq_default}}"

# shellcheck disable=SC2086
"${LOTUS_PYTHON}" evaluation/cot/eval_cot_phase2.py \
  --ckpt           "${CKPT}" \
  --llm_id         "${LLM_ID}" \
  --tasks          ${TASKS//,/ } \
  --split          test \
  --max_samples    "${MAX_SAMPLES}" \
  --max_new_tokens 1024 \
  --out_dir        "${OUT_DIR}" \
  ${FREQ_FLAG} ${EVAL_EXTRA:-}
echo "EVAL_DONE ${TASKS} -> ${OUT_DIR}"
