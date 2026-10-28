#!/bin/bash
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-License-Identifier: MIT
#SBATCH --job-name=eval_tsrbench
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

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

# Zero-shot TSRBench (arXiv:2601.18744) evaluation of a Phase-2 checkpoint.
# See evaluation/tsrbench/README.md for the data download, the prompt
# protocol and how to score with the benchmark's own scorer.
#
# Environment overrides (all optional unless marked):
#   EVAL_CKPT            checkpoint to evaluate
#                        (default: the 270M "ours" run under ${FTS_RESULTS_ROOT}_joint5_ours)
#   TSRBENCH_DIR         REQUIRED. TSRBench root holding <category>/<subset>.jsonl
#                        (a `huggingface_hub.snapshot_download` of umd-zhou-lab/TSRBench)
#   EVAL_LLM_ID          backbone id or local snapshot (default: $FTS_GEMMA_LOCAL or google/gemma-3-270m)
#   EVAL_OUT_DIR         output directory  (default: $FTS_RESULTS_ROOT/eval_tsrbench/<stage dir name>)
#   EVAL_SUBSETS         comma list of subset file stems (default: all)
#   EVAL_MAX_NEW_TOKENS  generation cap    (default: 2048, the benchmark prompt's own budget)
#   EVAL_LIMIT           first N rows per subset, 0 = all (default: 0)
#   EVAL_BATCH_SIZE      (default: 1)
#   EVAL_PERCEPTION_MODE inplace | append  (default: inplace; see README "Caveats")
#   EVAL_FREQ_FLAG       "--use_freq_embed" or ""; inferred from the stage dir name (…_freq / …_nofreq)
#   EVAL_EXTRA           extra flags for evaluation/tsrbench/run_tsrbench.py, e.g. the
#                        1B transformer-encoder geometry (same flags as eval_phase2_cot.sh),
#                        or "--resume" to continue an interrupted job.
#
#   TSRBENCH_DIR=/path/to/TSRBench sbatch bashfiles/evaluate/eval_tsrbench.sh

set -euo pipefail
mkdir -p output_job

cd "${FTS_REPO_ROOT}"

export TOKENIZERS_PARALLELISM=false

CKPT="${EVAL_CKPT:-${FTS_RESULTS_ROOT}_joint5_ours/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_freq/checkpoints/best_model.pt}"
LLM_ID="${EVAL_LLM_ID:-${FTS_GEMMA_LOCAL:-google/gemma-3-270m}}"
SUBSETS="${EVAL_SUBSETS:-all}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
LIMIT="${EVAL_LIMIT:-0}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
PERCEPTION_MODE="${EVAL_PERCEPTION_MODE:-inplace}"
# default out dir is keyed by the stage dir name (…/<stage>/checkpoints/best_model.pt)
_stage="$(basename "$(dirname "$(dirname "${CKPT}")")")"
OUT_DIR="${EVAL_OUT_DIR:-${FTS_RESULTS_ROOT}/eval_tsrbench/${_stage}}"

fts_require_dir TSRBENCH_DIR "TSRBench jsonl root (see evaluation/tsrbench/README.md)" || exit 2
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (set EVAL_CKPT)" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

echo "Checkpoint : ${CKPT}"
echo "TSRBench   : ${TSRBENCH_DIR}"
echo "Subsets    : ${SUBSETS}"
echo "LLM        : ${LLM_ID}"
echo "Out dir    : ${OUT_DIR}"
echo "Max tokens : ${MAX_NEW_TOKENS}"

# The freq flag is inferred from the stage dir name (…_freq vs …_nofreq); set
# EVAL_FREQ_FLAG explicitly to override.
case "${CKPT}" in
  *_nofreq/*) _freq_default="" ;;
  *)          _freq_default="--use_freq_embed" ;;
esac
FREQ_FLAG="${EVAL_FREQ_FLAG-${_freq_default}}"

# shellcheck disable=SC2086
"${FTS_PYTHON}" evaluation/tsrbench/run_tsrbench.py \
  --ckpt            "${CKPT}" \
  --llm_id          "${LLM_ID}" \
  --data_dir        "${TSRBENCH_DIR}" \
  --subsets         "${SUBSETS}" \
  --out_dir         "${OUT_DIR}" \
  --max_new_tokens  "${MAX_NEW_TOKENS}" \
  --batch_size      "${BATCH_SIZE}" \
  --limit           "${LIMIT}" \
  --perception_mode "${PERCEPTION_MODE}" \
  --export_for_official_scorer \
  ${FREQ_FLAG} ${EVAL_EXTRA:-}
echo "EVAL_DONE tsrbench ${SUBSETS} -> ${OUT_DIR}"
