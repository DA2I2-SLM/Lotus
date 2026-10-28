#!/bin/bash
# Language eval (TSQA MCQ accuracy + M4 captioning ROUGE) for the MLP
# joint_phase1_extend_balanced checkpoint trained by
# bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth.sh.
#
# Two independent Python entry points run sequentially in this one script:
#   1) evaluate_tsqa_opentslm.py -> TSQA/   metrics.json + predictions.jsonl
#   2) m4_eval.py                -> M4/     metrics.json + predictions.jsonl
# Each produces its own performance score + per-sample predictions (the
# "samples" the task asked for).
#
# Outputs:
#   eval_results/mlp_joint_phase1_extbal_nofreq/language/tsqa/
#   eval_results/mlp_joint_phase1_extbal_nofreq/language/m4/
#
#SBATCH --job-name=eval-lang-mlp-joint-extbal-nofreq
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


cd "${FTS_REPO_ROOT}"

ENV_PY="${FTS_PYTHON}"
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

LLM=google/gemma-3-270m
TSQA_EVAL=evaluation/baseline/evaluate_tsqa_opentslm.py
M4_EVAL=evaluation/captioning/m4_eval.py

CKPT=${FTS_PRETRAINED_DIR}/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt
OUT_BASE=./eval_results/mlp_joint_phase1_extbal_nofreq/language

echo "CKPT=${CKPT}"

echo "================================================================"
echo "[eval_language] TSQA MCQ -> ${OUT_BASE}/tsqa"
echo "================================================================"
${ENV_PY} ${TSQA_EVAL} \
  --ckpt       ${CKPT} \
  --llm_id     ${LLM} \
  --tasks      tsqa \
  --patch_size 32 \
  --pe_type    sincos \
  --enc_dec_type       mlp \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --device     cuda \
  --out_dir    ${OUT_BASE}/tsqa

echo "================================================================"
echo "[eval_language] M4 captioning -> ${OUT_BASE}/m4"
echo "================================================================"
${ENV_PY} ${M4_EVAL} \
  --ckpt           ${CKPT} \
  --llm_id         ${LLM} \
  --device         cuda \
  --batch_size     32 \
  --patch_size     32 \
  --pe_type        sincos \
  --enc_dec_type       mlp \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --max_new_tokens 128 \
  --out_dir        ${OUT_BASE}/m4
