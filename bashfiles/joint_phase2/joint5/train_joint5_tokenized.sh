#!/bin/bash
#SBATCH --job-name=j5_tok
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gpus=4
#SBATCH --constraint="gpu-v100|gpu-l40s"
#SBATCH --exclude=v100l-f-[01-06]
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

# Tokenized-LLM baseline: Qwen2.5-0.5B-Instruct fine-tuned on the SAME five-task
# phase-2 mixture as j5_ours / j5_off_sp / j5_off_fl, but with the time series
# written into the prompt as text instead of projected as a soft prompt.
#
# This is the control for the paper's architectural claim, and the stand-in for
# the TimeOmni-1 family (which is inference-only, has no released checkpoint
# under 4B, and feeds series as text exactly like this).
#
# Deliberately generous to the baseline: the backbone is FULLY fine-tuned
# (~494M trainable vs ~4.1M for ours), so an encoder win cannot be attributed
# to the baseline being under-parameterised.
#
# Usage: sbatch [--job-name=...] train_joint5_tokenized.sh [smoke]
set -euo pipefail

cd "${FTS_REPO_ROOT}"
MODE="${1:-full}"
# Backbone is selectable so the same recipe can be run twice:
#   qwen  -> Qwen2.5-0.5B-Instruct (494M) : stronger/newer LM, NOT backbone-matched
#   gemma -> google/gemma-3-270m  (268M)  : SAME backbone our encoder model wraps,
#            so this run varies ONLY the interface (projected patches vs digits).
# Full fine-tune in both cases, matching what text-interface methods actually do
# (TimeOmni-1 post-trains the whole model; there is no adapter in that family).
BACKBONE="${2:-qwen}"
case "$BACKBONE" in
  qwen)  MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"; TAG="" ;;
  gemma) MODEL_ID="${FTS_GEMMA_LOCAL:-google/gemma-3-270m}"; TAG="_gemma" ;;
  *) echo "usage: $0 [full|smoke] [qwen|gemma]"; exit 1 ;;
esac
RES=${FTS_RESULTS_ROOT}_joint5_tokenized${TAG}

if [ "$MODE" = "smoke" ]; then
  EPOCHS=1; NPROC=1; GEN_EM_N=4; ACCUM=4; EVAL_EVERY=2
  BALANCE_TARGET=50
  RES="${RES}_smoke"
  rm -rf "$RES"
else
  EPOCHS=20; NPROC=4; GEN_EM_N=192; ACCUM=32; EVAL_EVERY=500
  BALANCE_TARGET=40000
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${FTS_SCRATCH}/hf_cache
export HF_DATASETS_CACHE=${FTS_SCRATCH}/hf_cache/datasets

# ---- identical sampling to the three encoder runs ----
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=$BALANCE_TARGET

echo "=== [tokenized $BACKBONE: $MODEL_ID] JOINT-5 balanced_all | mode=$MODE | results=$RES ==="

"${FTS_TORCHRUN}" --standalone --nproc_per_node=$NPROC \
  baselines/tokenized_llm/tokenized_sft.py \
  --mode train \
  --model_id "$MODEL_ID" \
  --out_dir "$RES" \
  --epochs $EPOCHS \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum $ACCUM \
  --max_len 4096 \
  --max_values 512 \
  --gen_em_n $GEN_EM_N \
  --eval_every $EVAL_EVERY \
  --patience 8

echo "DONE_TOKENIZED_JOINT5 best=${RES}/best"
