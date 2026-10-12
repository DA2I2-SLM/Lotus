#!/bin/bash
#SBATCH --job-name=j5_tok_gemma
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

# Backbone-MATCHED tokenized baseline: google/gemma-3-270m -- the *same* LLM our
# OpenTSLMSPFixPos2D wraps -- fine-tuned on the same five-task phase-2 mixture,
# but with the series written into the prompt as text instead of projected into
# the embedding stream as patch soft prompts.
#
# This is the control the Qwen2.5-0.5B run cannot be. That run confounds three
# things at once: different backbone family, ~2x the parameters, and a different
# interface. Here only the interface differs, so a gap is attributable to the
# interface and nothing else.
#
# Two properties checked before launching, both favourable:
#   * numeric tokenisation cost is the same -- 8 serialised values cost 48 tokens
#     under gemma vs 47 under Qwen (both split numbers digit-by-digit), so the
#     baseline is not starved of series values by the backbone swap;
#   * gemma has a real <pad> (id 0) distinct from <eos> (id 1), which removes the
#     pad==eos attention-mask warning the Qwen run emits.
#
# gemma-3-270m has NO chat template (it is a base model, not -it). tokenized_sft
# falls back to raw text concatenation there, which is exactly how our model
# assembles pre_prompt / post_prompt. Same prompt string, two interfaces.
#
# Usage: sbatch [--job-name=...] train_joint5_tokenized_gemma.sh [smoke]
set -euo pipefail

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd ${FTS_REPO_ROOT}

MODE="${1:-full}"
RES=${FTS_RESULTS_ROOT}_joint5_tokenized_gemma

# Local snapshot: gemma-3-270m is gated, and the run must not depend on a live
# HF fetch on a compute node. Same snapshot cik_inference_demo.py pins.
MODEL=$(ls -d ${FTS_HF_HUB}/models--google--gemma-3-270m/snapshots/*/ | head -1)
echo "[gemma] backbone snapshot: $MODEL"

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

echo "=== [tokenized gemma-3-270m] JOINT-5 balanced_all | mode=$MODE | results=$RES ==="

torchrun --standalone --nproc_per_node=$NPROC \
  baselines/tokenized_llm/tokenized_sft.py \
  --mode train \
  --model_id "$MODEL" \
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

echo "DONE_TOKENIZED_GEMMA_JOINT5 best=${RES}/best"
