#!/bin/bash
#SBATCH --job-name=j5_llmtime
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=4
#SBATCH --constraint="gpu-l40s|gpu-v100|gpu-rtxp"
#SBATCH --exclude=v100l-f-[01-06]
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
#
# ─────────────────────────────────────────────────────────────────────────────
# The two published-style tokenized baselines, using Gruver et al. (2023)'s
# actual encoding rather than our "%.2f" rendering:
#
#   zeroshot : "Tokenized time-series" -- serialise with the LLMTime scheme and
#              evaluate the BASE model on the test set with no training at all.
#   lora     : "Tokenized finetuned"   -- same encoding, LoRA fine-tuned on the
#              training set, best model chosen by VALIDATION LOSS (not by our
#              gen-EM basket), then scored on test.
#
# Usage: sbatch train_joint5_llmtime.sh <zeroshot|lora> [smoke]
#
# ── WHY A SEPARATE SERIALIZER ────────────────────────────────────────────────
# serialize_llmtime.py is a faithful port of llmtime's data/serialize.py:
# digit-level spacing (bit_sep=' '), no decimal point (decimal_point=''),
# positional precision (prec=3), per-series rescale. Verified against the
# paper's own docstring examples (0.5->50, 3.52->352, 12.5->1250).
# It costs ~7.02 tokens/value against 5.99 for "%.2f" on gemma's tokenizer, a
# 17% premium funded from the SAME token budget -- so the bisection fit in
# SerializedDataset decimates these runs harder, and the recorded stride will
# show it. That is a property of the published encoding, not a bug, but it must
# be reported alongside the ECG column rather than passed off as a model result.
#
# ── WHY VALIDATION LOSS ──────────────────────────────────────────────────────
# The published protocol selects on validation loss. Our own runs select on a
# generative exact-match basket. Nothing computed a validation loss here before
# --select_by val_loss was added, so this is a genuinely different selection
# rule and the rows are NOT interchangeable with our gen-EM-selected ones.
# Keep them visually separated in the table.
#
# ── FAIRNESS NOTE ────────────────────────────────────────────────────────────
# The zero-shot arm is expected to score near zero on the label-accuracy
# columns, because the base model has never seen our CoT answer format and the
# scorer needs an exact label match. That is what the published baseline
# measures; it is a floor, not evidence about the interface. Do not compare it
# against our fine-tuned rows as though it were a like-for-like.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${LOTUS_REPO_ROOT}"
ARM="${1:?usage: train_joint5_llmtime.sh <zeroshot|lora> [smoke]}"
MODE="${2:-full}"

# BACKBONE selects the model scale so the OpenTSLM-protocol pair can exist in
# BOTH table blocks: 1b (default, matches ours-1b) and 270m (matches ours-270m).
BACKBONE="${BACKBONE:-1b}"
# PSFLAG: Qwen3-0.6B is the first backbone here with BOTH a chat template and a
# reasoning mode. tokenized_sft.py auto-detects the template, wraps the prompt in
# <|im_start|>assistant, and Qwen3 then opens a <think> block that never closes
# inside the 200-token generation budget -- measured 8/8 unscoreable predictions
# on PAMAP2. --prompt_style raw pins it to the same raw concatenation the gemma
# rows and the encoder model use. Empty for 1b/270m, so those rows are unchanged.
PSFLAG=""
case "$BACKBONE" in
  1b)   HUB=models--google--gemma-3-1b-pt; SUF="" ;;
  270m) HUB=models--google--gemma-3-270m;  SUF="_270m" ;;
  qwen3) HUB=models--Qwen--Qwen3-0.6B; SUF="_qwen3"; PSFLAG="--prompt_style raw" ;;
  *) echo "BACKBONE must be 1b, 270m or qwen3"; exit 1;;
esac
case "$BACKBONE" in
  1b)    HUB_ID=google/gemma-3-1b-pt ;;
  270m)  HUB_ID=google/gemma-3-270m ;;
  qwen3) HUB_ID=Qwen/Qwen3-0.6B ;;
esac
# Prefer a local snapshot (compute nodes may have no network); fall back to the hub id.
MODEL=$(ls -d ${LOTUS_HF_HUB}/$HUB/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && MODEL=$(ls -d ${LOTUS_SCRATCH}/hf_cache/hub/$HUB/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "[llmtime] no local $HUB snapshot; falling back to $HUB_ID (needs network + HF_TOKEN)"; MODEL=$HUB_ID; }
echo "[llmtime] backbone: $MODEL (suffix '$SUF') prompt_style='${PSFLAG:-auto}'"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${LOTUS_SCRATCH}/hf_cache
export HF_DATASETS_CACHE=${LOTUS_SCRATCH}/hf_cache/datasets

export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=40000

case "$ARM" in
  zeroshot)
    # No training. Evaluate the untouched base model on the test split.
    OUT="${EVAL_OUT_DIR:-${LOTUS_RESULTS_ROOT}/eval_phase2_cot}/j5_llmtime_zeroshot${SUF}"
    [ "$MODE" = "smoke" ] && OUT="${OUT}_smoke"
    echo "=== [LLMTime zero-shot] gemma-3-1b-pt, no training, test split ==="
    # eval mode has no --max_samples; it is capped with EVAL_SHARD_START/END,
    # the same mechanism the sharded eval relaunch scripts use.
    [ "$MODE" = "smoke" ] && export EVAL_SHARD_START=0 EVAL_SHARD_END=8
    for T in har sleep ecg pamap2 timemqa; do
      "${LOTUS_PYTHON}" -u baselines/tokenized_llm/tokenized_sft.py \
        --mode eval --model_id "$MODEL" --out_dir "$OUT/$T" \
        --split test --tasks "$T" --max_len 4096 --max_values 512 \
        --serializer llmtime $PSFLAG
    done
    echo "DONE_LLMTIME_ZEROSHOT out=$OUT"
    ;;

  lora)
    RES=${LOTUS_RESULTS_ROOT}_joint5_llmtime_lora${SUF}
    if [ "$MODE" = "smoke" ]; then
      EPOCHS=1; NPROC=1; GEN_EM_N=4; ACCUM=4; EVAL_EVERY=2
      export PHASE2_BALANCE_TARGET=50; RES="${RES}_smoke"; rm -rf "$RES"
    else
      EPOCHS=20; NPROC=4; GEN_EM_N=192; ACCUM=32; EVAL_EVERY=500
    fi
    echo "=== [LLMTime + LoRA] r=8 a=16, selection by VALIDATION LOSS ==="
    "${LOTUS_TORCHRUN}" --standalone --nproc_per_node=$NPROC \
      baselines/tokenized_llm/tokenized_sft.py \
      --mode train \
      --model_id "$MODEL" \
      --out_dir "$RES" \
      --epochs $EPOCHS \
      --lr 2e-4 \
      --batch_size 1 \
      --grad_accum $ACCUM \
      --max_len 4096 \
      --max_values 512 \
      --gen_em_n $GEN_EM_N \
      --eval_every $EVAL_EVERY \
      --patience 8 \
      --serializer llmtime \
      --select_by val_loss \
      --lora_r 8 --lora_alpha 16 $PSFLAG
    echo "DONE_LLMTIME_LORA best=${RES}/best"
    ;;

  *) echo "usage: $0 <zeroshot|lora> [smoke]"; exit 1;;
esac
