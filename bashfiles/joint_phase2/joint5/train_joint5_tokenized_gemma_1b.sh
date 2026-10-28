#!/bin/bash
#SBATCH --job-name=j5_tok_gemma1b
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=4
#SBATCH --constraint="gpu-l40s|gpu-v100"
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
#
# ─────────────────────────────────────────────────────────────────────────────
# Backbone-MATCHED tokenized baseline for the 1b block: google/gemma-3-1b-pt --
# the *same* LLM train_joint5_ours_1b.sh wraps -- fine-tuned on the same
# five-task phase-2 mixture, but reading the series as serialised digits instead
# of projected patches. It is to Ours-1b exactly what the gemma-270m tokenized
# row is to Ours-270m: the control that separates "the interface" from "the
# backbone". Without it, a win for Ours-1b cannot be attributed to the encoder.
#
# Identical to train_joint5_tokenized_gemma.sh except:
#   * MODEL -> the gemma-3-1b-pt snapshot (see naming note below)
#   * RES   -> ..._gemma_1b
#   * qos batch-long / 10 days and mem 200G: the 270m tokenized run needed ~5
#     days, and this backbone is ~3.7x larger.
# Everything the fairness of the comparison depends on is untouched: epochs 20,
# lr 1e-5, batch 1, grad_accum 32 (effective 128), max_len 4096, max_values 512,
# gen_em_n 192, patience 8 (epoch-level, per the DDP-broadcast fix), and the
# same balanced_all @ 40k/epoch five-task mixture.
#
# NAMING: the base 1B is google/gemma-3-1b-PT. Google suffixed the base model
# `-pt` (pretrained) for the 1B/4B/12B/27B family but left the 270m base
# unsuffixed, so `google/gemma-3-1b` and `google/gemma-3-270m-pt` both 404.
# `-pt` is the counterpart of `-it`; both this and the 270m row use the BASE,
# non-instruction-tuned model, which is what keeps the comparison clean.
#
# gemma-3 base models have NO chat template. tokenized_sft falls back to raw
# text concatenation there, which is exactly how our model consumes the prompt.
#
# Usage: sbatch train_joint5_tokenized_gemma_1b.sh [smoke]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "${FTS_REPO_ROOT}"
MODE="${1:-full}"
# LORA_R>0 runs the matched-capacity arm: the tokenized baseline gets an adapter
# of the SAME rank/alpha/target-modules our encoder model uses, instead of its
# default full fine-tune. Together with FULL_FT=1 on the ours launcher this
# forms a 2x2 over interface (patches vs digits) x capacity (LoRA vs full FT).
LORA_ARGS=""
_ARM_SUFFIX=""
if [ "${LORA_R:-0}" != "0" ]; then
  LORA_ARGS="--lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA:-16}"
  # An adapter needs a far higher LR than a full fine-tune. Defaulting to the
  # baseline's full-FT 1e-5 would under-train it by ~20x and make the LoRA arm
  # look artificially weak. Match our own model's LoRA LR instead (2e-4).
  TOK_LR="${TOK_LR:-2e-4}"
  _ARM_SUFFIX="_lora"
  echo "### ARM: LoRA r=${LORA_R} alpha=${LORA_ALPHA:-16} (matched to ours) ###"
fi
RES=${FTS_RESULTS_ROOT}_joint5_tokenized_gemma_1b${_ARM_SUFFIX}

# Local snapshot: gemma is gated, and the run must not depend on a live HF fetch
# from a compute node. Downloaded 2026-08-12 alongside the 1b phase-1 checkpoint.
MODEL=$(ls -d ${FTS_HF_HUB}/models--google--gemma-3-1b-pt/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then
  echo "[gemma] no local google/gemma-3-1b-pt snapshot under ${FTS_HF_HUB}; falling back to the hub id (needs network + HF_TOKEN)"
  MODEL=google/gemma-3-1b-pt
fi
echo "[gemma-1b] backbone snapshot: $MODEL"

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

# ---- identical sampling to every other joint5 run ----
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=$BALANCE_TARGET

echo "=== [tokenized gemma-3-1b-pt] JOINT-5 balanced_all | mode=$MODE | results=$RES ==="

"${FTS_TORCHRUN}" --standalone --nproc_per_node=$NPROC \
  baselines/tokenized_llm/tokenized_sft.py \
  --mode train \
  --model_id "$MODEL" \
  --out_dir "$RES" \
  --epochs $EPOCHS \
  --lr ${TOK_LR:-1e-5} \
  --batch_size 1 \
  --grad_accum $ACCUM \
  --max_len 4096 \
  --max_values 512 \
  --gen_em_n $GEN_EM_N \
  --eval_every $EVAL_EVERY \
  --patience 8 \
  $LORA_ARGS

echo "DONE_TOKENIZED_GEMMA1B_JOINT5 best=${RES}/best"
