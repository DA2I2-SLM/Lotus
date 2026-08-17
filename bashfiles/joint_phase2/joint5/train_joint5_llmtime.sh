#!/bin/bash
#SBATCH --job-name=j5_llmtime
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu-l40s|gpu-v100|gpu-rtxp"
#SBATCH --exclude=v100l-f-[01-06]
#SBATCH --output=/home/s223540177/FoundationalTS/output_job/%x_%j.out
#SBATCH --error=/home/s223540177/FoundationalTS/output_job/%x_%j.err
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

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd /home/s223540177/FoundationalTS

ARM="${1:?usage: train_joint5_llmtime.sh <zeroshot|lora> [smoke]}"
MODE="${2:-full}"

# BACKBONE selects the model scale so the OpenTSLM-protocol pair can exist in
# BOTH table blocks: 1b (default, matches ours-1b) and 270m (matches ours-270m).
BACKBONE="${BACKBONE:-1b}"
case "$BACKBONE" in
  1b)   HUB=models--google--gemma-3-1b-pt; SUF="" ;;
  270m) HUB=models--google--gemma-3-270m;  SUF="_270m" ;;
  *) echo "BACKBONE must be 1b or 270m"; exit 1;;
esac
MODEL=$(ls -d /home/s223540177/.cache/huggingface/hub/$HUB/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "FATAL: no local $HUB snapshot"; exit 1; }
echo "[llmtime] backbone: $MODEL (suffix '$SUF')"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/s223540177/hf_cache
export HF_DATASETS_CACHE=/scratch/s223540177/hf_cache/datasets

export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=40000

case "$ARM" in
  zeroshot)
    # No training. Evaluate the untouched base model on the test split.
    OUT=/scratch/s223540177/cik_work/eval_final/j5_llmtime_zeroshot${SUF}
    [ "$MODE" = "smoke" ] && OUT="${OUT}_smoke"
    echo "=== [LLMTime zero-shot] gemma-3-1b-pt, no training, test split ==="
    # eval mode has no --max_samples; it is capped with EVAL_SHARD_START/END,
    # the same mechanism the sharded eval relaunch scripts use.
    [ "$MODE" = "smoke" ] && export EVAL_SHARD_START=0 EVAL_SHARD_END=8
    for T in har sleep ecg pamap2 timemqa; do
      python -u baselines/tokenized_llm/tokenized_sft.py \
        --mode eval --model_id "$MODEL" --out_dir "$OUT/$T" \
        --split test --tasks "$T" --max_len 4096 --max_values 512 \
        --serializer llmtime
    done
    echo "DONE_LLMTIME_ZEROSHOT out=$OUT"
    ;;

  lora)
    RES=/scratch/s223540177/FoundationalTS/results_joint5_llmtime_lora${SUF}
    if [ "$MODE" = "smoke" ]; then
      EPOCHS=1; NPROC=1; GEN_EM_N=4; ACCUM=4; EVAL_EVERY=2
      export PHASE2_BALANCE_TARGET=50; RES="${RES}_smoke"; rm -rf "$RES"
    else
      EPOCHS=20; NPROC=4; GEN_EM_N=192; ACCUM=32; EVAL_EVERY=500
    fi
    echo "=== [LLMTime + LoRA] r=8 a=16, selection by VALIDATION LOSS ==="
    torchrun --standalone --nproc_per_node=$NPROC \
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
      --lora_r 8 --lora_alpha 16
    echo "DONE_LLMTIME_LORA best=${RES}/best"
    ;;

  *) echo "usage: $0 <zeroshot|lora> [smoke]"; exit 1;;
esac
