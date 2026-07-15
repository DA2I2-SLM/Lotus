#!/bin/bash
#SBATCH --job-name=run_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%A_%a.out
#SBATCH --error=output_job/%x_%A_%a.err
#SBATCH --constraint="gpu-l40s|gpu-v100"
#
# Reusable OpenTSLM evaluation -> Accuracy / macro-F1.
# Faithful to the README demo (demo/huggingface/0{3,4,5}_test_hf_*.py:
# `model.generate(batch, max_new_tokens=500)`) but over the FULL test set with metrics,
# via eval_official.py in the genuine-upstream clone (so the published patch4/dim128
# checkpoints load; the dim768/patch32 fork cannot load them).
#
# Usage:
#   sbatch hf_eval/run_eval.sh <REPO_ID> <TASK>                                      # single GPU, full set
#   sbatch --array=0-7 --export=ALL,TOTAL=930 hf_eval/run_eval.sh <REPO_ID> <TASK>   # sharded
# Env overrides (optional; sensible auto-defaults per model family):
#   FIX  1=Flamingo conditioning fix (attend_previous+no_cache); auto 1 for *flamingo else 0
#   PAD  padding override; auto 'right' for *sp (trained right-padded), else default
#   GREEDY 1 (default), TOK 500 (default max_new_tokens), N 0 (default = full split)

# --- capture args FIRST, then clear $@ so `source activate` doesn't receive them ---
REPO=${1:?need REPO_ID (e.g. OpenTSLM/gemma-3-270m-pt-sleep-flamingo)}
TASK=${2:-sleep}
GREEDY=${GREEDY:-1}; TOK=${TOK:-500}; N=${N:-0}
if [ -z "${FIX+x}" ]; then case "$REPO" in *flamingo*) FIX=1;; *) FIX=0;; esac; fi
if [ -z "${PAD+x}" ]; then case "$REPO" in *sp) PAD=right;; *) PAD="";; esac; fi
ARR_ID="$SLURM_ARRAY_TASK_ID"; ARR_CNT="${SLURM_ARRAY_TASK_COUNT:-1}"
set --   # IMPORTANT: conda's `source activate` inherits $@ and errors on extra args

module load Anaconda3
source activate
conda activate opentslm
cd /scratch/$(whoami)/OpenTSLM_upstream

RES=/scratch/$(whoami)/OpenTSLM_upstream/results
TAG=$(printf "%s_%s" "${REPO##*/}" "$TASK" | tr -c 'A-Za-z0-9_' '_')

if [ -n "$ARR_ID" ] && [ -n "$TOTAL" ]; then
  per=$(( (TOTAL + ARR_CNT - 1) / ARR_CNT )); START=$(( ARR_ID*per )); END=$(( START+per )); [ $END -gt $TOTAL ] && END=$TOTAL
  OUT=$RES/RUNEVAL_${TAG}.shard${ARR_ID}.json
  echo "shard $ARR_ID/$ARR_CNT START=$START END=$END repo=$REPO task=$TASK fix=$FIX pad=${PAD:-default} greedy=$GREEDY tok=$TOK"
  python -u eval_official.py --repo_id "$REPO" --task "$TASK" --n 0 --start $START --end $END \
    --conditioning_fix "$FIX" --greedy "$GREEDY" --max_new_tokens "$TOK" --override_padding "$PAD" --out "$OUT"
else
  OUT=$RES/RUNEVAL_${TAG}.json
  echo "single repo=$REPO task=$TASK fix=$FIX pad=${PAD:-default} greedy=$GREEDY tok=$TOK n=$N"
  python -u eval_official.py --repo_id "$REPO" --task "$TASK" --n "$N" \
    --conditioning_fix "$FIX" --greedy "$GREEDY" --max_new_tokens "$TOK" --override_padding "$PAD" --out "$OUT"
fi
echo "=== run_eval done: $REPO $TASK ==="
