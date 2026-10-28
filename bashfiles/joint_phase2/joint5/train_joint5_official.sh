#!/bin/bash
#SBATCH --job-name=j5_off
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
# JOINT fine-tune of an OFFICIAL OpenTSLM variant with OUR sampling method
# (PHASE2_COT_MIX=balanced_all @ 40k/epoch), on the EXACT jointall6 diet:
#   har, sleep, ecg, pamap2, tsqa, timemqa  (training uses TRAIN splits; the
#   pinned ecg_test_first5000 manifest is for EVAL fairness only).
# tsqa stays in TRAINING for diet symmetry with jointall6 but is footnoted in
# the benchmark (it was in ours' phase-1 pretraining); timemqa is the clean
# added benchmark column. Effective batch matched to jointall6: 4x1x32 = 128.
# WARM START: gemma-3-270m-m4-{sp,flamingo}, NOT the ecg-* checkpoints.
# The ecg-* checkpoints are pretrained on ECG-QA, which is one of the five
# downstream tasks -- that hands the baseline task-specific pretraining our
# phase-1 never had, and it showed up as the baselines matching us on ECG while
# trailing by 0.15-0.33 elsewhere. The m4-* checkpoints sit at the same point in
# OpenTSLM's curriculum as our phase-1 does in ours: general TSQA+M4 pretraining,
# no downstream task. That is the controlled comparison.
# Usage: sbatch --job-name=j5_off_<sp|fl> train_joint5_official.sh <sp|fl> [smoke]
set -euo pipefail

cd "${FTS_REPO_ROOT}"
# The official OpenTSLM trainer wrappers (train_official_sp.py /
# train_official_flamingo.py) live outside this repo; point FTS_OFFICIAL_VARIANTS
# at the directory that holds them.
OFFICIAL_VARIANTS="${FTS_OFFICIAL_VARIANTS:-${FTS_REPO_ROOT}/baselines/official_variants}"
VAR=$1
case $VAR in
  sp)
    WRAP=${OFFICIAL_VARIANTS}/train_official_sp.py
    MODEL=OpenTSLMSP
    OFFICIAL_CKPT="${OFFICIAL_CKPT:-$(ls ${FTS_HF_HUB}/models--OpenTSLM--gemma-3-270m-m4-sp/snapshots/*/model_checkpoint.pt 2>/dev/null | head -1)}"
    RES=${FTS_RESULTS_ROOT}_official_sp_joint5_m4
    ;;
  fl)
    WRAP=${OFFICIAL_VARIANTS}/train_official_flamingo.py
    MODEL=OpenTSLMFlamingo
    OFFICIAL_CKPT="${OFFICIAL_CKPT:-$(ls ${FTS_HF_HUB}/models--OpenTSLM--gemma-3-270m-m4-flamingo/snapshots/*/model_checkpoint.pt 2>/dev/null | head -1)}"
    RES=${FTS_RESULTS_ROOT}_official_flamingo_joint5_m4
    ;;
  *) echo "usage: $0 <sp|fl> [smoke]"; exit 1;;
esac
[ -f "$WRAP" ] || { echo "FATAL: official trainer wrapper not found: $WRAP (set FTS_OFFICIAL_VARIANTS)"; exit 1; }
[ -n "$OFFICIAL_CKPT" ] && [ -f "$OFFICIAL_CKPT" ] || { echo "FATAL: official checkpoint not found under ${FTS_HF_HUB} (set OFFICIAL_CKPT)"; exit 1; }

MODE="${2:-full}"
if [ "$MODE" = "smoke" ]; then
  EPOCHS=1; NPROC=1; GEN_EM_N=8; ACCUM=4; EXTRA="--max_samples_per_source 48"
  BALANCE_TARGET=200
  RES="${RES}_smoke"
else
  EPOCHS=20; NPROC=4; GEN_EM_N=192; ACCUM=32; EXTRA=""
  BALANCE_TARGET=40000
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENTSLM_EMBED_DIM=128
export OPENTSLM_PATCH_SIZE=4
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE=${FTS_SCRATCH}/hf_cache/datasets

# ---- our sampling method (mirrors train_joint6_from_phase1freq.sh) ----
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
# TSQA removed: 80.8% of its phase-2 test rows were in phase-1 training.
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=$BALANCE_TARGET
export PHASE2_SELECT_BY_GEN=1
export PHASE2_GEN_EM_N=$GEN_EM_N
# Matched to ours (was 5) so all three runs stop under the same rule.
export PHASE2_EARLY_STOP_PAT=8

export WARM_START_PRETRAINED_CKPT="$OFFICIAL_CKPT"
export RESULTS_BASE_DIR="$RES"

echo "=== [$MODEL] JOINT-5 (har,sleep,ecg,pamap2,timemqa) balanced_all | warm=$OFFICIAL_CKPT | results=$RES ==="
"${FTS_TORCHRUN}" --standalone --nproc_per_node=$NPROC "$WRAP" \
  --model $MODEL \
  --stages joint_phase2_downstream \
  --llm_id google/gemma-3-270m \
  --num_epochs $EPOCHS \
  --pe_type nope \
  --batch_size 1 --val_batch_size 1 \
  --max_tokens_per_batch 600 \
  --grad_accum_steps $ACCUM \
  --dataloader_workers 0 \
  --save_every_n_steps 800 \
  --warm_start_from pretrained \
  $EXTRA

CK=$(find "$RES" -name best_model.pt -path "*joint_phase2_downstream*" | head -1)
echo "DONE_OFFICIAL_JOINT5_$VAR ckpt=${CK:-MISSING}"
