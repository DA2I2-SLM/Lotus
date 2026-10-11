#!/bin/bash
#SBATCH --job-name=j5_off_1b
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gpus=8
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

# Resolve repo root + interpreter + data locations dynamically so this
# script runs on any account. Override any FTS_* var in your environment.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -f "$_d/bashfiles/_paths.sh" ]; do _d="$(dirname "$_d")"; done
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"
#
# ─────────────────────────────────────────────────────────────────────────────
# gemma-3-1b counterpart of train_joint5_official.sh: the OpenTSLM-SP and
# OpenTSLM-Flamingo baselines at the SAME backbone as train_joint5_ours_1b.sh,
# so the 1b block of the results table has its own comparators instead of being
# comparable only to the 270m rows.
#
# Warm start: OpenTSLM/gemma-3-1b-pt-m4-{sp,flamingo} — the exact 1b siblings of
# the gemma-3-270m-m4-* checkpoints the published 270m rows use, i.e. the same
# point in OpenTSLM's own curriculum (general TSQA+M4 pretraining, no downstream
# task). NOT the ecg-* checkpoints: those are pretrained on ECG-QA, which is one
# of our five downstream tasks, and would hand the baseline task-specific
# pretraining our phase-1 never had.
#
# Verified against the downloaded checkpoints, not assumed:
#   sp:       patch_embed (128,1,4), projector out (1152,128), lora_enabled=False,
#             epoch 56 — architecturally identical to the 270m release except the
#             projector's output dim, so OPENTSLM_EMBED_DIM=128 / PATCH_SIZE=4 carry
#             over unchanged and only --llm_id and the ckpt path differ.
#   flamingo: same top-level key set as the 270m release; model_state 2648 tensors
#             vs 1880 (26 layers vs 18), and all 1374 shape differences are exactly
#             the tensors touching LLM hidden 1152 vs 640. epoch 37.
#
# DELTAS vs train_joint5_official.sh, both forced by hardware, neither changing
# the recipe:
#   * 8 ranks on one rtxp6000 node instead of 4 on l40s/v100, with
#     --grad_accum_steps 16 instead of 32. Effective batch is 8x1x16 = 128,
#     identical to the 270m baselines' 4x1x32 = 128 and to both ours runs.
#   * qos batch-long / 10 days. The 270m baselines took 3d10h (SP, 12 epochs) and
#     2d21h (FL, 9 epochs) on 4 GPUs; at ~7x the non-embedding params the same
#     4-GPU setup would exceed the 5-day batch-short cap outright. The Blackwell
#     node is measured at ~4 h/epoch for the 1b ours run, versus ~4.5-4.9 h/epoch
#     for the 270m ours run on v100/l40s — the hardware gain roughly cancels the
#     model-size increase.
#
# NOTE ON COST: patch 4 means ECG is 12 leads x 1024 / 4 = 3072 soft tokens per
# sample (ours uses patch 32 -> 384). SP was already the slow variant at 270m.
# Watch the first epoch before assuming the 10-day budget holds.
#
# Usage: sbatch --job-name=j5_off_1b_sp train_joint5_official_1b.sh sp [smoke]
#        sbatch --job-name=j5_off_1b_fl train_joint5_official_1b.sh fl [smoke]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd ${FTS_REPO_ROOT}

VAR=$1
case $VAR in
  sp)
    WRAP=${FTS_VAST}/cik_work/official_variants/train_official_sp.py
    MODEL=OpenTSLMSP
    OFFICIAL_CKPT=$(ls ${FTS_HF_HUB}/models--OpenTSLM--gemma-3-1b-pt-m4-sp/snapshots/*/model_checkpoint.pt 2>/dev/null | head -1)
    RES=${FTS_RESULTS_ROOT}_official_sp_joint5_m4_1b
    ;;
  fl)
    WRAP=${FTS_VAST}/cik_work/official_variants/train_official_flamingo.py
    MODEL=OpenTSLMFlamingo
    OFFICIAL_CKPT=$(ls ${FTS_HF_HUB}/models--OpenTSLM--gemma-3-1b-pt-m4-flamingo/snapshots/*/model_checkpoint.pt 2>/dev/null | head -1)
    RES=${FTS_RESULTS_ROOT}_official_flamingo_joint5_m4_1b
    ;;
  *) echo "usage: $0 <sp|fl> [smoke]"; exit 1;;
esac

# Fail loudly rather than silently training a fresh random model: the Python
# side only prints a warning when the warm-start checkpoint is missing.
if [ -z "${OFFICIAL_CKPT}" ] || [ ! -f "${OFFICIAL_CKPT}" ]; then
  echo "FATAL: 1b OpenTSLM warm-start checkpoint not found for VAR=$VAR"
  exit 1
fi

MODE="${2:-full}"
if [ "$MODE" = "smoke" ]; then
  EPOCHS=1; NPROC=1; GEN_EM_N=8; ACCUM=4; EXTRA="--max_samples_per_source 48"
  BALANCE_TARGET=200
  RES="${RES}_smoke"
  rm -rf "$RES"
else
  EPOCHS=20; NPROC="${NPROC:-8}"; GEN_EM_N=192; EXTRA=""
  # Effective batch is pinned at 128 for every run in the table, so accumulation
  # has to track world size rather than being hardcoded: 8 ranks -> 16,
  # 4 ranks -> 32 (which is exactly what the 270m baselines used). This lets the
  # job be scheduled on whatever hardware frees first without silently changing
  # the recipe. Override with ACCUM= only if you intend a different batch.
  ACCUM="${ACCUM:-$(( 128 / NPROC ))}"
  BALANCE_TARGET=40000
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENTSLM_EMBED_DIM=128
export OPENTSLM_PATCH_SIZE=4
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE=${FTS_SCRATCH}/hf_cache/datasets

# ---- our sampling method: identical to the 270m baselines and to ours ----
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
# TSQA removed: 80.8% of its phase-2 test rows were in phase-1 training.
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=$BALANCE_TARGET
export PHASE2_SELECT_BY_GEN=1
export PHASE2_GEN_EM_N=$GEN_EM_N
export PHASE2_EARLY_STOP_PAT=8

export WARM_START_PRETRAINED_CKPT="$OFFICIAL_CKPT"
export RESULTS_BASE_DIR="$RES"

echo "=== [$MODEL @ gemma-3-1b-pt] JOINT-5 balanced_all ==="
echo "  warm      = $OFFICIAL_CKPT"
echo "  results   = $RES"
echo "  world     = $NPROC ranks x batch 1 x accum $ACCUM = $(( NPROC * ACCUM )) effective"
echo "  encoder   = OPENTSLM_EMBED_DIM=128 patch=4 (ECG -> 3072 soft tokens/sample)"

torchrun --standalone --nproc_per_node=$NPROC "$WRAP" \
  --model $MODEL \
  --stages joint_phase2_downstream \
  --llm_id google/gemma-3-1b-pt \
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
echo "DONE_OFFICIAL_JOINT5_1B_$VAR ckpt=${CK:-MISSING}"
