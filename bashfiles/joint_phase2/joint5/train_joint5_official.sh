#!/bin/bash
#SBATCH --job-name=j5_off
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu-v100|gpu-l40s"
#SBATCH --exclude=v100l-f-[01-06]
#SBATCH --output=/home/s223540177/FoundationalTS/output_job/%x_%j.out
#SBATCH --error=/home/s223540177/FoundationalTS/output_job/%x_%j.err
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

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd /home/s223540177/FoundationalTS

VAR=$1
case $VAR in
  sp)
    WRAP=/vast/s223540177/cik_work/official_variants/train_official_sp.py
    MODEL=OpenTSLMSP
    OFFICIAL_CKPT=/home/s223540177/.cache/huggingface/hub/models--OpenTSLM--gemma-3-270m-m4-sp/snapshots/3199027ea112348ea22933027d8ae75f047df710/model_checkpoint.pt
    RES=/scratch/s223540177/FoundationalTS/results_official_sp_joint5_m4
    ;;
  fl)
    WRAP=/vast/s223540177/cik_work/official_variants/train_official_flamingo.py
    MODEL=OpenTSLMFlamingo
    OFFICIAL_CKPT=/home/s223540177/.cache/huggingface/hub/models--OpenTSLM--gemma-3-270m-m4-flamingo/snapshots/a48666258e769d6dc918cd69734c7e43e629a624/model_checkpoint.pt
    RES=/scratch/s223540177/FoundationalTS/results_official_flamingo_joint5_m4
    ;;
  *) echo "usage: $0 <sp|fl> [smoke]"; exit 1;;
esac

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
export HF_DATASETS_CACHE=/scratch/s223540177/hf_cache/datasets

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
torchrun --standalone --nproc_per_node=$NPROC "$WRAP" \
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
