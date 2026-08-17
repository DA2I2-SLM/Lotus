#!/bin/bash
#SBATCH --job-name=j5_patsmoke
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu-v100|gpu-l40s"
#SBATCH --exclude=v100l-f-[01-06]
#SBATCH --output=/home/s223540177/FoundationalTS/output_job/%x_%j.out
#SBATCH --error=/home/s223540177/FoundationalTS/output_job/%x_%j.err

# Verifies the two early-stopping fixes in tokenized_sft.py:
#   1. patience is counted per EPOCH (one "CoT patience" line per epoch, not 4)
#   2. the stop decision is broadcast, so ALL ranks exit -> clean exit, no SIGABRT
# patience=1 forces the stop to fire on the first non-improving epoch, which is
# precisely the path that aborted the real Qwen run.
set -euo pipefail

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd /home/s223540177/FoundationalTS

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
export PHASE2_BALANCE_TARGET=60

RES=/scratch/s223540177/FoundationalTS/results_patsmoke
rm -rf "$RES"

torchrun --standalone --nproc_per_node=2 \
  baselines/tokenized_llm/tokenized_sft.py \
  --mode train \
  --model_id /home/s223540177/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --out_dir "$RES" \
  --epochs 4 \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum 2 \
  --max_len 4096 \
  --max_values 512 \
  --gen_em_n 4 \
  --eval_every 3 \
  --patience 1

echo "SMOKE_EXIT_CODE=$?"
echo "DONE_PATIENCE_SMOKE"
