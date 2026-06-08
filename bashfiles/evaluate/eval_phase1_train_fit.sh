#!/bin/bash
#SBATCH --job-name=phase1-train-fit
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]
module load Anaconda3
source activate
conda activate opentslm

USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

RESULTS=/scratch/${USER_NAME}/OpenTSLM/results/sincos_patch96/gemma_3_270m/OpenTSLMSPFixPos2D
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/inference.py

# Evaluate on the TRAINING split of the Phase-1 Chronos corpus.
# Checks whether the model fits its own training distribution.

# CKPT=${RESULTS}/joint_phase1_extend_balanced/checkpoints/best_model.pt
CKPT=${RESULTS}/joint_phase1_extend_balanced/checkpoints/best_model.pt
STAGE=joint_phase1_extend_balanced

python ${INFER} \
  --ckpt         ${CKPT} \
  --llm_id       ${LLM} \
  --stage_name   ${STAGE} \
  --eval_format  phase1_pretrain \
  --eval_split   train \
  --mode         autoregressive \
  --device       cuda \
  --batch_size   32 \
  --patch_size   96 \
  --history_len  480 \
  --pred_len     96 \
  --plot_every   200 \
  --out_dir      ./eval_results/${STAGE}/chronos_train_split/autoregressive
