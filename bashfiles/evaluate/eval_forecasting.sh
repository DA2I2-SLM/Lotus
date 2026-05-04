#!/bin/bash
#SBATCH --job-name=etth1-forecast-eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]
module load Anaconda3
source activate
conda activate opentslm

cd /home/s223540177/FoundationalTS

RESULTS=/scratch/s223540177/OpenTSLM/results/sincos_patch96/gemma_3_270m/OpenTSLMSPFixPos2D
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/inference.py
ETTh1=/scratch/s223540177/TimeSeriesECM/dataset/ETTh1.csv

# ── joint_phase1_extend_balanced checkpoint ─────────────────────────
CKPT=${RESULTS}/joint_phase1_extend_balanced/checkpoints/best_model.pt
STAGE=joint_phase1_extend_balanced

python ${INFER} \
  --ckpt         ${CKPT} \
  --llm_id       ${LLM} \
  --stage_name   ${STAGE} \
  --eval_format  phase1_pretrain \
  --eval_split   test \
  --mode         teacher_forced \
  --device       cuda \
  --batch_size   32 \
  --patch_size   96 \
  --history_len  480 \
  --pred_len     96 \
  --save_plots_every 1 \
  --out_dir      ./eval_results/${STAGE}/phase1_pretrain/teacher_forced_llm_test \
  --forecast_backbone llm
