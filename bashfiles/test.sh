#!/bin/bash
#SBATCH --job-name=test_tsfm
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100

module load Anaconda3
source activate
conda activate opentsfm

python curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase1_extend \
  --llm_id google/gemma-3-270m \
  --num_epochs 1 \
  --pe_type sincos \
  --patch_size 32 \
  --max_samples_per_source 100 \
  --grad_accum_steps 1