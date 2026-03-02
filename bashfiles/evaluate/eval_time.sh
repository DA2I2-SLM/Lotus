#!/bin/bash
#SBATCH --job-name=tsqa-evaluation
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


python /home/s223540177/FoundationalTS/evaluation/baseline/evaluate_tsqa_opentslm.py     /scratch/s223540177/OpenTSLM/results/sincos_time_only/gemma_3_270m/OpenTSLMSPFixPos2D/stage1_mcq/checkpoints/best_model.pt     --base_model_id google/gemma-3-270m     --pe_type sincos