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
USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS
python /home/${USER_NAME}/FoundationalTS/evaluation/baseline/evaluate_tsqa_opentslm.py --ckpt /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch96/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4/checkpoints/best_model.pt --llm_id google/gemma-3-270m --patch_size 96