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

python evaluation/captioning/m4_eval.py \
  --ckpt      /scratch/${USER_NAME}/OpenTSLM/results/sincos_patch4/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend/checkpoints/best_model.pt  \
  --llm_id     google/gemma-3-270m \
  --device     cuda \
  --batch_size 128 \
  --patch_size 16 \
  --pe_type    nope \
  --max_new_tokens 256 \
  --out_dir    /home/${USER_NAME}/FoundationalTS/phase1_extended_m4_eval
