#!/bin/bash
#SBATCH --job-name=dbg_enc_var
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s
module load Anaconda3
source activate
conda activate opentslm
cd /home/$(whoami)/FoundationalTS
python -u hf_eval/debug_encoder_var.py
echo "=== debug_encoder_var done ==="
