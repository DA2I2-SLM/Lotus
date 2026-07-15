#!/bin/bash
#SBATCH --job-name=dbg_attn
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:50:00
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
python -u hf_eval/debug_attn.py
echo "=== debug_attn done ==="
