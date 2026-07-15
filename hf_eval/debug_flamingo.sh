#!/bin/bash
#SBATCH --job-name=dbg_flamingo
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s
module load Anaconda3
source activate
conda activate opentslm
cd /home/$(whoami)/FoundationalTS
python -u hf_eval/debug_flamingo.py
echo "=== debug_flamingo done ==="
