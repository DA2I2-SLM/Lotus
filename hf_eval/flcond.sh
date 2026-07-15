#!/bin/bash
#SBATCH --job-name=flcond2
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-l40s
module load Anaconda3; source activate; conda activate opentslm
cd /scratch/s223540177/OpenTSLM_upstream
python -u dbg_flamingo_cond.py
echo "=== flcond done ==="
