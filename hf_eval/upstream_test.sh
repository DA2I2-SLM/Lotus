#!/bin/bash
#SBATCH --job-name=upstream_test
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:40:00
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
cd /scratch/$(whoami)/OpenTSLM_upstream
echo "### GENUINE UPSTREAM: sleep-sp on sleep (no shims) ###"
python -u eval_upstream.py --repo_id OpenTSLM/gemma-3-270m-pt-sleep-sp --task sleep --n 32
echo "=== upstream_test done ==="
