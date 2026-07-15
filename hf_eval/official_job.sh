#!/bin/bash
#SBATCH --job-name=official
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint="gpu-l40s|gpu-v100"
module load Anaconda3; source activate; conda activate opentslm
cd /scratch/$(whoami)/OpenTSLM_upstream
# args: $1 repo $2 task $3 n $4 conditioning_fix $5 greedy $6 out
python -u eval_official.py --repo_id "$1" --task "$2" --n "${3:-120}" \
  --conditioning_fix "${4:-0}" --greedy "${5:-0}" --max_new_tokens 500 --out "${6:-}"
echo "=== official done: $1 $2 fix=$4 greedy=$5 ==="
