#!/bin/bash
#SBATCH --job-name=up_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=03:00:00
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
cd /scratch/$(whoami)/OpenTSLM_upstream
# args: $1 repo  $2 task  $3 n  $4 out
python -u eval_upstream.py --repo_id "$1" --task "$2" --n "${3:-300}" --padding "${PADDING:-right}" --greedy "${GREEDY:-1}" --max_new_tokens 300 --out "${4:-}"
echo "=== up_eval done: $1 $2 ==="
