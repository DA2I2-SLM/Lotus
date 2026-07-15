#!/bin/bash
#SBATCH --job-name=of201
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint="gpu-l40s|gpu-v100"
REPO=${1:?}; N=${2:-60}
set --
module load Anaconda3; source activate; conda activate opentslm
cd /scratch/$(whoami)/OpenTSLM_upstream
python -u pure_demo_test_of201.py --repo_id "$REPO" --n "$N"
echo "=== of201 done ==="
