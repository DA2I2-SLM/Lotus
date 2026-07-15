#!/bin/bash
#SBATCH --job-name=fixpos2d
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
#SBATCH --constraint="gpu-l40s|gpu-v100"
CKPT=${1:?}; OUT=${2:?}
set --
module load Anaconda3; source activate; conda activate opentslm
cd /home/$(whoami)/FoundationalTS
python -u hf_eval/eval_fixpos2d_matched.py --ckpt "$CKPT" --n 0 --max_new_tokens 500 --out "$OUT"
echo "=== fixpos2d done ==="
