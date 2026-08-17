#!/bin/bash
#SBATCH --job-name=up_diag
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]
# Faithful OpenTSLM-SP and -Flamingo diagonal via the GENUINE UPSTREAM evaluator.
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opentslm
export TOKENIZERS_PARALLELISM=false
cd /scratch/s223540177/OpenTSLM_upstream
OUTB=/scratch/s223540177/OpenTSLM_upstream/results/diagonal_upstream
mkdir -p "$OUTB"
for model in sp flamingo; do
  for task in har sleep ecg; do
    repo="OpenTSLM/gemma-3-270m-pt-${task}-${model}"
    echo "############ $repo -> $task ($(date +%H:%M)) ############"
    python eval_upstream.py --repo_id "$repo" --task "$task" --n 300 \
      --max_new_tokens 300 --padding right --greedy 1 \
      --out "$OUTB/${task}_${model}.json" > "$OUTB/${task}_${model}.log" 2>&1
    echo "exit=$?"; grep -iE "accuracy|f1|EM|Error|RuntimeError|size mismatch" "$OUTB/${task}_${model}.log" | tail -5
    echo "==== DONE ${task}_${model} ($(date +%H:%M)) ===="
  done
done
echo "ALL_UPSTREAM_DIAGONAL_COMPLETE"
