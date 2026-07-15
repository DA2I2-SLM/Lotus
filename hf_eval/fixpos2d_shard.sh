#!/bin/bash
#SBATCH --job-name=fp2d_shard
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%A_%a.out
#SBATCH --error=output_job/%x_%A_%a.err
#SBATCH --constraint="gpu-l40s|gpu-v100"
set --
module load Anaconda3; source activate; conda activate opentslm
cd /home/$(whoami)/FoundationalTS
S=$SLURM_ARRAY_TASK_ID; NS=${SLURM_ARRAY_TASK_COUNT:-6}
per=$(( (TOTAL + NS - 1) / NS )); START=$(( S*per )); END=$(( START+per )); [ $END -gt $TOTAL ] && END=$TOTAL
python -u hf_eval/eval_fixpos2d_matched.py --ckpt "$CKPT" --task "$TASK" --n 0 --start $START --end $END \
  --max_new_tokens 500 --repetition_penalty "${REPPEN:-1.0}" --no_repeat_ngram_size "${NOREP:-0}" \
  --out "${OUTBASE}.shard${S}.json"
echo "=== fp2d shard $S done ==="
