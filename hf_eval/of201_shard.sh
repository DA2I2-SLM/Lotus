#!/bin/bash
#SBATCH --job-name=of201_full
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=03:00:00
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
cd /scratch/$(whoami)/OpenTSLM_upstream
S=$SLURM_ARRAY_TASK_ID; NS=${SLURM_ARRAY_TASK_COUNT:-4}
per=$(( (TOTAL + NS - 1) / NS )); START=$(( S*per )); END=$(( START+per )); [ $END -gt $TOTAL ] && END=$TOTAL
python -u eval_of201.py --repo_id "$REPO" --task "$TASK" --n 0 --start $START --end $END \
  --max_new_tokens 500 --out "${OUTBASE}.shard${S}.json"
echo "=== of201 shard $S done ==="
