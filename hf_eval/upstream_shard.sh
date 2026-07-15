#!/bin/bash
#SBATCH --job-name=up_shard
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%A_%a.out
#SBATCH --error=output_job/%x_%A_%a.err
#SBATCH --constraint="gpu-l40s|gpu-v100"
# env: REPO TASK TOTAL NSHARDS OUTBASE
module load Anaconda3; source activate; conda activate opentslm
cd /scratch/$(whoami)/OpenTSLM_upstream
S=$SLURM_ARRAY_TASK_ID
per=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
START=$(( S * per )); END=$(( START + per )); [ $END -gt $TOTAL ] && END=$TOTAL
echo "shard $S/$NSHARDS  START=$START END=$END  repo=$REPO task=$TASK"
python -u eval_upstream.py --repo_id "$REPO" --task "$TASK" --n 0 --start $START --end $END \
  --padding "${PADDING:-right}" --greedy "${GREEDY:-1}" --max_new_tokens 300 --out "${OUTBASE}.shard${S}.json"
echo "=== shard $S done ==="
