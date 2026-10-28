#!/bin/bash
#SBATCH --job-name s2gen
#SBATCH --output=slurm/logs/s2gen_%A_%a.out
#SBATCH --error=slurm/logs/s2gen_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time 1-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster
#SBATCH --array=0-9

# Generate SymTime / S2Generator pretrain data under the new constraint envelope:
#   - output_dim (n_variates) sampled per (ID, OD) pair from [3, 10]
#   - length fixed at 1024 (within [1024, 4096]; variable length here is
#     prohibitively slow given S2Generator's ~50s/file generation cost)
#   - input_dim sampled in [1, 6] (driving signal, not the target variate count)
#
# 10 array tasks * 10 seeds/chunk = 100 seeds total. With ID in [1,6] and
# OD in [3,10], that's 48 files per seed = 4,800 total files.

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate

cd "${SYNTH_SYMTIME_REPO:?set SYNTH_SYMTIME_REPO to the wwhenxuan/SymTime checkout}"
conda activate "${SYNTH_SYMTIME_ENV:-./.env}"   # SymTime venv with s2generator installed

SEEDS_PER_CHUNK=10
START_SEED=$((SLURM_ARRAY_TASK_ID * SEEDS_PER_CHUNK))
END_SEED=$((START_SEED + SEEDS_PER_CHUNK))

ROOT_PATH=${SYNTH_SYMTIME_REPO}/data/datasets/pretrain_data/
mkdir -p "${ROOT_PATH}"

echo "[$(date)] array_task=${SLURM_ARRAY_TASK_ID} seeds=[${START_SEED}, ${END_SEED})"

python -u data_provider/s2generator_main.py \
  --root_path "${ROOT_PATH}" \
  --start_seed ${START_SEED} \
  --end_seed   ${END_SEED} \
  --min_input_dim 1 \
  --max_input_dim 6 \
  --min_output_dim 3 \
  --max_output_dim 10 \
  --length 1024 \
  --num_threads 32

echo "[$(date)] done array_task=${SLURM_ARRAY_TASK_ID}"
