#!/bin/bash
#SBATCH --job-name=s2gen_sweep
#SBATCH --output=slurm/logs/s2gen_%A_%a.out
#SBATCH --error=slurm/logs/s2gen_%A_%a.err
#SBATCH --array=0-23
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Single-sbatch chunked sweep for SymTime / S2Generator pretrain data.
# Submit with just:
#   sbatch slurm/s2generate_sweep.slurm
#
# 24 array tasks dispatch to three tags, each targeting ~100k successful
# samples (~45% S2Generator success rate assumed):
#
#   TASK_ID  | tag    | chunk# | SEEDS_PER_CHUNK | per-task seeds | ~samples/task
#   ---------+--------+--------+-----------------+----------------+--------------
#   0-3      | short  | 0-3    | 5000            | 5000           | ~27,000
#   4-11     | medium | 0-7    | 1250            | 1250           | ~14,000
#   12-23    | long   | 0-11   | 600             | 600            | ~9,750
#
# Totals: short ~108k, medium ~112k, long ~117k samples.
#
# All chunks within a tag write to the same pretrain_data_<tag>/ directory,
# with disjoint seed ranges so seed subdirectories never collide.
#
# Quick smoke test (one short chunk, just 10 seeds, ~3 min):
#   sbatch --array=0 --export=ALL,SEEDS_PER_CHUNK_SHORT=10 slurm/s2generate_sweep.slurm

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate

REPO=${SYNTH_SYMTIME_REPO:?set SYNTH_SYMTIME_REPO to the wwhenxuan/SymTime checkout}
cd "${REPO}"
# SymTime's own venv (py3.12) carries the S2Generator package.
conda activate "${SYNTH_SYMTIME_ENV:-./.env}"   # SymTime venv with s2generator installed

: "${SLURM_ARRAY_TASK_ID:?must be run as an array job}"

# Tag-specific seeds-per-chunk (override any of these at submit time).
SEEDS_PER_CHUNK_SHORT=${SEEDS_PER_CHUNK_SHORT:-5000}
SEEDS_PER_CHUNK_MEDIUM=${SEEDS_PER_CHUNK_MEDIUM:-1250}
SEEDS_PER_CHUNK_LONG=${SEEDS_PER_CHUNK_LONG:-600}

# Map SLURM_ARRAY_TASK_ID -> (TAG, CHUNK_IDX, config). Mapping is:
#   short:  ids 0-3   -> chunks 0-3
#   medium: ids 4-11  -> chunks 0-7
#   long:   ids 12-23 -> chunks 0-11
if   [ "${SLURM_ARRAY_TASK_ID}" -le 3 ]; then
    TAG=short
    CHUNK_IDX=$((SLURM_ARRAY_TASK_ID))
    LENGTH=1024
    MIN_ID=1; MAX_ID=4
    MIN_OD=3; MAX_OD=5
    SEEDS_PER_CHUNK=${SEEDS_PER_CHUNK_SHORT}
elif [ "${SLURM_ARRAY_TASK_ID}" -le 11 ]; then
    TAG=medium
    CHUNK_IDX=$((SLURM_ARRAY_TASK_ID - 4))
    LENGTH=2048
    MIN_ID=2; MAX_ID=6
    MIN_OD=4; MAX_OD=8
    SEEDS_PER_CHUNK=${SEEDS_PER_CHUNK_MEDIUM}
elif [ "${SLURM_ARRAY_TASK_ID}" -le 23 ]; then
    TAG=long
    CHUNK_IDX=$((SLURM_ARRAY_TASK_ID - 12))
    LENGTH=4096
    MIN_ID=3; MAX_ID=8
    MIN_OD=5; MAX_OD=10
    SEEDS_PER_CHUNK=${SEEDS_PER_CHUNK_LONG}
else
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0,23]" >&2
    exit 1
fi

START_SEED=$((CHUNK_IDX * SEEDS_PER_CHUNK))
END_SEED=$((START_SEED + SEEDS_PER_CHUNK))

ROOT_PATH="${REPO}/data/datasets/pretrain_data_${TAG}/"
mkdir -p "${ROOT_PATH}"

echo "[$(date)] s2gen_sweep array_task=${SLURM_ARRAY_TASK_ID} TAG=${TAG} chunk=${CHUNK_IDX} seeds=[${START_SEED}, ${END_SEED}) L=${LENGTH} ID=[${MIN_ID},${MAX_ID}] OD=[${MIN_OD},${MAX_OD}] -> ${ROOT_PATH}"

python -u data_provider/s2generator_main.py \
    --root_path "${ROOT_PATH}" \
    --start_seed "${START_SEED}" \
    --end_seed   "${END_SEED}" \
    --min_input_dim  "${MIN_ID}" --max_input_dim  "${MAX_ID}" \
    --min_output_dim "${MIN_OD}" --max_output_dim "${MAX_OD}" \
    --length "${LENGTH}" \
    --num_threads 32

echo "[$(date)] done array_task=${SLURM_ARRAY_TASK_ID} (TAG=${TAG} chunk=${CHUNK_IDX})"
