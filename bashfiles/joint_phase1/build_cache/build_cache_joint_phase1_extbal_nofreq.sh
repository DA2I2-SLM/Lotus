#!/bin/bash
# Build the on-disk Chronos splits + stub cache for joint_phase1_extend_balanced
# (nofreq). Same data scaffolding as the phase1_chronos_mse builders, plus
# eagerly loads TSQA + M4 so the joint stage's _load_splits has nothing left
# to do at GPU-job start.
#
#SBATCH --job-name=build_cache_joint_phase1_extbal_nofreq
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --output=output_job/build_cache_joint_phase1_extbal_nofreq.out
#SBATCH --error=output_job/build_cache_joint_phase1_extbal_nofreq.err

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env
ENV_PY="$(pwd)/.env/bin/python"
if [[ ! -x "$ENV_PY" ]]; then
  echo "FATAL: no python interpreter at $ENV_PY"; exit 1
fi
echo "Using python: $ENV_PY"
"$ENV_PY" -c "import torch, sys; print(\"torch\", torch.__version__, \"|\", sys.executable)"


export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# How many worker processes HF Dataset .map calls fork during the synthetic
# arrow loader + univariate normaliser. The synthetic transpose + per-row
# fan-out are the only steps that meaningfully parallelise; the rest of the
# build is single-threaded (concatenate / split / save_to_disk / pickle).
export BUILD_CACHE_WORKERS="${BUILD_CACHE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

# Outer file-level parallelism: process N synthetic .arrow files in parallel
# using a ProcessPoolExecutor inside _load_or_build_chronos_raw. Each worker
# runs single-threaded internally (BUILD_CACHE_WORKERS=1 inside the worker)
# to avoid fork-on-fork. Default = min(8, num_files=17) → 8.
export BUILD_CACHE_FILE_WORKERS="${BUILD_CACHE_FILE_WORKERS:-8}"

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal

SYNTHETIC_DATA_DIR=/vast/s225250685/project/synthetic_data/unified/data

HOST_SHORT="$(hostname -s)"
export HF_HOME="${TMPDIR}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
unset TRANSFORMERS_CACHE || true
export HF_TOKEN=<HF_TOKEN_REDACTED>

echo "Job: ${SLURM_JOB_ID:-no_slurm}"
echo "Stage: joint_phase1_extend_balanced (cache build only — no training, no GPU)"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "Synthetic MV dir: ${SYNTHETIC_DATA_DIR}"

"$ENV_PY" curriculum_learning.py \
    --model OpenTSLMSPFixPos2D \
    --stages joint_phase1_extend_balanced \
    --llm_id google/gemma-3-270m \
    --num_epochs 1 \
    --pe_type sincos \
    --patch_size 32 \
    --enc_dec_type mlp \
    --chronos_window_sampling random \
    --chronos_random_samples_per_row 25 \
    --chronos_univariate \
    --synthetic_data_dir "${SYNTHETIC_DATA_DIR}" \
    --val_batch_size 4 \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim 256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim 256 \
    --device cpu \
    --build_cache_only
