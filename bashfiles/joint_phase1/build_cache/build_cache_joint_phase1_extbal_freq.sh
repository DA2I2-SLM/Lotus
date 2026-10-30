#!/bin/bash
# Build the on-disk Chronos splits + stub cache for joint_phase1_extend_balanced
# (freq). --use_freq_embed doesn't change the data; the separate
# CHRONOS_PREPROCESSED_SPLITS_DIR matches the GPU script's expectation.
#
#SBATCH --job-name=build_cache_joint_phase1_extbal_freq
#SBATCH --partition=cpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G
#SBATCH --output=output_job/build_cache_joint_phase1_extbal_freq.out
#SBATCH --error=output_job/build_cache_joint_phase1_extbal_freq.err

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any LOTUS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${LOTUS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set LOTUS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job


cd "${LOTUS_REPO_ROOT}"
ENV_PY="${LOTUS_PYTHON}"
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

lotus_require_dir LOTUS_SYNTHETIC_DIR "synthetic MV arrow corpus (unified_*.arrow)"
SYNTHETIC_DATA_DIR=${LOTUS_SYNTHETIC_DIR}

HOST_SHORT="$(hostname -s)"
export HF_HOME="${TMPDIR}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
unset TRANSFORMERS_CACHE || true
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
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
    --use_freq_embed \
    --chronos_window_sampling random \
    --chronos_random_samples_per_row 25 \
    --chronos_univariate \
    ${SYNTHETIC_DATA_DIR:+--synthetic_data_dir "${SYNTHETIC_DATA_DIR}"} \
    --val_batch_size 4 \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim 256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim 256 \
    --device cpu \
    --build_cache_only
