#!/bin/bash
# Evaluate TimeMoE-50M on the held-out synthetic test split.
#
# Fair-comparison protocol vs. our model's val_mse_synthetic ≈ 0.35:
#   - Same data      : test_synthetic split from _preprocessed_chronos_uv_synth_phase1_mse
#   - Same preprocess: per-channel z-norm (std_floor=1e-3, clip=±20)
#   - Same window    : WINDOW_LEN=2048, PATCH_SIZE=32
#   - Same metric    : teacher-forced next-patch MSE in z-norm space
#
#SBATCH --job-name=timemoe_synthetic_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/timemoe_synthetic_eval.out
#SBATCH --error=output_job/timemoe_synthetic_eval.err

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any FTS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set FTS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=0
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

# Use local /tmp HF cache to avoid NFS stale-handle issues
HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Copy pre-downloaded TimeMoE weights if present in permanent cache,
# to avoid re-downloading on every run.
PERM_HF="${HOME}/.cache/huggingface/hub"
TIMEMOE_SNAP="models--Maple728--TimeMoE-50M"
if [ -d "${PERM_HF}/${TIMEMOE_SNAP}" ]; then
  cp -r "${PERM_HF}/${TIMEMOE_SNAP}" "${HF_HUB_CACHE}/" || true
  echo "[setup] copied TimeMoE weights from permanent cache"
fi

echo ""
echo "=========================================================="
echo " TimeMoE synthetic evaluation"
echo " splits_dir : _preprocessed_chronos_uv_synth_phase1_mse/all"
echo " n_samples  : 2000  seed=42"
echo " device     : cuda"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/timemoe_synthetic_eval.py \
    --splits_dir "_preprocessed_chronos_uv_synth_phase1_mse/all" \
    --n_samples 2000 \
    --seed 42 \
    --device cuda \
    --batch_size 16 \
    --model_id "Maple728/TimeMoE-50M"
