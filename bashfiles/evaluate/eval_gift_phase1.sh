#!/bin/bash
#SBATCH --job-name=gift_p1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

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

# GIFT-Eval zero-shot evaluation of OpenTSLM phase-1 on a representative subset.
set -euo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

export GIFT_EVAL=${FTS_SCRATCH}/gift_work/data
export HF_HOME=/tmp/${USER}_gift_hf_${SLURM_JOB_ID}
export FTS_ROOT=${FTS_REPO_ROOT}
export TOKENIZERS_PARALLELISM=false
export GIFT_OUT=${FTS_SCRATCH}/gift_work/results_phase1.csv
export GIFT_MODEL=opentslm_phase1

cd ${FTS_SCRATCH}/gift_work/gift-eval
python ${FTS_SCRATCH}/gift_work/gift_run.py
