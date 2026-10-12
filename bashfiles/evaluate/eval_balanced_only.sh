#!/bin/bash
#SBATCH --job-name=ecg_bal_only
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=2:00:00
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
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opentslm
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/${USER}_bal_${SLURM_JOB_ID}
cd ${FTS_REPO_ROOT}
SNAP=/home/${USER}/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
CK=${FTS_RESULTS_ROOT}_harsleep_ecg_balanced/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt
python evaluation/cot/eval_cot_phase2.py --ckpt "$CK" --llm_id "$SNAP" \
  --tasks har sleep ecg --split test --max_samples 300 \
  --out_dir ${FTS_SCRATCH}/cik_work/ecg_eval_current/balanced 2>&1 | grep -E "Stage:|✓ |accuracy=|epoch=|Error|OOM"
echo "BALANCED_ONLY_COMPLETE"
