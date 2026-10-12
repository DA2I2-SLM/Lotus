#!/bin/bash
#SBATCH --job-name=otslm_diag
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
# Paper diagonal: OpenTSLM-SP and -Flamingo = each task with its OWN task-specific ckpt.
# No HF_HUB_OFFLINE (transformers tokenizer makes a spurious model_info API call).
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opentslm
export TOKENIZERS_PARALLELISM=false
cd ${FTS_REPO_ROOT}/hf_eval
OUTB=${FTS_SCRATCH}/FoundationalTS/hf_eval_results/diagonal
for model in sp flamingo; do
  for task in har sleep ecg; do
    repo="OpenTSLM/gemma-3-270m-pt-${task}-${model}"
    echo "############ $repo -> $task ($(date +%H:%M)) ############"
    python run_eval.py --repo_id "$repo" --tasks "$task" \
      --max_samples 300 --patch_size 4 --max_new_tokens 220 --batch_size 1 \
      --out_dir "$OUTB/${task}_${model}" > "$OUTB/${task}_${model}.log" 2>&1
    echo "exit=$?"; grep -iE "accuracy|macro_f1|Error|Traceback|OfflineMode|No module" "$OUTB/${task}_${model}.log" | tail -6
    echo "==== DONE ${task}_${model} ($(date +%H:%M)) ===="
  done
done
echo "ALL_DIAGONAL_COMPLETE"
