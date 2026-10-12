#!/bin/bash
#SBATCH --job-name=hf_pt_ecg
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

# EM + F1 for the published OpenTSLM ECG checkpoints (SP and Flamingo) on HAR/Sleep/ECG.
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
# NOTE: no HF_HUB_OFFLINE — transformers' tokenizer load makes a spurious
# is_base_mistral() model_info() API call that offline mode blocks. Model files
# are cached, so only that tiny API call needs network.
export TOKENIZERS_PARALLELISM=false
cd ${FTS_REPO_ROOT}/hf_eval
OUTB=${FTS_SCRATCH}/FoundationalTS/hf_eval_results

for repo in OpenTSLM/gemma-3-270m-pt-ecg-sp OpenTSLM/gemma-3-270m-pt-ecg-flamingo; do
  tag=$(basename "$repo")
  echo "############ $repo ($(date +%H:%M)) ############"
  python run_eval.py --repo_id "$repo" --tasks har sleep ecg \
    --max_samples 300 --patch_size 4 --max_new_tokens 220 --batch_size 1 \
    --out_dir "$OUTB/$tag" > "$OUTB/${tag}_full.log" 2>&1
  echo "exit=$?"; grep -iE "accuracy|macro_f1|Error|Traceback|OfflineMode|No module|✓" "$OUTB/${tag}_full.log" | tail -20
  echo "==== DONE $tag ($(date +%H:%M)) ===="
done
echo "ALL_HF_PT_ECG_COMPLETE"
