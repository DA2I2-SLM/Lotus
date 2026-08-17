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
cd /home/s223540177/FoundationalTS/hf_eval
OUTB=/scratch/s223540177/FoundationalTS/hf_eval_results

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
