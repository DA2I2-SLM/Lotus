#!/bin/bash
#SBATCH --job-name=eval_all4
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

# Resolve repo root + interpreter + data locations dynamically so this
# script runs on any account. Override any FTS_* var in your environment.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -f "$_d/bashfiles/_paths.sh" ]; do _d="$(dirname "$_d")"; done
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

# Evaluate the CURRENT best_model.pt of all 4 phase-2 trainings on HAR/Sleep/ECG.
# CP3 is a freq checkpoint (needs --use_freq_embed). FC also gets an ETT forecasting eval.
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/${USER}_evalall4_${SLURM_JOB_ID}
cd ${FTS_REPO_ROOT}
SNAP=/home/${USER}/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
B=${FTS_SCRATCH}/FoundationalTS
OUTB=${FTS_SCRATCH}/cik_work/eval_all4
DATA_ROOT=${FTS_ETT_DIR}

# name : ckpt-path : freq-flag
CP1="$B/results_balecg_cp1_extbal/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt"
CP2="$B/results_balecg_cp2_chronosmse/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt"
CP3="$B/results_balecg_cp3_chronosfreq/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_freq/checkpoints/best_model.pt"
FC="$B/results_fc_fullhorizon/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt"

cot_eval() {  # name ckpt extra_flags
  local name=$1 ck=$2 extra=${3:-}
  echo "############ CoT EVAL $name ($(date +%H:%M)) ############"
  [ -f "$ck" ] || { echo "MISSING $ck"; return; }
  python evaluation/cot/eval_cot_phase2.py --ckpt "$ck" --llm_id "$SNAP" \
    --tasks har sleep ecg --split test --max_samples 300 $extra \
    --out_dir "$OUTB/$name" 2>&1 | grep -E "Stage:|✓ |accuracy=|epoch=|Error|OOM"
  echo "==== DONE CoT $name ($(date +%H:%M)) ===="
}

cot_eval cp1_extbal    "$CP1"
cot_eval cp2_chronosmse "$CP2"
cot_eval cp3_chronosfreq "$CP3" "--use_freq_embed"
cot_eval fc_fullhorizon "$FC"

# FC additionally: ETT forecasting eval (does the full-horizon MSE fix help?)
echo "############ ETT FORECAST EVAL fc_fullhorizon ($(date +%H:%M)) ############"
for ds in etth1 etth2 ettm1 ettm2; do
  python evaluation/forecasting/eval_phase2_downstream.py \
    --ckpt "$FC" --llm_id "$SNAP" --device cuda \
    --dataset "$ds" --data_root "$DATA_ROOT" \
    --patch_size 32 --pred_len 96 192 336 720 --max_cols 7 --batch_size 4 --eval_split test \
    --lora_r 4 --lora_alpha 8 --enc_dec_type mlp \
    --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
    --pe_type sincos --out_dir "$OUTB/fc_forecast/$ds" \
    2>&1 | grep -E "Summary|etth|ettm|pred=|MSE|nan|Error" | grep -vE "it/s"
done
echo "ALL_EVAL4_COMPLETE"
