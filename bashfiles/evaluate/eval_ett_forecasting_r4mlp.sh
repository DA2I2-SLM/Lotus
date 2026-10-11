#!/bin/bash
#SBATCH --job-name=eval_ett_fc
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

# Verify downstream forecasting quality on the ETT family (etth1/h2/m1/m2) for
# the r4/mlp/patch32/sincos checkpoints. Autoregressive multi-step decoding
# (the honest deployment metric), MSE/MAE per pred_len vs a persistence baseline.
set -euo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd ${FTS_REPO_ROOT}
export TOKENIZERS_PARALLELISM=false

DATA_ROOT=${FTS_ETT_DIR}
B=${FTS_SCRATCH}/FoundationalTS
REL=sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt

# name:ckpt  — ordered by relevance. phase-1 is the forecast-trained reference.
declare -A CKPTS=(
  [balanced_cotonly]="$B/results_harsleep_balanced/$REL"
  [balanced_fc]="$B/results_harsleep_balanced_fc/$REL"
  [phase1_extbal]="$B/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt"
)
ORDER=(balanced_cotonly balanced_fc phase1_extbal)

for name in "${ORDER[@]}"; do
  ck="${CKPTS[$name]}"
  echo "################ $name ################"
  if [ ! -f "$ck" ]; then echo "MISSING: $ck"; continue; fi
  for ds in etth1 etth2 ettm1 ettm2; do
    echo "==== $name / $ds ===="
    python evaluation/forecasting/eval_phase2_downstream.py \
      --ckpt "$ck" --llm_id google/gemma-3-270m --device cuda \
      --dataset "$ds" --data_root "$DATA_ROOT" \
      --patch_size 32 --pred_len 96 192 336 720 --max_cols 7 --batch_size 4 \
      --eval_split test \
      --lora_r 4 --lora_alpha 8 --enc_dec_type mlp \
      --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
      --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
      --pe_type sincos \
      --out_dir "${FTS_SCRATCH}/cik_work/fc_eval/$name/$ds" \
      2>&1 | grep -E "Summary|dataset|pred=|etth|ettm|MSE|nan|Error|Traceback" | grep -vE "it/s"
  done
  echo "==== DONE $name ===="
done
echo "ALL_FC_EVAL_COMPLETE"
