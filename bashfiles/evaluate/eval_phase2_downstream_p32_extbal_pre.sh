#!/bin/bash
#SBATCH --job-name=eval_p2_extbal_pre
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

# Evaluates the joint_phase2_downstream checkpoint trained from extend_balanced
# pretrained Phase-1 weights (MLP enc/dec, patch=32, LoRA r=4/alpha=8).
#
# Runs:
#   1. eval_phase2_downstream.py  — LTSF benchmark (MSE/MAE, all 7 datasets)
#   2. eval_chatts.py             — ChatTS ROUGE-1/2/L + CE loss

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

export USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

export TOKENIZERS_PARALLELISM=false

CKPT="/scratch/${USER_NAME}/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt"
LLM="google/gemma-3-270m"
DATA_ROOT="/scratch/${USER_NAME}/TimeSeriesECM/dataset"
OUT_BASE="/scratch/${USER_NAME}/FoundationalTS/eval_results/joint_phase2_downstream_from_pretrained_nofreq"

echo "Checkpoint : ${CKPT}"
echo "Data root  : ${DATA_ROOT}"
echo "Output base: ${OUT_BASE}"

# ── 1. LTSF forecasting benchmark ────────────────────────────────────────────
python evaluation/forecasting/eval_phase2_downstream.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --all_datasets \
  --data_root          "${DATA_ROOT}" \
  --patch_size         32 \
  --pred_len           96 192 336 720 \
  --max_cols           16 \
  --batch_size         4 \
  --eval_split         test \
  --lora_r             4 \
  --lora_alpha         8 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       mlp \
  --pe_type            sincos \
  --out_dir            "${OUT_BASE}/forecasting"

# ── 2. ChatTS ROUGE evaluation ────────────────────────────────────────────────
python evaluation/forecasting/eval_chatts.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --patch_size         32 \
  --lora_r             4 \
  --lora_alpha         8 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       mlp \
  --pe_type            sincos \
  --batch_size         8 \
  --out_dir            "${OUT_BASE}/chatts"
