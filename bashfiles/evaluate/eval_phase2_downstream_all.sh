#!/bin/bash
#SBATCH --job-name=eval_p2ds_all
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

# Evaluates joint_phase2_downstream checkpoint on all 7 standard long-horizon
# benchmarks across pred_lens {96, 192, 336, 720}.
#
# Each dataset uses multivariate samples (all columns, up to --max_cols)
# matching the training distribution exactly.  Evaluation uses teacher-forced
# decoding (same forward pass as training loss), which is faster and more
# reproducible than autoregressive mode.
#
# Pass the checkpoint path as the first argument or edit CKPT below:
#   sbatch eval_phase2_downstream_all.sh /path/to/best_model.pt
#
# To use autoregressive decoding instead, add --autoregressive.

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

cd /home/s223540177/FoundationalTS

export TOKENIZERS_PARALLELISM=false

CKPT="${1:-/scratch/s223540177/FoundationalTS/result/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt}"
LLM=google/gemma-3-270m
DATA_ROOT=/scratch/s223540177/TimeSeriesECM/dataset
OUT_DIR=./eval_results/joint_phase2_downstream/all

echo "Checkpoint : ${CKPT}"
echo "Data root  : ${DATA_ROOT}"
echo "Output dir : ${OUT_DIR}"

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
  --lora_r             32 \
  --lora_alpha         64 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       transformer \
  --pe_type            sincos \
  --out_dir            "${OUT_DIR}"

python /home/s223540177/FoundationalTS/evaluation/forecasting/eval_chatts.py --ckpt /scratch/s223540177/FoundationalTS/result/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt --llm_id google/gemma-3-270m --patch_size 32  --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 --mlp_dec_hidden_dim 64  --mlp_dec_ff_dim 256 --lora_r 32 --lora_alpha 64 --batch_size 8 --out_dir ./eval_results/chatts



export TOKENIZERS_PARALLELISM=false

# CKPT="${1:-/scratch/s223540177/FoundationalTS/result/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt}"
CKPT="${1:-/scratch/s223540177/FoundationalTS/result/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend/checkpoints/best_model.pt}"

LLM=google/gemma-3-270m
DATA_ROOT=/scratch/s223540177/TimeSeriesECM/dataset
OUT_DIR=./eval_results/joint_phase2_downstream/all

echo "Checkpoint : ${CKPT}"
echo "Data root  : ${DATA_ROOT}"
echo "Output dir : ${OUT_DIR}"

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
  --lora_r             32 \
  --lora_alpha         64 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       transformer \
  --pe_type            sincos \
  --out_dir            "${OUT_DIR}"


