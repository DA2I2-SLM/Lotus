#!/bin/bash
#SBATCH --job-name=eval_time_benchmark
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

# Zero-shot TIME benchmark evaluation (all 49 datasets, short-horizon term).
#
# Pass the checkpoint as the first argument or edit CKPT below:
#   sbatch eval_time_benchmark.sh /path/to/best_model.pt
#
# To run all horizon terms (short + medium + long), add --term all.
#
# timebench is installed directly in the opentslm conda env (editable install
# from /scratch/s223540177/TIME with relaxed version pins).
#
# Prerequisites (already done):
#   /scratch/s223540177/TIME  — cloned and installed via pip install -e .
#   echo "TIME_DATASET=/path/to/time_datasets" >> /home/s223540177/FoundationalTS/.env

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

cd /home/s223540177/FoundationalTS

export TOKENIZERS_PARALLELISM=false
export TIME_DATASET=/scratch/s223540177/TIME_datasets


CKPT="${1:-/scratch/$(whoami)/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt}"
LLM=google/gemma-3-270m
OUT_DIR=./eval_results/time_benchmark

echo "Checkpoint : ${CKPT}"
echo "Output dir : ${OUT_DIR}"

python evaluation/time_benchmark/eval_time.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --patch_size         32 \
  --lora_r             4 \
  --lora_alpha         8 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --enc_dec_type       mlp \
  --pe_type            sincos \
  --term               short \
  --out_dir            "${OUT_DIR}"
