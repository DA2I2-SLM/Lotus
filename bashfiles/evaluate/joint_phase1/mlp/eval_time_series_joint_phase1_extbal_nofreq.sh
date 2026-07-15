#!/bin/bash
# Time-series eval (forecasting) for the MLP joint_phase1_extend_balanced run
# trained by bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth.sh.
#
# Runs forecasting once per source (chronos_uv, synthetic). TSQA + M4 evals
# are in the sibling eval_language_*.sh script.
#
# Outputs:
#   eval_results/mlp_joint_phase1_extbal_nofreq/time_series/chronos_uv/
#   eval_results/mlp_joint_phase1_extbal_nofreq/time_series/synthetic/
#
#SBATCH --job-name=eval-ts-mlp-joint-extbal-nofreq
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x.out
#SBATCH --error=output_job/%x.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /home/s223540177/FoundationalTS
conda activate opentslm

ENV_PY="/home/s223540177/.conda/envs/opentslm/bin/python"
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal

LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=/scratch/s223540177/TimeSeriesECM/dataset/ETT-small/ETTh1.csv

CKPT=/scratch/s223540177/FoundationalTS/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt
STAGE=joint_phase1_extend_balanced
OUT_BASE=./eval_results/mlp_joint_phase1_extbal_nofreq/time_series

echo "CHRONOS_PREPROCESSED_SPLITS_DIR=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "CKPT=${CKPT}"

for SRC in chronos_uv synthetic; do
  OUT=${OUT_BASE}/${SRC}
  echo "================================================================"
  echo "[eval_time_series] source=${SRC} -> ${OUT}"
  echo "================================================================"
  ${ENV_PY} ${INFER} \
    --ckpt              ${CKPT} \
    --llm_id            ${LLM} \
    --stage_name        ${STAGE} \
    --eval_format       phase1_pretrain \
    --etth1_csv         ${ETTh1} \
    --mode              autoregressive \
    --device            cuda \
    --batch_size        32 \
    --patch_size        32 \
    --history_len       576 \
    --pred_len          192 \
    --plot_every        50 \
    --enc_dec_type      mlp \
    --out_dir           ${OUT} \
    --forecast_backbone llm \
    --eval_split        test \
    --source_filter     ${SRC} \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim     256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim     256 \
    --lora_r             4 \
    --lora_alpha         8
done
