#!/bin/bash
#SBATCH --job-name=etth1-forecast-eval-mlp
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
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
module load Anaconda3
source activate
conda activate opentslm

cd ${FTS_REPO_ROOT}

RESULTS=results/eval
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=${FTS_ETT_DIR}/ETTh1.csv

# ── MLP-backbone joint_phase1_extend_balanced checkpoint ────────────
# Matches the results_dir written by train_mlp.sh, which adds a "_mlp"
# suffix to the patch tag to keep MLP and transformer runs separate.
CKPT=${FTS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse/checkpoints/best_model.pt
STAGE=phase1_chronos_mse

DATA=monash_australian_electricity

OUT_DIR=./eval_results/${DATA}/${STAGE}_transformer/phase1_pretrain/autoregressive_pure_mse
mkdir -p ${OUT_DIR}

python ${INFER} \
  --ckpt            ${CKPT} \
  --llm_id          ${LLM} \
  --stage_name      ${STAGE} \
  --eval_format     phase1_pretrain \
  --etth1_csv       ${ETTh1} \
  --mode            autoregressive \
  --device          cuda \
  --batch_size      32 \
  --patch_size      32 \
  --history_len     256 \
  --pred_len        32 \
  --plot_every      10 \
  --enc_dec_type    transformer \
  --out_dir         ${OUT_DIR} \
  --forecast_backbone llm \
  --eval_split      test \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim 256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim 256 \
  --source_dataset ${DATA} \
  2>&1 | tee ${OUT_DIR}/run_log.txt