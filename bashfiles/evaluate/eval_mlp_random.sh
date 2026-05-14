#!/bin/bash
# Eval counterpart for train_mlp_processor_random.sh (Chronos-style random
# anchor sampling). Inference itself is identical to eval_mlp.sh — the
# sampling mode only affects HOW the model was trained, not how it forecasts.
# We just point at the same checkpoint and write results to a sibling out_dir
# so stride-trained and random-trained eval artefacts stay separate.
#
# Note on checkpoint paths:
#   The results_dir built by curriculum_learning.py does NOT include the
#   sampling mode (only patch_size, pe_type, enc_dec_type, optional flags).
#   So both train_mlp_processor.sh and train_mlp_processor_random.sh write to
#   the SAME directory; whichever finished last is the one this script reads.
#   If you want to keep both, rename or copy best_model.pt between runs.
#
#SBATCH --job-name=etth1-forecast-eval-mlp-random
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
module load Anaconda3
source activate
# conda activate opentslm

cd /scratch/s225250685/project/time-series-foundation-model/FoundationalTS
conda activate ./.env
cd  /scratch/s225250685/project/FoundationalTS

RESULTS=results/eval
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=/scratch/s225250685/project/time-series-foundation-model/dataset/ETT-small/ETTh1.csv

# ── MLP-backbone phase1_chronos_mse checkpoint (random-anchor training) ──
# Same path as the strided run; the sampling mode lives in the training
# config, not the results_dir.
CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse/checkpoints/best_model.pt
STAGE=phase1_chronos_mse

python ${INFER} \
  --ckpt            ${CKPT} \
  --llm_id          ${LLM} \
  --stage_name      ${STAGE} \
  --eval_format     phase1_pretrain \
  --etth1_csv       ${ETTh1} \
  --mode            autoregressive \
  --device          cuda \
  --batch_size      32 \
  --patch_size 32 \
  --history_len     576 \
  --pred_len        192 \
  --plot_every      200 \
  --enc_dec_type    mlp \
  --out_dir         ./eval_results/${STAGE}_mlp_random/phase1_pretrain/autoregressive_pure_mse \
  --forecast_backbone llm \
  --eval_split      test \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim 256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim 256
