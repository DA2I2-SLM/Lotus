#!/bin/bash
# Evaluation counterpart for train_mlp_processor.sh.
#
# fixed_inference.py builds its own DataLoader (with the legacy
# extend_time_series_to_match_patch_size_and_aggregate collate) and goes
# through the legacy `pad_and_apply_batch` path, so --use_processor is a
# training-time switch only and is not added here. The checkpoint layout
# is unchanged by --use_processor, so this script reads from the same
# CKPT path that train_mlp.sh / train_mlp_processor.sh write to.
#
#SBATCH --job-name=etth1-forecast-eval-mlp-proc
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

cd /scratch/s225250685/project/time-series-foundation-model/FoundationalTS
conda activate ./.env
cd /scratch/s225250685/project/FoundationalTS

RESULTS=results/eval
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=/scratch/s225250685/project/time-series-foundation-model/dataset/ETT-small/ETTh1.csv

# MLP-backbone joint_phase1_extend_balanced checkpoint — written by
# train_mlp.sh OR train_mlp_processor.sh (same path; the processor flag
# does not change the results_dir layout).
CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced/checkpoints/periodic_checkpoint.pt
STAGE=joint_phase1_chronos_mse

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
  --history_len     576 \
  --pred_len        192 \
  --plot_every      200 \
  --enc_dec_type    mlp \
  --out_dir         ./eval_results/${STAGE}_mlp_processor/phase1_pretrain/autoregressive_pure_mse \
  --forecast_backbone llm \
  --eval_split      test \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim 256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim 256
