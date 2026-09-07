#!/bin/bash
# ETTh1/ETTh2/Weather benchmark eval — TS-only LoRA, ctx=1024 and ctx=3072.
# Companion to eval_ts_only_lora_benchmark.sh (which covers ctx=512/2048).
# Together they match the two-run setup used on the baseline model.
#
#SBATCH --job-name=eval_bm_ts_only_lora_xc
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/eval_bm_ts_only_lora_xc.out
#SBATCH --error=output_job/eval_bm_ts_only_lora_xc.err

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
GEMMA_LOCAL="/vast/s225250685/models/gemma-3-270m"

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="/scratch/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"

echo ""
echo "=========================================================="
echo " Benchmark eval (extra ctx) — TS-only LoRA (phase1_chronos_mse_lora_freq_mvpos)"
echo " ckpt     : ${CKPT}"
echo " contexts : 1024 3072  (512 and 2048 in companion job)"
echo " horizons : 96 192 336 720"
echo " datasets : ETTh1 ETTh2 weather"
echo " lora_r=4  lora_alpha=8"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/our_model_benchmark_eval.py \
    --ckpt             "${CKPT}" \
    --llm_id           "${GEMMA_LOCAL}" \
    --data_dir         "${DATA_DIR}" \
    --datasets         ETTh1 ETTh2 weather \
    --context_lengths  1024 3072 \
    --horizons         96 192 336 720 \
    --n_eval_windows   500 \
    --seed             42 \
    --device           cuda \
    --outer_bs         256 \
    --inner_bs         16 \
    --lora_r           4 \
    --lora_alpha       8 \
    --out_json         output_job/benchmark_results_ts_only_lora_extra_ctx.json
