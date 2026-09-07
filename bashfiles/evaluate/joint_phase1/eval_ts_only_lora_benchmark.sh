#!/bin/bash
# ETTh1/ETTh2/Weather benchmark eval — TS-only LoRA model (phase1_chronos_mse_lora_freq_mvpos).
# Matches Time-MoE run_eval.py protocol (StandardScaler, channel-independent, ctx=512/2048).
#
#SBATCH --job-name=eval_bm_ts_only_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/eval_bm_ts_only_lora.out
#SBATCH --error=output_job/eval_bm_ts_only_lora.err

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=0
export HF_TOKEN=<HF_TOKEN_REDACTED>

HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

PERM_HF="${HOME}/.cache/huggingface/hub"
for MODEL_SNAP in "models--google--gemma-3-270m"; do
  if [ -d "${PERM_HF}/${MODEL_SNAP}" ]; then
    cp -r "${PERM_HF}/${MODEL_SNAP}" "${HF_HUB_CACHE}/" || true
    echo "[setup] copied ${MODEL_SNAP} from permanent cache"
  fi
done

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="/scratch/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"

echo ""
echo "=========================================================="
echo " Benchmark eval — TS-only LoRA (phase1_chronos_mse_lora_freq_mvpos)"
echo " ckpt     : ${CKPT}"
echo " data_dir : ${DATA_DIR}"
echo " datasets : ETTh1 ETTh2 weather  ctx: 512 2048  H: 96 192 336 720"
echo " lora_r=4  lora_alpha=8"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/our_model_benchmark_eval.py \
    --ckpt "${CKPT}" \
    --llm_id "google/gemma-3-270m" \
    --data_dir "${DATA_DIR}" \
    --datasets ETTh1 ETTh2 weather \
    --context_lengths 512 2048 \
    --horizons 96 192 336 720 \
    --n_eval_windows 500 \
    --seed 42 \
    --device cuda \
    --outer_bs 256 \
    --inner_bs 16 \
    --lora_r 4 \
    --lora_alpha 8 \
    --out_json output_job/benchmark_results_ts_only_lora.json
