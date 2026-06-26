#!/bin/bash
# Zero-shot benchmark evaluation of OpenTSLMSPFixPos2D on ETTh1, ETTh2, Weather.
#
# Protocol matches Time-MoE run_eval.py:
#   - StandardScaler normalisation (fit on train)
#   - Channel-independent sliding-window test split
#   - MSE + MAE in normalised space
#   - Horizons 96 / 192 / 336 / 720
#   - Context lengths 512 and 2048
#
# Compares against TimeMoE-50M numbers from paper Table 3 (printed in output).
#
#SBATCH --job-name=our_model_benchmark_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/our_model_benchmark_eval.out
#SBATCH --error=output_job/our_model_benchmark_eval.err

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

# ── HF cache on local /tmp (avoid NFS stale-handle issues) ──────────────────
HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Copy pre-downloaded Gemma-3-270m weights
PERM_HF="${HOME}/.cache/huggingface/hub"
for MODEL_SNAP in "models--google--gemma-3-270m"; do
    if [ -d "${PERM_HF}/${MODEL_SNAP}" ]; then
        cp -r "${PERM_HF}/${MODEL_SNAP}" "${HF_HUB_CACHE}/" || true
        echo "[setup] copied ${MODEL_SNAP} from permanent HF cache"
    fi
done

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
DATA_DIR="/scratch/s225250685/project/TS-Reasoning/tc_lrpd/data/raw"

echo ""
echo "=========================================================="
echo " Benchmark evaluation — OpenTSLMSPFixPos2D"
echo " ckpt     : ${CKPT}"
echo " data_dir : ${DATA_DIR}"
echo " datasets : ETTh1 ETTh2 weather"
echo " contexts : 512 2048"
echo " horizons : 96 192 336 720"
echo " n_eval_windows : 500  seed=42"
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
    --out_json output_job/benchmark_results.json
