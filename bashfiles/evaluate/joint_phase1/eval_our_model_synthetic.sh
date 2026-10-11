#!/bin/bash
# Evaluate our pretrained OpenTSLMSPFixPos2D checkpoint on the held-out
# test_synthetic split — using the EXACT same protocol as val_mse_synthetic.
#
# Fair comparison vs TimeMoE-50M (MSE=0.4198):
#   - Same data      : test_synthetic split from _preprocessed_chronos_uv_synth_phase1_mse
#   - Same preprocess: per-channel z-norm via Phase1ChronosFuturePatchDataset
#   - Same metric    : compute_pretrain_loss(batch, mse_only=True) → recon_loss
#
#SBATCH --job-name=our_model_synthetic_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/our_model_synthetic_eval.out
#SBATCH --error=output_job/our_model_synthetic_eval.err

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
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

# Use local /tmp HF cache to avoid NFS stale-handle issues
HOST_SHORT="$(hostname -s)"
export LOCAL_SCRATCH_BASE="${TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}_${HOST_SHORT}"
export HF_HOME="${LOCAL_SCRATCH_BASE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Copy pre-downloaded model weights from permanent cache
PERM_HF="${HOME}/.cache/huggingface/hub"
for MODEL_SNAP in "models--google--gemma-3-270m"; do
  if [ -d "${PERM_HF}/${MODEL_SNAP}" ]; then
    cp -r "${PERM_HF}/${MODEL_SNAP}" "${HF_HUB_CACHE}/" || true
    echo "[setup] copied ${MODEL_SNAP} from permanent cache"
  fi
done

echo ""
echo "=========================================================="
echo " Our model synthetic evaluation"
echo " ckpt : results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
echo " n_samples : 2000  seed=42"
echo " device    : cuda"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/our_model_synthetic_eval.py \
    --ckpt "results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt" \
    --llm_id "google/gemma-3-270m" \
    --test_split_dir "_preprocessed_chronos_uv_synth_phase1_mse/all/test_synthetic" \
    --n_samples 2000 \
    --seed 42 \
    --device cuda \
    --batch_size 8
