#!/bin/bash
# Synthetic test-split evaluation — TS+Language LoRA model (joint_phase1_extend_balanced_lora_freq_mvpos).
# Uses the SAME test_synthetic split as the baseline for fair comparison.
#
#SBATCH --job-name=eval_syn_ts_lang_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --output=output_job/eval_syn_ts_lang_lora.out
#SBATCH --error=output_job/eval_syn_ts_lang_lora.err

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

CKPT="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos/checkpoints/best_model.pt"

echo ""
echo "=========================================================="
echo " Synthetic eval — TS+Language LoRA (joint_phase1_extend_balanced_lora_freq_mvpos)"
echo " ckpt : ${CKPT}"
echo " n_samples : 2000  seed=42  lora_r=4  lora_alpha=8"
echo "=========================================================="
echo ""

nvidia-smi -L || true

./.env/bin/python evaluation/forecasting/our_model_synthetic_eval.py \
    --ckpt "${CKPT}" \
    --llm_id "google/gemma-3-270m" \
    --test_split_dir "_preprocessed_chronos_uv_synth_phase1_mse_lora/all/test" \
    --filter_source synthetic \
    --n_samples 2000 \
    --seed 42 \
    --device cuda \
    --batch_size 8 \
    --lora_r 4 \
    --lora_alpha 8
