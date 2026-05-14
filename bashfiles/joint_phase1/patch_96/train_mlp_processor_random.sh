#!/bin/bash
# Mirror of train_mlp_processor.sh, but with Chronos-style RANDOM window
# sampling instead of strided enumeration.
#
# What changes vs. train_mlp_processor.sh:
#   --chronos_window_sampling random       (was implicit "stride")
#   --chronos_random_samples_per_row 25    (NEW; tune per-dataset)
#   --chronos_window_stride is dropped     (unused in random mode)
#
# Semantics (matches Chronos's `ExpectedNumInstanceSampler` in
# chronos-forecasting/scripts/training/train.py):
#   * For each training row, emit N=chronos_random_samples_per_row copies of
#     an anchor-less stub. _sample_phase1_window then picks a fresh uniform
#     random start in [0, T - history_len - patch_size] on every __getitem__.
#   * Val/test stay on the deterministic last-window logic (no change).
#   * Downstream slicing, MV channel structure, normalisation: unchanged.
#
# How to pick N (--chronos_random_samples_per_row):
#   Total stubs/epoch ~= 520k rows * N. At batch=64 * grad_accum=8 = eff_batch
#   64 per opt step, opt_steps/epoch ~= 520k * N / 64 = 8125 * N. For ~Chronos
#   training scale (~200k opt steps over the full run at ~8 epochs), pick
#   N = 25. Smaller N = faster epoch / less coverage per row. Larger N = more
#   variety per row per epoch.
#
# Output / checkpoint paths follow the same results_dir layout as the strided
# script (results/sincos_patch32_mlp/...). If you want to keep the strided
# checkpoint, rename or back it up before launching this script.
#
#SBATCH --job-name=joint_phase1_mlp_proc_random
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/joint_phase1_mlp_processor_random.out
#SBATCH --error=output_job/joint_phase1_mlp_processor_random.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/time-series-foundation-model/FoundationalTS
conda activate ./.env
cd /scratch/s225250685/project/FoundationalTS


TAG="${1:-run_processor_random}"
export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export HF_TOKEN=<HF_TOKEN_REDACTED>
NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"
echo "Sampling: chronos_window_sampling=random, samples_per_row=25"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

srun --export=ALL --kill-on-bad-exit=1 \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=2 \
  bash -lc '
    set -euo pipefail

    echo "[$(hostname)] SLURM_NODEID=${SLURM_NODEID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    nvidia-smi -L || true

    HOST_SHORT="$(hostname -s)"
    export LOCAL_SCRATCH_BASE="${TMPDIR}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"

    export HF_HOME="${LOCAL_SCRATCH_BASE}"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

    unset TRANSFORMERS_CACHE || true

    export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

    echo "[$(hostname)] HF_HOME=${HF_HOME}"
    echo "[$(hostname)] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    echo "[$(hostname)] HF_HUB_CACHE=${HF_HUB_CACHE}"
    df -h /tmp || true

    if [ -n "${HF_TOKEN:-}" ]; then
      echo "[$(hostname)] Using HF_TOKEN from environment"
    else
      echo "[$(hostname)] HF_TOKEN not set, trying token file fallback"

      if [ -f "${HOME}/.cache/huggingface/token" ]; then
        cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
        chmod 600 "${HF_HOME}/token" || true
        echo "[$(hostname)] Copied token from ~/.cache/huggingface/token"
      elif [ -f "${HOME}/.huggingface/token" ]; then
        cp "${HOME}/.huggingface/token" "${HF_HOME}/token"
        chmod 600 "${HF_HOME}/token" || true
        echo "[$(hostname)] Copied token from ~/.huggingface/token"
      else
        echo "[$(hostname)] No saved HF token file found"
      fi
    fi

    find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=2 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages phase1_chronos_mse \
        --llm_id google/gemma-3-270m \
        --num_epochs 20 \
        --pe_type sincos \
        --patch_size 32 \
        --enc_dec_type mlp \
        --chronos_window_sampling random \
        --chronos_random_samples_per_row 25 \
        --val_batch_size 4 \
        --batch_size 64 \
        --grad_accum_steps 8 \
        --mlp_enc_hidden_dim 64 \
        --mlp_enc_ff_dim 256 \
        --mlp_dec_hidden_dim 64 \
        --mlp_dec_ff_dim 256 \
        --use_processor
'
