#!/bin/bash
# Phase-1 Chronos MSE pretraining — NO frequency embedding, WITH LLM LoRA (r=4, alpha=8).
#
# Stage: phase1_chronos_mse_lora
# Identical data, sampler, and LRs to train_phase1_chronos_mse_nofreq.sh, but
# the Gemma-3-270M backbone is lightly adapted via LoRA while the frozen-LLM run
# continues in parallel. Enables a direct apples-to-apples comparison of frozen
# vs. LoRA-adapted LLM on Chronos MSE pretraining.
#
#SBATCH --job-name=phase1_chronos_mse_nofreq_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/phase1_chronos_mse_nofreq_lora.out
#SBATCH --error=output_job/phase1_chronos_mse_nofreq_lora.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any FTS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set FTS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"

set -euo pipefail
mkdir -p output_job


cd "${FTS_REPO_ROOT}"

# ── Fresh-start cleanup ──────────────────────────────────────────────
CKPT_DIR="results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_nofreq"
if [[ "${KEEP_CKPT:-0}" != "1" ]]; then
  if [[ -e "${CKPT_DIR}" ]]; then
    echo "[cleanup] removing ${CKPT_DIR} for fresh-start training"
    rm -rf "${CKPT_DIR}"
  else
    echo "[cleanup] no existing ${CKPT_DIR}; nothing to remove"
  fi
else
  echo "[cleanup] KEEP_CKPT=1 set; preserving ${CKPT_DIR}"
fi

TAG="${1:-run_phase1_chronos_mse_nofreq_lora}"
export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK_WEIGHT_CHRONOS_UV="${TASK_WEIGHT_CHRONOS_UV:-1.0}"
TASK_WEIGHT_SYNTHETIC="${TASK_WEIGHT_SYNTHETIC:-2.0}"

DEBUG_FLAG=""
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_FLAG="--debug"
fi

if [[ -n "${DEBUG_FLAG}" ]]; then
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse_lora_debug
  echo "🐛 DEBUG MODE: cache=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
else
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse
fi

SYNTHETIC_DATA_DIR=${FTS_SYNTHETIC_DIR}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
# NOTE: never hard-code a token here. Read it from the environment, falling back to the
# standard huggingface-cli credential file.
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"
echo "Stage: phase1_chronos_mse_lora (Chronos-only MSE pretraining + LLM LoRA r=4)"
echo "Encoder: time-domain only (use_freq_embed=OFF)"
echo "Synthetic MV dir: ${SYNTHETIC_DATA_DIR}"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

srun --export=ALL --kill-on-bad-exit=1 \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
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

    "${FTS_TORCHRUN}" \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages phase1_chronos_mse_lora \
        --llm_id google/gemma-3-270m \
        --num_epochs 100 \
        --pe_type sincos \
        --patch_size 32 \
        --pe_base_c "${PE_BASE_C:-20}" \
        --enc_dec_type mlp \
        --chronos_window_sampling random \
        --chronos_random_samples_per_row 25 \
        --chronos_univariate \
        --synthetic_data_dir '"${SYNTHETIC_DATA_DIR}"' \
        --task_weight_chronos_uv '"${TASK_WEIGHT_CHRONOS_UV}"' \
        --task_weight_synthetic  '"${TASK_WEIGHT_SYNTHETIC}"' \
        '"${DEBUG_FLAG}"' \
        --val_batch_size 4 \
        --batch_size 16 \
        --grad_accum_steps 1 \
        --mlp_enc_hidden_dim 64 \
        --mlp_enc_ff_dim 256 \
        --mlp_dec_hidden_dim 64 \
        --mlp_dec_ff_dim 256 \
        --use_processor \
        --train_log_every_n_steps 10 \
        --eval_every_n_iters 1000 \
        --wandb_run_name "phase1_chronos_mse_lora_nofreq" \
        --wandb_project "Time_Lang_Foundation"
'
