#!/bin/bash
# Joint Phase-1 (joint_phase1_extend_balanced) — TRANSFORMER + LoRA LLM
# + frequency embed + mv_position_ids.
#
# ABLATION RUN 2/2 — TS+Language (Chronos UV + synthetic MV + TSQA + M4 + TSUMM).
# Counterpart: train_trans_phase1_chronos_mse_lora_freq_mvpos.sh (TS-only).
#
# Architecture identical to train_trans_joint_phase1_extbal_freq_mvpos.sh
# plus LLM LoRA (r=4, alpha=8) via stage joint_phase1_extend_balanced_lora.
#
# Transformer sizing (1.5× MLP+LoRA reference, 6,176,736 enc/dec params):
#   enc: d=192 L=6 H=8 ff=896  |  dec: d=192 L=4 H=8 ff=896
#
# Extra synthetic data: TSUMM-Suite_Training forecasting (38.9K rows, 363MB)
# added via colon-separated synthetic_data_dir (TS+Language only).
#
#SBATCH --job-name=ts_lang_lora_mvpos
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/ts_lang_lora_mvpos.out
#SBATCH --error=output_job/ts_lang_lora_mvpos.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06],v100-f-[14-15]

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

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd ${FTS_REPO_ROOT}
conda activate ./.env

# ── Fresh-start cleanup ──────────────────────────────────────────────
CKPT_DIR="results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos"
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

TAG="${1:-run_ts_lang_lora_mvpos}"
export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK_WEIGHT_CHRONOS_UV="${TASK_WEIGHT_CHRONOS_UV:-1.0}"
TASK_WEIGHT_SYNTHETIC="${TASK_WEIGHT_SYNTHETIC:-1.0}"
TASK_WEIGHT_TSQA="${TASK_WEIGHT_TSQA:-0.75}"
TASK_WEIGHT_M4="${TASK_WEIGHT_M4:-0.75}"

DEBUG_FLAG=""
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_FLAG="--debug"
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal_debug
  echo "DEBUG MODE: cache=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
else
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal_lora
fi

# Base synthetic + TSUMM forecasting (TS+Language only)
SYNTHETIC_DATA_DIR=${FTS_SYNTHETIC_DIR}:${FTS_TSUMM_DIR}

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
echo "Stage: joint_phase1_extend_balanced_lora (TS+Language + LLM LoRA r=4 alpha=8)"
echo "Encoder/decoder: TRANSFORMER + freq embed + mv_position_ids"
echo "Trans sizing: enc d=192 L=6 H=8 ff=896 | dec d=192 L=4 H=8 ff=896"
echo "Synthetic MV dirs: ${SYNTHETIC_DATA_DIR}"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "Task weights chronos_uv:synthetic:tsqa:m4 = ${TASK_WEIGHT_CHRONOS_UV}:${TASK_WEIGHT_SYNTHETIC}:${TASK_WEIGHT_TSQA}:${TASK_WEIGHT_M4}"

# Pre-warm NFS cache
PREWARM_START=$(date +%s)
echo "[$(date)] pre-warming NFS cache on each node …"
srun --export=ALL --kill-on-bad-exit=0 \
  --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -lc "
    cd ${FTS_REPO_ROOT}
    sz_before=\$(du -sh \"${CHRONOS_PREPROCESSED_SPLITS_DIR}/all\" 2>/dev/null | cut -f1)
    find \"${CHRONOS_PREPROCESSED_SPLITS_DIR}/all\" -type f -exec cat {} + > /dev/null 2>/dev/null || true
    echo \"  [\$(hostname)] pre-warmed \${sz_before}\"
  " || echo "[warn] pre-warm step failed; continuing"
echo "[$(date)] pre-warm took $(( $(date +%s) - PREWARM_START )) seconds"

srun --export=ALL --kill-on-bad-exit=1 \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  bash -lc '
    set -euo pipefail
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8

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

    if [ -n "${HF_TOKEN:-}" ]; then
      echo "[$(hostname)] Using HF_TOKEN from environment"
    else
      if [ -f "${HOME}/.cache/huggingface/token" ]; then
        cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"; chmod 600 "${HF_HOME}/token" || true
      elif [ -f "${HOME}/.huggingface/token" ]; then
        cp "${HOME}/.huggingface/token" "${HF_HOME}/token"; chmod 600 "${HF_HOME}/token" || true
      fi
    fi
    find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

    ${FTS_TORCHRUN} \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages joint_phase1_extend_balanced_lora \
        --llm_id google/gemma-3-270m \
        --num_epochs 100 \
        --pe_type sincos \
        --patch_size 32 \
        --enc_dec_type transformer \
        --trans_enc_dim 192 \
        --trans_enc_num_layers 6 \
        --trans_enc_num_heads 8 \
        --trans_enc_ff_dim 896 \
        --trans_dec_dim 192 \
        --trans_dec_num_layers 4 \
        --trans_dec_num_heads 8 \
        --trans_dec_ff_dim 896 \
        --run_suffix mvpos \
        --llm_mv_position_ids \
        --use_freq_embed \
        --chronos_window_sampling random \
        --chronos_random_samples_per_row 25 \
        --chronos_univariate \
        --synthetic_data_dir '"${SYNTHETIC_DATA_DIR}"' \
        --task_weight_chronos_uv '"${TASK_WEIGHT_CHRONOS_UV}"' \
        --task_weight_synthetic  '"${TASK_WEIGHT_SYNTHETIC}"' \
        --task_weight_tsqa       '"${TASK_WEIGHT_TSQA}"' \
        --task_weight_m4         '"${TASK_WEIGHT_M4}"' \
        '"${DEBUG_FLAG}"' \
        --val_batch_size 8 \
        --batch_size 8 \
        --grad_accum_steps 1 \
        --use_processor \
        --train_log_every_n_steps 10 \
        --eval_every_n_iters 1000 \
        --eval_max_samples_per_task 512 \
        --wandb_run_name "ts_lang_lora_mvpos" \
        --wandb_project "Time_Lang_Foundation"
'
