#!/bin/bash
#SBATCH --job-name=j5_ours
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gpus=4
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
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

TAG="${1:-run}"

# ─────────────────────────────────────────────────────────────────────────────
# Warm-start loads the pretrained checkpoint DIRECTLY via `--warm_start_from
# pretrained` (see curriculum_learning.py), reading WARM_START_PRETRAINED_CKPT.
# Default: the paper's 270M Phase-1 run (joint_phase1_extend_balanced_freq).
# Export WARM_START_PRETRAINED_CKPT before sbatch to point at a different file.
# ─────────────────────────────────────────────────────────────────────────────

export RESULTS_BASE_DIR="${FTS_RESULTS_ROOT}_joint5_ours"
export WARM_START_PRETRAINED_CKPT="${WARM_START_PRETRAINED_CKPT:-${FTS_PRETRAINED_DIR}/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt}"
[ -f "$WARM_START_PRETRAINED_CKPT" ] || { echo "ERROR: warm-start checkpoint not found: $WARM_START_PRETRAINED_CKPT (download with: huggingface-cli download hiepnh137/LangTimeFoundationModel --local-dir \"$FTS_PRETRAINED_DIR\")" >&2; exit 2; }

# HAR+Sleep-only phase-2 (no forecasting, no ChatTS)
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
# TSQA removed: 80.8% of its phase-2 test rows were in phase-1 training.
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=40000
export PHASE2_SELECT_BY_GEN=1
export PHASE2_GEN_EM_N=192
export PHASE2_EARLY_STOP_PAT=8


export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# NOT set: TORCH_NCCL_BLOCKING_WAIT=1 disables the NCCL watchdog thread, so a
# collective desync hangs forever instead of aborting (cost ~2 days on an earlier run).
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"

NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29511}"

echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG}"
echo "Warm start: --warm_start_from pretrained (PRETRAINED_WARM_START_CKPT)"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

# Outer retry loop: each iteration is a fresh srun -> bash -> torchrun
# launch with fresh c10d rendezvous. Python exits 42 on memory-leak restart;
# we re-launch the whole srun stack until python exits 0 (training done).
MAX_RESTARTS=30
for try in $(seq 1 ${MAX_RESTARTS}); do
  echo "[outer-retry] attempt ${try}/${MAX_RESTARTS}"
  # Disable set -e around srun: srun returns non-zero when python exits 42
  # for the mem-restart workaround, and we want the loop to handle it, not
  # have the script abort.
  set +e
  srun --export=ALL \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    bash -lc '
      set -euo pipefail

      echo "[$(hostname)] SLURM_NODEID=${SLURM_NODEID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      nvidia-smi -L || true

      HOST_SHORT="$(hostname -s)"
      export LOCAL_SCRATCH_BASE="/tmp/${USER}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"

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
        --rdzv_id='"${SLURM_JOB_ID}"'_'"${try}"' \
        --node_rank="${SLURM_NODEID}" \
        "${FTS_REPO_ROOT}/curriculum_learning.py" \
          --model OpenTSLMSPFixPos2D \
          --stages joint_phase2_downstream \
          --llm_id google/gemma-3-270m \
          --num_epochs 20 \
          --pe_type sincos \
          --pe_base_c 10 \
          --patch_size 32 \
          --enc_dec_type mlp \
          --use_freq_embed \
          --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
          --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
          --batch_size 1 \
          --val_batch_size 1 \
          --max_tokens_per_batch 600 \
          --grad_accum_steps 16 \
          --mse_alpha 1.0 \
          --windows_per_col_train 0 \
          --dataloader_workers 0 \
          --save_every_n_steps 800 \
          --gradient_checkpointing \
          --warm_start_from pretrained
    '
  EXIT_CODE=$?
  set -e  # re-enable for safety after srun
  echo "[outer-retry] srun exit=${EXIT_CODE}"
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[outer-retry] training completed cleanly"
    break
  fi
  if [ ${try} -ge ${MAX_RESTARTS} ]; then
    echo "[outer-retry] hit max restarts; exiting ${EXIT_CODE}"
    exit ${EXIT_CODE}
  fi
  echo "[outer-retry] restarting after exit=${EXIT_CODE}"
  sleep 10
done
