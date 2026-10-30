#!/bin/bash
#SBATCH --job-name=p1_tsqa_m4_lora
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any LOTUS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${LOTUS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set LOTUS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"
#
# Phase-1 TEXT-ONLY pretraining: joint TSQA (MCQ) + M4 (captioning), CE-only.
# NO Chronos / forecasting / MSE (the joint_phase1_tsqa_m4 stage is CE-only by design).
# Arch matches the user's other checkpoints: sincos / patch32 / enc_dec=mlp (64/256) / nofreq.
# LoRA r4/alpha8 enabled via FORCE_STAGE_LORA env hook (needs the _enable_lora_if_needed patch).
# Output: results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4_nofreq/

set -euo pipefail
mkdir -p output_job


cd "${LOTUS_REPO_ROOT}"

export RESULTS_BASE_DIR="${LOTUS_RESULTS_ROOT}_tsqam4_lora"

# Enable LoRA r=4, alpha=8 for this stage (requires the _enable_lora_if_needed env hook).
export FORCE_STAGE_LORA="4,8"

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

NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29503}"

echo "Phase-1 TSQA+M4 (text-only, CE-only, no forecasting)"
echo "Nodes: ${NNODES}  MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "Results base: ${RESULTS_BASE_DIR}"

MAX_RESTARTS=10
for try in $(seq 1 ${MAX_RESTARTS}); do
  echo "[outer-retry] attempt ${try}/${MAX_RESTARTS}"
  set +e
  srun --export=ALL --kill-on-bad-exit=1 \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    bash -lc '
      set -euo pipefail
      HOST_SHORT="$(hostname -s)"
      export LOCAL_SCRATCH_BASE="/tmp/${USER}/hf_${SLURM_JOB_ID}_${HOST_SHORT}"
      export HF_HOME="${LOCAL_SCRATCH_BASE}"
      export HF_DATASETS_CACHE="${HF_HOME}/datasets"
      export HF_HUB_CACHE="${HF_HOME}/hub"
      export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
      unset TRANSFORMERS_CACHE || true
      export HF_DATASETS_DISABLE_PROGRESS_BARS=1
      mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

      if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
        cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
        chmod 600 "${HF_HOME}/token" || true
      fi
      find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

      "${LOTUS_TORCHRUN}" \
        --nnodes='"${NNODES}"' \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
        --rdzv_id='"${SLURM_JOB_ID}"'_'"${try}"' \
        --node_rank="${SLURM_NODEID}" \
        "${LOTUS_REPO_ROOT}/curriculum_learning.py" \
          --model OpenTSLMSPFixPos2D \
          --stages joint_phase1_tsqa_m4 \
          --llm_id google/gemma-3-270m \
          --num_epochs 20 \
          --pe_type sincos \
          --patch_size 32 \
          --enc_dec_type mlp \
          --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
          --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
          --batch_size 2 \
          --val_batch_size 4 \
          --grad_accum_steps 8 \
          --save_every_n_steps 800 \
          --gradient_checkpointing
    '
  EXIT_CODE=$?
  set -e
  echo "[outer-retry] srun exit=${EXIT_CODE}"
  if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[outer-retry] phase-1 TSQA+M4 completed cleanly"
    break
  fi
  if [ ${try} -ge ${MAX_RESTARTS} ]; then
    echo "[outer-retry] hit max restarts; exiting ${EXIT_CODE}"
    exit ${EXIT_CODE}
  fi
  sleep 10
done
