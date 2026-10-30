#!/bin/bash
# LANGUAGE-ONLY Phase-1 (joint_phase1_tsqa_m4) — TSQA + M4 captioning, CE only,
# TRANSFORMER encoder + frequency embed + mv_position_ids + LoRA r=8 alpha=16,
# Gemma-3-270m backbone.
#
# H2 ablation arm: "language-only warm start". Uses Phase1JointTsqaM4Dataset,
# which contains NO Chronos/synthetic rows at all — chosen over zeroing the
# task weights on the joint stage because TaskWeightedDistributedBatchSampler
# floors every task at max(1, ...) batches/epoch, so weight-0 would still leak
# a trickle of forecasting gradient. Here the numerical corpora simply do not
# exist in the dataset. The regression decoder gets no gradient (stays at
# init); Phase 2 discards it anyway.
#
# LoRA r=8 registered for this stage in curriculum_learning.py
# stages_with_lora (2026-08 edit). Run dir carries lora_r8 via --run_suffix:
#   results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4_freq_mvpos_lora_r8/
#
# Data: TSQA 38.4k + M4 80k ≈ 118k samples/epoch → ~1.8k steps/epoch at
# batch 64: epochs are ~40 min, val fires every 1000 iters, early stopping
# (patience 5) will end this long before the 5-day cap.
#
#SBATCH --job-name=270m_lang_lora8
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/trans_tsqa_m4_lora_r8_freq_mvpos_270m_%j.out
#SBATCH --error=output_job/trans_tsqa_m4_lora_r8_freq_mvpos_270m_%j.err
#SBATCH --constraint=gpu-v100

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

set -euo pipefail
mkdir -p output_job


cd "${LOTUS_REPO_ROOT}"

# ── Fresh-start cleanup ──────────────────────────────────────────────
CKPT_DIR="${RESULTS_BASE_DIR}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4_freq_mvpos_lora_r8"
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

TAG="${1:-run_trans_tsqa_m4_lora_r8_freq_mvpos_270m}"
export TMPDIR=tmp
# LoRA r=8/alpha=16 for the language-only (H2) arm. joint_phase1_tsqa_m4 is
# intentionally absent from curriculum_learning.py's stages_with_lora map,
# because membership there would also switch LoRA on for the no-LoRA
# train_tsqa_m4_textonly.sh / train_tsqa_m4_patch16.sh launchers.
export FORCE_STAGE_LORA="8,16"
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
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
echo "Stage: joint_phase1_tsqa_m4 (LANGUAGE ONLY — TSQA + M4, CE, no forecasting)"
echo "LLM  : Gemma-3-270m, LoRA r=8 alpha=16"
echo "Encoder/decoder: TRANSFORMER + frequency embedding + mv_position_ids"
echo "Trans sizing: enc d=192 L=6 H=8 ff=896 | dec d=192 L=4 H=8 ff=896"

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

    # Stage model weights from the persistent cache; STAY ONLINE — TSQA and
    # M4 load from the hub at startup (see project memory: offline mode
    # killed an earlier run).
    PERM_HF="${HOME}/.cache/huggingface/hub"
    if [ -d "${PERM_HF}/models--google--gemma-3-270m" ]; then
      cp -r "${PERM_HF}/models--google--gemma-3-270m" "${HF_HUB_CACHE}/" 2>/dev/null || true
      echo "[$(hostname)] gemma-3-270m weights staged from local cache"
    fi

    echo "[$(hostname)] HF_HOME=${HF_HOME}"
    df -h /tmp || true
    find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

    "${LOTUS_TORCHRUN}" \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages joint_phase1_tsqa_m4 \
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
        --run_suffix mvpos_lora_r8 \
        --llm_mv_position_ids \
        --use_freq_embed \
        --val_batch_size 8 \
        --batch_size 8 \
        --grad_accum_steps 1 \
        --use_processor \
        --train_log_every_n_steps 10 \
        --eval_every_n_iters 1000 \
        --eval_max_samples_per_task 512 \
        --wandb_run_name "trans_tsqa_m4_lora_r8_freq_mvpos_270m" \
        --wandb_project "Time_Lang_Foundation"
'
