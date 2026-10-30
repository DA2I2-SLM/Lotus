#!/bin/bash
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-License-Identifier: MIT
#SBATCH --job-name=abl_270m
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
# ─────────────────────────────────────────────────────────────────────────────
# Gemma3-270M Phase-2 ablations (paper Table 4).
#
# Every arm runs the SAME Phase-2 recipe as the main 270M run
# (train_joint5_ours.sh: joint_phase2_downstream, 5 tasks, balanced_all with
# PHASE2_BALANCE_TARGET=40000, PHASE2_GEN_EM_N=192, patience 8, batch 1 x
# grad-accum 16, 4 GPUs) and differs ONLY in the Phase-1 warm start and the
# frequency branch. One sbatch per arm:
#
#   sbatch --job-name=abl_270m_<arm> bashfiles/joint_phase2/joint5/train_ablation_270m.sh <arm>
#
#   arm              | Table 4 row                              | Phase-1 warm start (under $LOTUS_PRETRAINED_DIR)                    | Phase-2 freq branch
#   -----------------+------------------------------------------+-------------------------------------------------------------------+---------------------------------
#   control          | (a) LOTUS                                 | joint_phase1_extend_balanced_freq                                 | on  (--use_freq_embed)
#   nofreq_phase2    | (b) w/o frequency branch (Phase 2 only)   | joint_phase1_extend_balanced_freq (same ckpt as control)          | off; freq tensors dropped from
#                    |                                           |                                                                   |      the warm start (ABLATION_DROP_FREQ_KEYS=1)
#   nofreq_both      | (c) w/o frequency branch (both phases)    | joint_phase1_extend_balanced_lora_nofreq (MLP encoder, no freq)   | off
#   langonly_phase1  | (d) Language-only Phase 1 (same budget)   | gemma_3_270m_joint_phase1_tsqa_m4_langonly_lora_r8_freq_mvpos     | on
#
# Warm-start checkpoints are the released ones in hiepnh137/LangTimeFoundationModel
# (see docs/hf_model_card.md); download them with
#   huggingface-cli download hiepnh137/LangTimeFoundationModel --local-dir "$LOTUS_PRETRAINED_DIR"
# Export WARM_START_PRETRAINED_CKPT before sbatch to warm-start an arm from a
# different file (e.g. your own Phase-1 run of
# bashfiles/joint_phase1/train_tsqa_m4_textonly.sh for the MLP language-only
# variant, or bashfiles/joint_phase1/trans_enc/train_trans_tsqa_m4_lora_r8_freq_mvpos_270m.sh
# for the transformer one).
#
# Encoder flags follow the warm-start checkpoint, because the Phase-2 encoder
# must have the tensor shapes the checkpoint was trained with:
#   * control / nofreq_phase2 / nofreq_both : MLP encoder 64/256 (as train_joint5_ours.sh)
#   * langonly_phase1 : the released language-only checkpoint was trained with the
#     transformer encoder (d=192, L=6, H=8, ff=896; decoder L=4) and LoRA r=8/alpha=16,
#     so this arm uses those flags (the Phase-1 adapter is inherited automatically
#     by the warm-start code in curriculum_learning.py). Override with ABL_ENC_FLAGS
#     if you point WARM_START_PRETRAINED_CKPT at an MLP language-only checkpoint.
#
# How "nofreq_phase2" works: the control checkpoint carries freq_embed/concat_proj
# tensors that an encoder built without --use_freq_embed has no slots for.
# ABLATION_DROP_FREQ_KEYS=1 makes _load_weights_from_stage() in
# curriculum_learning.py drop exactly those keys and load everything else strict.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p output_job

cd "${LOTUS_REPO_ROOT}"

ARM="${1:?usage: train_ablation_270m.sh <control|nofreq_phase2|nofreq_both|langonly_phase1>}"

MLP_ENC_FLAGS="--enc_dec_type mlp --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256"
TRANS_ENC_FLAGS="--enc_dec_type transformer --trans_enc_dim 192 --trans_enc_num_layers 6 --trans_enc_num_heads 8 --trans_enc_ff_dim 896 --trans_dec_dim 192 --trans_dec_num_layers 4 --trans_dec_num_heads 8 --trans_dec_ff_dim 896"

case "$ARM" in
  control)
    _default_ckpt="${LOTUS_PRETRAINED_DIR}/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt"
    FREQ_FLAG="--use_freq_embed"; export ABLATION_DROP_FREQ_KEYS=0
    ENC_FLAGS="${ABL_ENC_FLAGS:-${MLP_ENC_FLAGS}}" ;;
  nofreq_phase2)
    _default_ckpt="${LOTUS_PRETRAINED_DIR}/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt"
    FREQ_FLAG="";                 export ABLATION_DROP_FREQ_KEYS=1
    ENC_FLAGS="${ABL_ENC_FLAGS:-${MLP_ENC_FLAGS}}" ;;
  nofreq_both)
    _default_ckpt="${LOTUS_PRETRAINED_DIR}/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt"
    FREQ_FLAG="";                 export ABLATION_DROP_FREQ_KEYS=0
    ENC_FLAGS="${ABL_ENC_FLAGS:-${MLP_ENC_FLAGS}}" ;;
  langonly_phase1)
    _default_ckpt="${LOTUS_PRETRAINED_DIR}/gemma_3_270m_joint_phase1_tsqa_m4_langonly_lora_r8_freq_mvpos/checkpoints/best_model.pt"
    FREQ_FLAG="--use_freq_embed"; export ABLATION_DROP_FREQ_KEYS=0
    ENC_FLAGS="${ABL_ENC_FLAGS:-${TRANS_ENC_FLAGS}}" ;;
  *) echo "arm must be control|nofreq_phase2|nofreq_both|langonly_phase1 (got '$ARM')" >&2; exit 1 ;;
esac

export RESULTS_BASE_DIR="${LOTUS_RESULTS_ROOT}_ablation_270m_${ARM}"
export WARM_START_PRETRAINED_CKPT="${WARM_START_PRETRAINED_CKPT:-${_default_ckpt}}"
[ -f "$WARM_START_PRETRAINED_CKPT" ] || { echo "ERROR: warm-start checkpoint not found: $WARM_START_PRETRAINED_CKPT (download with: huggingface-cli download hiepnh137/LangTimeFoundationModel --local-dir \"$LOTUS_PRETRAINED_DIR\")" >&2; exit 2; }

# Phase-2 recipe: identical to train_joint5_ours.sh
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
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
# collective desync hangs forever instead of aborting.
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

echo "=== ABLATION (Table 4) arm=${ARM} ==="
echo "Warm start : ${WARM_START_PRETRAINED_CKPT}"
echo "Freq flag  : '${FREQ_FLAG}'  ABLATION_DROP_FREQ_KEYS=${ABLATION_DROP_FREQ_KEYS}"
echo "Encoder    : ${ENC_FLAGS}"
echo "Results    : ${RESULTS_BASE_DIR}"
echo "Nodes: ${NNODES}"
echo "Node list:"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

export ABL_FREQ_FLAG="${FREQ_FLAG}"
export ABL_ENC_FLAGS_RESOLVED="${ENC_FLAGS}"

# Outer retry loop: each iteration is a fresh srun -> bash -> torchrun
# launch with fresh c10d rendezvous. Python exits 42 on memory-leak restart;
# we re-launch the whole srun stack until python exits 0 (training done).
MAX_RESTARTS=30
for try in $(seq 1 ${MAX_RESTARTS}); do
  echo "[outer-retry] attempt ${try}/${MAX_RESTARTS}"
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

      # ABL_FREQ_FLAG / ABL_ENC_FLAGS_RESOLVED are intentionally unquoted:
      # they are flag lists resolved by the arm table above.
      # shellcheck disable=SC2086
      "${LOTUS_TORCHRUN}" \
        --nnodes='"${NNODES}"' \
        --nproc_per_node=4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
        --rdzv_id='"${SLURM_JOB_ID}"'_'"${try}"' \
        --node_rank="${SLURM_NODEID}" \
        "${LOTUS_REPO_ROOT}/curriculum_learning.py" \
          --model OpenTSLMSPFixPos2D \
          --stages joint_phase2_downstream \
          --llm_id google/gemma-3-270m \
          --num_epochs 20 \
          --pe_type sincos \
          --pe_base_c 10 \
          --patch_size 32 \
          ${ABL_ENC_FLAGS_RESOLVED} \
          ${ABL_FREQ_FLAG} \
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
echo "DONE_ABLATION_${ARM}"
