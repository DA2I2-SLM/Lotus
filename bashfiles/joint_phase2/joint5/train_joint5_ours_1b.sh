#!/bin/bash
#SBATCH --job-name=j5_ours_1b
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gpus=8
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

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
#
# ─────────────────────────────────────────────────────────────────────────────
# Phase-2 joint downstream training for the gemma-3-1b backbone.
#
# This is train_joint5_ours.sh with the BACKBONE and ENCODER swapped and the
# recipe held fixed: same stage, same task mix, same balancing, same selection
# criterion, same patience, same EFFECTIVE BATCH. Any difference in the final
# table is therefore attributable to the model, not the recipe.
#
# Warm start: hiepnh137/LangTimeFoundationModel ->
#   gemma_3_1b_pt_joint_phase1_extend_balanced_lora_r8_freq_mvpos
# downloaded 2026-08-12 to pretrained_weights/ (780 MB).
#
# The phase-1 architecture below was NOT taken from the checkpoint's name — it
# was read out of the checkpoint's tensor shapes and then verified by a strict
# (strict=True) load of every module plus the LoRA adapter
# (${FTS_VAST}/cik_work/gemma1b/smoke_load_1b.py, job 3476109):
#   patch_embed (192,1,32)   -> patch 32, d_model 192
#   freq_embed  (192,34)     -> use_freq_embed, rfft(32) -> 17 bins -> 34
#   encoder.layers.0..5, linear1 (896,192) -> 6 layers, ff 896
#   no output_proj           -> output_dim == d_model (trans_enc_dim sets both)
#   decoder 4 layers d192 ff896; projector 192 -> 1152 -> 1152
#   llm_input_embeddings (262147, 1152); lora_config r=8 alpha=16,
#   base_model_name_or_path = google/gemma-3-1b-pt
# Encoder params 3,052,032 and decoder 1,983,520 match, to the parameter, the
# "1.5x" sizing documented in
#   bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_freq_1p5x.sh
#
# NHEAD=8 is the one value tensor shapes cannot reveal (all 14 divisors of 192
# give an identical param count and state_dict signature, so strict loads and
# logs can never expose it) — and the CODE DEFAULT IS 12, so this must be passed
# explicitly. It was settled from the trained weights themselves: the QK-circuit
# head-block statistic sum_h ||Wq[h]^T Wk[h]||_F, z-scored against row
# permutations that preserve (q_i,k_i) pairing but destroy contiguous head
# blocks, peaks at h=8 in 6/6 encoder layers, with a random-init negative
# control showing no peak at 8 and the 270m sibling checkpoint (whose in-repo
# launcher explicitly passes 8) peaking at 8 in 6/6. Loss-based probes were
# tried first and are all UNPOWERED here — do not reopen this on their basis.
#
# DELTAS vs train_joint5_ours.sh (270m), each deliberate:
#   1. --llm_id google/gemma-3-1b-pt          (was google/gemma-3-270m)
#   2. --enc_dec_type transformer + --trans_{enc,dec}_* replace the --mlp_* flags
#   3. warm-start ckpt + RESULTS_BASE_DIR point at the 1b tree
#   4. qos batch-long / 10 days (270m needed ~4.5-5 h/epoch at world 8; this
#      backbone has 697.8M non-embedding params vs 100.3M)
#   5. --llm_dtype bf16 stated explicitly, and the job runs on bf16-native GPUs.
#
# HARDWARE — chosen from measurements, not intuition:
#   * dtype: "auto" resolves via torch.cuda.is_bf16_supported(), which in torch
#     2.9.1 defaults to including_emulation=True and returns True even on a V100
#     (sm_70, no bf16 tensor cores). Proof: the published 270m phase-2 ckpt has
#     llm_input_embeddings in bfloat16 and scaler_state == {} — exactly what
#     GradScaler(enabled=False) emits, i.e. the AMP dtype was bf16. So "pin a
#     V100 to get fp16" is false; it only buys bf16 without tensor cores.
#   * world size: the published 270m run was NOT 4 GPUs. sacct -j 3458673 shows
#     gres/gpu=8, node=2 (l40s-f-03 + v100-f-06) and its log prints "Distributed
#     training with 8 GPUs". At batch 1 x accum 16 that is an effective batch of
#     128. Running 4 ranks here would silently halve it to 64 and change the LR
#     schedule, so world size MUST stay 8.
#   * memory is not the constraint: the 270m peaked at 2.08 GB allocated on both
#     a 31.7 GB V100 and a 44.4 GB L40S (the dominant tensors are 262147-vocab
#     logits, which do not scale with the backbone); the 1b lands near 3 GB.
#   * placement: all four l40s nodes were fully allocated at submit time, while
#     rtxp6000l-f-01 was IDLE with 8 free GPUs — and 8 ranks on ONE node avoids
#     inter-node NCCL altogether. Fallback if that node is unavailable:
#       --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --constraint=gpu-l40s
#       --cpus-per-task=16 --mem=200G   with NPROC=4  (world 8, proven topology)
#
# NOT changed, on purpose: --pe_base_c 10. Phase 1 used the argparse default
# 10000, so both the 270m and 1b runs make the same 10000 -> 10 transition
# entering phase 2. Matching "ours" matters more than matching this
# checkpoint's own phase 1, and the encoder is fully trainable in phase 2.
#
# Env overrides for smoke tests: NPROC (ranks/node), SMOKE_EPOCHS,
# SMOKE_BALANCE_TARGET. Unset -> the real run.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

export USER_NAME=$(whoami)
cd /home/${USER_NAME}/FoundationalTS

TAG="${1:-run}"
NPROC="${NPROC:-8}"

# FULL_FT=1 runs the capacity-ablation arm: the whole 1b backbone is trained
# instead of a LoRA adapter, matching the tokenized baseline's ~1B trainable so
# that "which interface is better" is no longer confounded with "which arm has
# 90x the capacity". FORCE_FULL_LLM_FT is read by _is_full_ft() in
# curriculum_learning.py, which routes the optimizer to the full-LLM param
# group, suppresses LoRA (mutually exclusive), and merges the phase-1 adapter
# into the base weights so both arms start from the same function.
if [ "${FULL_FT:-0}" = "1" ]; then
  export FORCE_FULL_LLM_FT=1
  _ARM_SUFFIX="_fullft"
  echo "### ARM: FULL FINE-TUNE (whole backbone trainable, no LoRA) ###"
else
  _ARM_SUFFIX=""
fi
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${FTS_RESULTS_ROOT}_joint5_ours_1b${_ARM_SUFFIX}}"
export WARM_START_PRETRAINED_CKPT=${FTS_PRETRAINED_DIR}/gemma_3_1b_pt_joint_phase1_extend_balanced_lora_r8_freq_mvpos/checkpoints/best_model.pt

# Fail fast and loudly rather than "⚠️  Warm-start failed — starting fresh",
# which would quietly train a randomly-initialised encoder for days.
if [ ! -f "${WARM_START_PRETRAINED_CKPT}" ]; then
  echo "FATAL: warm-start checkpoint missing: ${WARM_START_PRETRAINED_CKPT}"
  exit 1
fi

# ── phase-2 task mix: identical to train_joint5_ours.sh ──────────────────────
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
# TSQA removed: 80.8% of its phase-2 test rows were in phase-1 training.
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET="${SMOKE_BALANCE_TARGET:-40000}"
export PHASE2_SELECT_BY_GEN=1
export PHASE2_GEN_EM_N=192
export PHASE2_EARLY_STOP_PAT=8
EPOCHS="${SMOKE_EPOCHS:-20}"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=warn
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# NOT set: TORCH_NCCL_BLOCKING_WAIT=1 disables the NCCL watchdog thread, so a
# collective desync hangs forever instead of aborting (cost us ~2 days on 3455465).
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

unset GLOO_SOCKET_IFNAME || true
unset NCCL_SOCKET_IFNAME || true
export GLOO_DEVICE_TRANSPORT=tcp
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-^lo,docker0,virbr0,br-*,veth*}"

NNODES="${SLURM_NNODES:-1}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
MASTER_ADDR="$(echo "${NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29514}"

echo "Nodes: ${NNODES} x ${NPROC} ranks = world $(( NNODES * NPROC ))"
echo "${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "Job: ${SLURM_JOB_ID:-no_slurm} Tag: ${TAG} Epochs: ${EPOCHS} Balance: ${PHASE2_BALANCE_TARGET}"
echo "Backbone: google/gemma-3-1b-pt (hidden 1152, 26 layers) | dtype bf16"
echo "Encoder:  TransformerCNNEncoder2D d192 L6 H8 ff896 + freq, patch 32"
echo "Warm start: ${WARM_START_PRETRAINED_CKPT}"
echo "Results:    ${RESULTS_BASE_DIR}/sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_freq"

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
      elif [ -f "${HOME}/.cache/huggingface/token" ]; then
        cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
        chmod 600 "${HF_HOME}/token" || true
        echo "[$(hostname)] Copied token from ~/.cache/huggingface/token"
      else
        echo "[$(hostname)] No saved HF token file found"
      fi

      find "${HF_HOME}" -name "*.lock" -type f -delete 2>/dev/null || true

      torchrun \
        --nnodes='"${NNODES}"' \
        --nproc_per_node='"${NPROC}"' \
        --rdzv_backend=c10d \
        --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
        --rdzv_id='"${SLURM_JOB_ID}"'_'"${try}"' \
        --node_rank="${SLURM_NODEID}" \
        /home/'"${USER_NAME}"'/FoundationalTS/curriculum_learning.py \
          --model OpenTSLMSPFixPos2D \
          --stages joint_phase2_downstream \
          --llm_id google/gemma-3-1b-pt \
          --llm_dtype bf16 \
          --num_epochs '"${EPOCHS}"' \
          --pe_type sincos \
          --pe_base_c 10 \
          --patch_size 32 \
          --enc_dec_type transformer \
          --use_freq_embed \
          --trans_enc_dim 192 \
          --trans_enc_num_layers 6 \
          --trans_enc_num_heads 8 \
          --trans_enc_ff_dim 896 \
          --trans_dec_dim 192 \
          --trans_dec_num_layers 4 \
          --trans_dec_num_heads 8 \
          --trans_dec_ff_dim 896 \
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
  set -e
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
