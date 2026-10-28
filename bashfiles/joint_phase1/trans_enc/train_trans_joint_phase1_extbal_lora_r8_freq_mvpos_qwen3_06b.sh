#!/bin/bash
# Joint Phase-1 (joint_phase1_extend_balanced_lora_r8) — TRANSFORMER + frequency
# embed + mv_position_ids + LLM LoRA r=8 alpha=16, Qwen3-0.6B backbone.
#
# First non-Gemma backbone. Preflighted before first submit:
#   - transformers 5.5.3 loads Qwen3ForCausalLM (hidden 1024, 28 layers)
#   - eos <|im_end|>, pad <|endoftext|>, bos None — all BOS sites in
#     OpenTSLMSPFixPos2D guard `bos_id is not None`
#   - EOS is injected at collate time (OpenTSLMProcessor holds the live
#     tokenizer), so the Gemma-built split cache is tokenizer-agnostic
#   - LoRA targets (q/k/v/o/gate/up/down_proj) exist in the Qwen3 arch;
#     CPU smoke test attached r=8 adapters: 5,046,272 params (0.84%)
# WATCH ITEM: V100 has no bf16 so weights+AMP run fp16; Qwen models have a
# history of fp16 overflow. If val loss goes NaN early, rerun with
# --llm_dtype fp32 (0.6B weights are only 2.4 GB in fp32).
#
# Checkpoints land in:
#   results/sincos_patch32/Qwen3_0_6B/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_r8_freq_mvpos/
#
#SBATCH --job-name=qwen06_lora_r8
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b_%j.out
#SBATCH --error=output_job/trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b_%j.err
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


cd "${FTS_REPO_ROOT}"

# ── Fresh-start cleanup ──────────────────────────────────────────────
CKPT_DIR="results/sincos_patch32/Qwen3_0_6B/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_r8_freq_mvpos"
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

TAG="${1:-run_trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b}"
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
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal_1b_debug
  echo "DEBUG MODE: cache=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
else
  # Reuse the "_1b" split cache DELIBERATELY, despite the name: it is the only
  # cache built with the current dataset code and contains the m4 rows. Rows
  # are stored raw — EOS/prompt/tokenization all happen at collate time with
  # the live tokenizer — so it is safe across backbones, Qwen included.
  # Same mixture as the Gemma 1B-r8 and 270M-r8-nofreq runs. Read-only reuse.
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal_1b
fi

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
echo "Stage: joint_phase1_extend_balanced_lora_r8 (Chronos UV + synthetic + TSQA + M4 + TSUMM)"
echo "LLM  : Qwen3-0.6B, LoRA r=8 alpha=16"
echo "Encoder/decoder: TRANSFORMER + frequency embedding + mv_position_ids"
echo "Trans sizing: enc d=192 L=6 H=8 ff=896 | dec d=192 L=4 H=8 ff=896"
echo "Synthetic MV dirs: ${SYNTHETIC_DATA_DIR}"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "Task weights chronos_uv:synthetic:tsqa:m4 = " \
     "${TASK_WEIGHT_CHRONOS_UV}:${TASK_WEIGHT_SYNTHETIC}:${TASK_WEIGHT_TSQA}:${TASK_WEIGHT_M4}"

# Pre-warm NFS cache on each compute node
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

    # Copy Qwen3-0.6B from the persistent cache (prefetched on the login node)
    PERM_HF="${HOME}/.cache/huggingface/hub"
    if [ -d "${PERM_HF}/models--Qwen--Qwen3-0.6B" ]; then
      cp -r "${PERM_HF}/models--Qwen--Qwen3-0.6B" "${HF_HUB_CACHE}/" 2>/dev/null || true
    fi
    # NOTE: do NOT set HF_HUB_OFFLINE here. Training loads the TSQA and M4
    # corpora via datasets.load_dataset() at startup, which needs the hub
    # API — offline mode killed an earlier run in 3 minutes with
    # OfflineModeIsEnabled on ChengsenWang/TSQA. The weight-cache copy
    # above still avoids the model download; the handful of tokenizer
    # hub pings from 8 ranks is well inside the API rate limit.
    if [ -d "${HF_HUB_CACHE}/models--Qwen--Qwen3-0.6B/snapshots" ]; then
      echo "[$(hostname)] Qwen3-0.6B weights staged from local cache"
    fi

    echo "[$(hostname)] HF_HOME=${HF_HOME}"
    df -h /tmp || true
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
        --stages joint_phase1_extend_balanced_lora_r8 \
        --llm_id Qwen/Qwen3-0.6B \
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
        --wandb_run_name "trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b" \
        --wandb_project "Time_Lang_Foundation"
'
