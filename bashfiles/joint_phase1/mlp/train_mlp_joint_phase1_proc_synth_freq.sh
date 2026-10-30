#!/bin/bash
# Joint Phase-1 training (joint_phase1_extend_balanced) on the univariate-Chronos
# + synthetic-MV corpus, with the OpenTSLMProcessor collate — WITH frequency
# embedding.
#
# "Joint" Phase-1 = three tasks trained together with task-balanced (1:1:1)
# sampling:
#   * Chronos next-patch MSE   <- registry (univariate) + synthetic (MV)
#   * TSQA  MCQ  (cross-entropy)
#   * M4    captioning (cross-entropy)
# TSQA and M4 are loaded independently and are NOT affected by --chronos_univariate
# or --synthetic_data_dir — those two flags only shape the Chronos sub-corpus.
#
# Encoder: --use_freq_embed adds a frequency-domain branch to the MLP encoder's
# patch embedding — each patch is rfft'd, embedded independently, concatenated
# with the time-domain patch embedding, and projected back to
# transformer_input_dim.
#
# vs. train_mlp_joint_phase1_proc_synth.sh (the nofreq joint script):
#   * --use_freq_embed is added here.
#   * results_dir = results/sincos_patch32_mlp_seed_42_freq/...
#     (the trainer appends "_seed_<data_seed>_freq" when --use_freq_embed is on),
#     so this run never clobbers the nofreq joint checkpoints in
#     results/sincos_patch32_mlp/...
#   * CHRONOS_PREPROCESSED_SPLITS_DIR is per-script (no longer shared) — freq is
#     an encoder-side change so the corpus / split is identical, but each run
#     owns its own cache to avoid concurrent-write races.
#
# Chronos corpus controls (same as the nofreq joint script):
#   --chronos_univariate            registry Chronos kept strictly univariate.
#   --synthetic_data_dir <dir>      unified_*.arrow MV datasets folded in.
#   --chronos_window_sampling random + --chronos_random_samples_per_row 25
#
# Checkpoints: results/sincos_patch32_mlp_seed_42_freq/.../joint_phase1_extend_balanced/...
#
# Batch size 16 / grad_accum 16 over 8 GPUs (2 nodes x 4) — global eff. batch
# (C up to 10, history up to 2048) on a 32 GB V100, matching the phase1_chronos
# scripts after the OOM fix.
#
#SBATCH --job-name=joint_phase1_extbal_mlp_proc_synth_freq
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/joint_phase1_extbal_mlp_proc_synth_freq.out
#SBATCH --error=output_job/joint_phase1_extbal_mlp_proc_synth_freq.err
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
# Remove the stage's checkpoint dir before launching training so the run
# always starts from a clean slate with the current code. Override with:
#   KEEP_CKPT=1 sbatch <this script>
CKPT_DIR="${RESULTS_BASE_DIR}/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq"
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


TAG="${1:-run_joint_phase1_proc_synth_freq}"
export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Per-task sampler weights for joint_phase1_extend_balanced. Default 1:2:1:1
# (chronos_uv : synthetic : tsqa : m4). The chronos_uv pool is tiny (~125
# stubs vs ~500k synthetic) so it's heavily oversampled — synthetic still
# carries the bulk of the chronos gradient, but uv contributes a fixed share
# instead of being a rounding error.
TASK_WEIGHT_CHRONOS_UV="${TASK_WEIGHT_CHRONOS_UV:-1.0}"
TASK_WEIGHT_SYNTHETIC="${TASK_WEIGHT_SYNTHETIC:-2.0}"
TASK_WEIGHT_TSQA="${TASK_WEIGHT_TSQA:-1.0}"
TASK_WEIGHT_M4="${TASK_WEIGHT_M4:-1.0}"

# Split-cache dir for the univariate-Chronos + synthetic-MV corpus.
# Isolated per-script (no longer shared with the nofreq joint or the
# phase1_chronos_mse scripts) so each run owns its own _parts/ and all/ rebuild.
# Read at module import time; --export=ALL propagates it to the srun tasks.
# DEBUG=1 enables tiny-corpus debug mode (small caps + few anchors + own cache).
DEBUG_FLAG=""
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_FLAG="--debug"
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal_debug
  echo "🐛 DEBUG MODE: cache=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
else
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal
fi

# Directory of unified_*.arrow synthetic multivariate datasets.
lotus_require_dir LOTUS_SYNTHETIC_DIR "synthetic MV arrow corpus (unified_*.arrow; see README, Data / Phase 1)" || exit 1
SYNTHETIC_DATA_DIR=${LOTUS_SYNTHETIC_DIR}

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
echo "Stage: joint_phase1_extend_balanced (Chronos MSE + TSQA + M4)"
echo "Encoder: time + frequency embedding (use_freq_embed=ON)"
echo "Chronos: univariate registry + synthetic MV; sampling=random, N=25"
echo "Synthetic MV dir: ${SYNTHETIC_DATA_DIR}"
echo "Task weights chronos_uv : synthetic : tsqa : m4 = " \
     "${TASK_WEIGHT_CHRONOS_UV} : ${TASK_WEIGHT_SYNTHETIC} : ${TASK_WEIGHT_TSQA} : ${TASK_WEIGHT_M4}"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

# Pre-warm the OS page cache for the cache dir on each compute node. The cache
# lives on /vast NFS; cold reads during validation deadlocked NCCL collectives
# in a previous run because one rank blocked while others waited in all_reduce.
# A single sequential pass through every cache file populates each node's NFS
# client cache so subsequent reads come from RAM at ~GB/s instead of NFS.
PREWARM_START=$(date +%s)
echo "[$(date)] pre-warming NFS cache on each node …"
srun --export=ALL --kill-on-bad-exit=0 \
  --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -lc "
    cd ${LOTUS_REPO_ROOT}
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

    "${LOTUS_TORCHRUN}" \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages joint_phase1_extend_balanced \
        --llm_id google/gemma-3-270m \
        --num_epochs 100 \
        --pe_type sincos \
        --patch_size 32 \
        --pe_base_c "${PE_BASE_C:-10}" \
        --enc_dec_type mlp \
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
        --batch_size 16 \
        --grad_accum_steps 1 \
        --mlp_enc_hidden_dim 64 \
        --mlp_enc_ff_dim 256 \
        --mlp_dec_hidden_dim 64 \
        --mlp_dec_ff_dim 256 \
        --use_processor \
        --train_log_every_n_steps 10 \
        --eval_every_n_iters 1000 \
        --eval_max_samples_per_task 512 \
        --wandb_run_name "joint_phase1_extbal_mlp_proc_synth_freq" \
        --wandb_project "Time_Lang_Foundation"
'
