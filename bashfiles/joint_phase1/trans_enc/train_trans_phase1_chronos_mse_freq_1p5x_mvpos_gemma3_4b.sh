#!/bin/bash
# Phase-1 Chronos MSE pretraining — TRANSFORMER encoder/decoder, WITH frequency
# embedding, no LoRA. Same trans sizing as the 270M run (1.5× MLP+LoRA reference).
#
# Stage: phase1_chronos_mse  (Chronos-only, MSE-only — no TSQA, no M4, no CE).
#
# Encoder:  TransformerCNNEncoder2D + rfft frequency branch (--use_freq_embed).
# Decoder:  TransformerCNNDecoder2D.
# LLM:      Gemma-3-4B, FROZEN (no LoRA, no full FT).
#           hidden_size=2560 (vs 640 for 270M) — projector dim auto-inferred.
#
# Trans sizing (same as 270M run):
#   --trans_enc_dim 192 --trans_enc_num_layers 6 --trans_enc_num_heads 8
#   --trans_enc_ff_dim 896
#   --trans_dec_dim 192 --trans_dec_num_layers 4 --trans_dec_num_heads 8
#   --trans_dec_ff_dim 896
#
#SBATCH --job-name=trans_phase1_chronos_mse_freq_1p5x_mvpos_4b
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=5-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --output=output_job/trans_phase1_chronos_mse_freq_1p5x_mvpos_gemma3_4b.out
#SBATCH --error=output_job/trans_phase1_chronos_mse_freq_1p5x_mvpos_gemma3_4b.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06],v100-f-[14-15]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

# ── Fresh-start cleanup ──────────────────────────────────────────────
# Remove the stage's checkpoint dir before launching training so the run
# always starts from a clean slate with the current code. Override with:
#   KEEP_CKPT=1 sbatch <this script>
CKPT_DIR="results/sincos_patch32/gemma_3_4b/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos"
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


TAG="${1:-run_trans_phase1_chronos_mse_freq_1p5x_mvpos_gemma3_4b}"
export TMPDIR=tmp
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export NCCL_DEBUG=warn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK_WEIGHT_CHRONOS_UV="${TASK_WEIGHT_CHRONOS_UV:-1.0}"
TASK_WEIGHT_SYNTHETIC="${TASK_WEIGHT_SYNTHETIC:-1.0}"

DEBUG_FLAG=""
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_FLAG="--debug"
fi

if [[ -n "${DEBUG_FLAG}" ]]; then
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse_debug
  echo "DEBUG MODE: cache=${CHRONOS_PREPROCESSED_SPLITS_DIR}"
else
  export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_phase1_mse
fi

SYNTHETIC_DATA_DIR=/vast/s225250685/project/synthetic_data/unified/data

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
echo "Stage: phase1_chronos_mse (Chronos-only MSE pretraining)"
echo "LLM: Gemma-3-4B (FROZEN, hidden_size=2560)"
echo "Encoder/decoder: TRANSFORMER + frequency embedding (use_freq_embed=ON)"
echo "Sampling: chronos_window_sampling=random, samples_per_row=25"
echo "Chronos: univariate registry + synthetic MV"
echo "Synthetic MV dir: ${SYNTHETIC_DATA_DIR}"
echo "Split cache: ${CHRONOS_PREPROCESSED_SPLITS_DIR}"
echo "Source weights chronos_uv : synthetic = ${TASK_WEIGHT_CHRONOS_UV} : ${TASK_WEIGHT_SYNTHETIC}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set in parent environment"
else
  echo "HF_TOKEN is NOT set in parent environment, will try token file fallback"
fi

# Pre-warm the OS page cache for the cache dir on each compute node.
PREWARM_START=$(date +%s)
echo "[$(date)] pre-warming NFS cache on each node …"
srun --export=ALL --kill-on-bad-exit=0 \
  --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
  bash -lc "
    cd /scratch/s225250685/project/FoundationalTS
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

    echo "[$(hostname)] HF_HOME=${HF_HOME}"
    echo "[$(hostname)] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    echo "[$(hostname)] HF_HUB_CACHE=${HF_HUB_CACHE}"
    df -h /tmp || true

    # Copy Gemma-3-4B from permanent HF hub cache to local scratch
    PERM_HF="${HOME}/.cache/huggingface/hub"
    for MODEL_SNAP in "models--google--gemma-3-4b"; do
      if [ -d "${PERM_HF}/${MODEL_SNAP}" ]; then
        cp -r "${PERM_HF}/${MODEL_SNAP}" "${HF_HUB_CACHE}/" || true
        echo "[$(hostname)] copied ${MODEL_SNAP} from permanent cache"
      fi
    done

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

    /scratch/s225250685/project/FoundationalTS/.env/bin/torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      --rdzv_id='"${SLURM_JOB_ID}"' \
      --node_rank="${SLURM_NODEID}" \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages phase1_chronos_mse \
        --llm_id google/gemma-3-4b \
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
        '"${DEBUG_FLAG}"' \
        --val_batch_size 4 \
        --batch_size 4 \
        --grad_accum_steps 2 \
        --use_processor \
        --train_log_every_n_steps 10 \
        --eval_every_n_iters 1000 \
        --eval_max_samples_per_task 512 \
        --wandb_run_name "trans_phase1_chronos_mse_freq_1p5x_mvpos_gemma3_4b" \
        --wandb_project "Time_Lang_Foundation"
'
