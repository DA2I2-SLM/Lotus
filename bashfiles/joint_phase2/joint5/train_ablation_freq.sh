#!/bin/bash
#SBATCH --job-name=abl_freq
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gpus=4
#SBATCH --constraint="gpu-l40s|gpu-v100|gpu-rtxp"
#SBATCH --exclude=v100l-f-[01-06]
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
# Frequency-branch ablation (paper §5.3 "What does the encoder contribute?").
#
# TWO ARMS, both REDUCED-BUDGET so the comparison is budget-fair — the control
# is retrained under the identical reduced budget rather than reusing the
# full-budget published model, so the delta isolates the branch, not the budget:
#   control : encoder WITH the rfft frequency branch (--use_freq_embed)
#   nofreq  : encoder WITHOUT it; the warm start's freq_embed/concat_proj
#             tensors are dropped (ABLATION_DROP_FREQ_KEYS=1), all other
#             weights load strict from the SAME phase-1 checkpoint.
#
# Reduced budget: 3 tasks (har + sleep + ecg — sleep/ecg are where frequency
# content should matter), balance 20k/epoch, 4 epochs, no early stop expected.
# gemma-3-270m only. Everything else matches train_joint5_ours.sh.
#
# NOTE the caveat for the paper: the phase-1 encoder was PRETRAINED with the
# freq branch, so the nofreq arm ablates it only at phase 2. A from-scratch
# nofreq phase 1 would cost ~5 days; this is the defensible reduced version.
#
# Usage: sbatch --job-name=abl_freq_<control|nofreq> train_ablation_freq.sh <control|nofreq>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${FTS_REPO_ROOT}"
ARM="${1:?usage: train_ablation_freq.sh <control|nofreq>}"
case "$ARM" in
  control) FREQ_FLAG="--use_freq_embed"; export ABLATION_DROP_FREQ_KEYS=0 ;;
  nofreq)  FREQ_FLAG="";                 export ABLATION_DROP_FREQ_KEYS=1 ;;
  *) echo "arm must be control|nofreq"; exit 1;;
esac

export RESULTS_BASE_DIR="${FTS_RESULTS_ROOT}_ablation_freq_${ARM}"
export WARM_START_PRETRAINED_CKPT=${FTS_PRETRAINED_DIR}/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt
[ -f "$WARM_START_PRETRAINED_CKPT" ] || { echo "FATAL: warm-start ckpt missing"; exit 1; }

export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_COT_TASKS=har,sleep,ecg
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=20000
export PHASE2_SELECT_BY_GEN=1
export PHASE2_GEN_EM_N=192
export PHASE2_EARLY_STOP_PAT=8

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME=/tmp/$USER/hf_abl_$SLURM_JOB_ID
mkdir -p "$HF_HOME"
[ -f "$HOME/.cache/huggingface/token" ] && cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"

echo "=== ABLATION arm=$ARM freq_flag='${FREQ_FLAG}' drop_keys=$ABLATION_DROP_FREQ_KEYS ==="
"${FTS_TORCHRUN}" --standalone --nproc_per_node=4 curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase2_downstream \
  --llm_id google/gemma-3-270m \
  --num_epochs 4 \
  --pe_type sincos --pe_base_c 10 --patch_size 32 \
  --enc_dec_type mlp $FREQ_FLAG \
  --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
  --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
  --batch_size 1 --val_batch_size 1 \
  --max_tokens_per_batch 600 --grad_accum_steps 32 \
  --mse_alpha 1.0 --windows_per_col_train 0 --dataloader_workers 0 \
  --save_every_n_steps 800 --gradient_checkpointing \
  --warm_start_from pretrained
echo "DONE_ABLATION_$ARM"
