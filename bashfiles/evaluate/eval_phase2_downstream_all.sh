#!/bin/bash
#SBATCH --job-name=eval_p2ds_all
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
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

# Evaluates joint_phase2_downstream checkpoint on all 7 standard long-horizon
# benchmarks across pred_lens {96, 192, 336, 720}.
#
# Each dataset uses multivariate samples (all columns, up to --max_cols)
# matching the training distribution exactly.  Evaluation uses teacher-forced
# decoding (same forward pass as training loss), which is faster and more
# reproducible than autoregressive mode.
#
# Pass the checkpoint path as the first argument or edit CKPT below:
#   sbatch eval_phase2_downstream_all.sh /path/to/best_model.pt
#

set -euo pipefail
mkdir -p output_job


cd "${LOTUS_REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false

CKPT="${1:-${LOTUS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt}"
LLM=google/gemma-3-270m
DATA_ROOT=${LOTUS_ETT_DIR}
OUT_DIR=./eval_results/joint_phase2_downstream/all

echo "Checkpoint : ${CKPT}"
echo "Data root  : ${DATA_ROOT}"
echo "Output dir : ${OUT_DIR}"

"${LOTUS_PYTHON}" evaluation/forecasting/eval_phase2_downstream.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --all_datasets \
  --data_root          "${DATA_ROOT}" \
  --patch_size         32 \
  --pred_len           96 192 336 720 \
  --max_cols           16 \
  --batch_size         4 \
  --eval_split         test \
  --lora_r             32 \
  --lora_alpha         64 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       transformer \
  --pe_type            sincos \
  --out_dir            "${OUT_DIR}"

"${LOTUS_PYTHON}" ${LOTUS_REPO_ROOT}/evaluation/forecasting/eval_chatts.py --ckpt ${LOTUS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt --llm_id google/gemma-3-270m --patch_size 32  --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 --mlp_dec_hidden_dim 64  --mlp_dec_ff_dim 256 --lora_r 32 --lora_alpha 64 --batch_size 8 --out_dir ./eval_results/chatts


export TOKENIZERS_PARALLELISM=false

# CKPT="${1:-${LOTUS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt}"
CKPT="${1:-${LOTUS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend/checkpoints/best_model.pt}"

LLM=google/gemma-3-270m
DATA_ROOT=${LOTUS_ETT_DIR}
OUT_DIR=./eval_results/joint_phase2_downstream/all

echo "Checkpoint : ${CKPT}"
echo "Data root  : ${DATA_ROOT}"
echo "Output dir : ${OUT_DIR}"

"${LOTUS_PYTHON}" evaluation/forecasting/eval_phase2_downstream.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --all_datasets \
  --data_root          "${DATA_ROOT}" \
  --patch_size         32 \
  --pred_len           96 192 336 720 \
  --max_cols           16 \
  --batch_size         4 \
  --eval_split         test \
  --lora_r             32 \
  --lora_alpha         64 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       transformer \
  --pe_type            sincos \
  --out_dir            "${OUT_DIR}"


