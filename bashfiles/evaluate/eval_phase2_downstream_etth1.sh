#!/bin/bash
#SBATCH --job-name=eval_p2ds_etth1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
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

module load Anaconda3
source activate
conda activate opentslm

cd ${FTS_REPO_ROOT}

export TOKENIZERS_PARALLELISM=false

CKPT="${1:-${FTS_RESULTS_ROOT}/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt}"
LLM=google/gemma-3-270m
ETTh1=${FTS_ETT_DIR}/ETTh1.csv

python evaluation/forecasting/eval_phase2_downstream.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --dataset            etth1 \
  --csv                "${ETTh1}" \
  --patch_size         32 \
  --pred_len           96 192 336 720 \
  --batch_size         8 \
  --eval_split         test \
  --lora_r             16 \
  --lora_alpha         32 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --enc_dec_type       transformer \
  --pe_type            sincos \
  --out_dir            ./eval_results/joint_phase2_downstream/etth1


