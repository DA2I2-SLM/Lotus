#!/bin/bash
#SBATCH --job-name=eval_time_benchmark
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

# Zero-shot TIME benchmark evaluation (all 49 datasets, short-horizon term).
#
# Pass the checkpoint as the first argument or edit CKPT below:
#   sbatch eval_time_benchmark.sh /path/to/best_model.pt
#
# To run all horizon terms (short + medium + long), add --term all.
#
# timebench is installed directly in the opentslm conda env (editable install
# from ${FTS_SCRATCH}/TIME with relaxed version pins).
#
# Prerequisites (already done):
#   ${FTS_SCRATCH}/TIME  — cloned and installed via pip install -e .
#   echo "TIME_DATASET=/path/to/time_datasets" >> ${FTS_REPO_ROOT}/.env

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm

cd ${FTS_REPO_ROOT}

export TOKENIZERS_PARALLELISM=false
export TIME_DATASET=${FTS_SCRATCH}/TIME_datasets


CKPT="${1:-/scratch/$(whoami)/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt}"
LLM=google/gemma-3-270m
OUT_DIR=./eval_results/time_benchmark

echo "Checkpoint : ${CKPT}"
echo "Output dir : ${OUT_DIR}"

python evaluation/time_benchmark/eval_time.py \
  --ckpt               "${CKPT}" \
  --llm_id             "${LLM}" \
  --patch_size         32 \
  --lora_r             4 \
  --lora_alpha         8 \
  --mlp_enc_hidden_dim 64 \
  --mlp_enc_ff_dim     256 \
  --mlp_dec_hidden_dim 64 \
  --mlp_dec_ff_dim     256 \
  --enc_dec_type       mlp \
  --pe_type            sincos \
  --term               short \
  --out_dir            "${OUT_DIR}"
