#!/bin/bash
#SBATCH --job-name=etth1-forecast-eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
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
module load Anaconda3
source activate
# conda activate opentslm

cd ${FTS_REPO_ROOT}
conda activate ./.env

RESULTS=results/eval
LLM=google/gemma-3-270m
INFER=evaluation/forecasting/fixed_inference.py
ETTh1=${FTS_ETT_DIR}/ETT-small/ETTh1.csv

# ── joint_phase1_extend_balanced checkpoint ─────────────────────────
# CKPT=checkpoint/best_model.pt
# CKPT=${FTS_REPO_ROOT}/results/sincos_patch96/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced/checkpoints/best_model.pt
CKPT=${FTS_REPO_ROOT}/results/sincos_patch96/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse/checkpoints/best_model.pt
STAGE=joint_phase1_extend_balanced

python ${INFER} \
  --ckpt         ${CKPT} \
  --llm_id       ${LLM} \
  --stage_name   ${STAGE} \
  --eval_format  phase1_pretrain \
  --etth1_csv    ${ETTh1} \
  --mode         teacher_forced \
  --device       cuda \
  --batch_size   32 \
  --patch_size   96 \
  --history_len  576 \
  --pred_len     192 \
  --plot_every   200 \
  --out_dir      ./eval_results/${STAGE}/phase1_pretrain/autoregressive_pure_mse \
  --forecast_backbone llm \
  --eval_split train

