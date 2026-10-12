#!/bin/bash
# GPU smoke test for phase-2 training. NOT a research run — it answers exactly
# one question: does a phase-2 stage complete real optimizer steps with a
# FINITE loss on a GPU?
#
# Why it exists: joint_phase2 has no test coverage (`grep -rln joint_phase2
# test/ test_phases/` is empty), so "it auto-merged and compiles" was the only
# signal available when the origin/hiepnh merge silently broke every phase-2
# run by nulling the MV position_ids cache. This catches that class of break.
#
#   sbatch bashfiles/smoke/smoke_phase2_gpu.sh
#   PASS -> exit 0 and "SMOKE PASS" in the log
#   FAIL -> non-zero exit, and the reason is the last line of the log
#
#SBATCH --job-name=smoke_phase2_gpu
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

set -uo pipefail

# Under sbatch SLURM runs a COPY of this script from its spool dir, so
# ${BASH_SOURCE[0]} is not in the repo; SLURM_SUBMIT_DIR is.
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
cd "${FTS_REPO_ROOT}"

module load Anaconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FTS_CONDA_ENV}" || true
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TQDM_DISABLE=1
export HF_HUB_OFFLINE=1          # compute nodes have no internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Backbone. google/gemma-3-270m (what the real launchers use) is GATED and its
# local cache blobs are broken symlinks, so offline it cannot load. gemma-3-1b-pt
# has intact blobs and exercises the identical Gemma3 RoPE path, which is what
# this test is about. Override with SMOKE_LLM=<hf id or snapshot dir>.
if [ -z "${SMOKE_LLM:-}" ]; then
  SMOKE_LLM="$(ls -d "${FTS_HF_HUB}"/models--google--gemma-3-1b-pt/snapshots/*/ 2>/dev/null | head -1)"
  [ -z "${SMOKE_LLM}" ] && SMOKE_LLM="$(ls -d "$HOME"/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/*/ 2>/dev/null | head -1)"
fi
if [ -z "${SMOKE_LLM}" ] || [ ! -f "${SMOKE_LLM}/config.json" ]; then
  echo "SMOKE FAIL: no usable local backbone; set SMOKE_LLM=<snapshot dir>" >&2
  exit 2
fi

# Bounded phase-2 config: HAR CoT only, tiny budget, no warm start, results in
# scratch so a smoke run can never touch real experiment output.
export RESULTS_BASE_DIR="${FTS_SCRATCH}/fts_smoke_phase2_gpu"
export PHASE2_COT_ONLY=1
export PHASE2_COT_TASKS=har
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET="${SMOKE_BALANCE_TARGET:-16}"
export PHASE2_SELECT_BY_GEN=0
rm -rf "${RESULTS_BASE_DIR}"

LOG="${FTS_OUTPUT_DIR}/smoke_phase2_${SLURM_JOB_ID:-local}.trainlog"
echo "repo=${FTS_REPO_ROOT}"
echo "python=${FTS_PYTHON}"
echo "backbone=${SMOKE_LLM}"
nvidia-smi -L || true

set +e
"${FTS_PYTHON}" curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase2_downstream \
  --llm_id "${SMOKE_LLM}" \
  --num_epochs 1 \
  --pe_type sincos --pe_base_c 10 --patch_size 32 \
  --enc_dec_type mlp --use_freq_embed \
  --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
  --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
  --batch_size 1 --val_batch_size 1 \
  --max_tokens_per_batch 600 \
  --grad_accum_steps 1 \
  --mse_alpha 1.0 \
  --windows_per_col_train 0 \
  --dataloader_workers 0 \
  --train_log_every_n_steps 1 \
  --eval_max_samples_per_task 2 \
  2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e

echo "---------------- smoke verdict ----------------"
if [ "${RC}" -ne 0 ]; then
  echo "SMOKE FAIL: trainer exited ${RC}" >&2
  exit "${RC}"
fi
# The break this test exists to catch surfaces as a TypeError on step 1.
if grep -q "NoneType' object is not subscriptable" "${LOG}"; then
  echo "SMOKE FAIL: position_ids cache was None — the MV position_ids regression is back" >&2
  exit 1
fi
# A run where every batch is skipped completes but learns nothing.
if grep -q "non-finite loss" "${LOG}"; then
  echo "SMOKE FAIL: non-finite loss (batches were skipped; nothing trained)" >&2
  exit 1
fi
if ! grep -qE "Curriculum Learning Complete|joint_phase2_downstream marked as completed" "${LOG}"; then
  echo "SMOKE FAIL: stage did not complete" >&2
  exit 1
fi
# Assert on the TRAINING loss, not test_loss. test_loss is ALWAYS nan by
# construction: _evaluate_stage initialises test_loss=0.0, never accumulates
# it, and assigns `avg_test_loss = float("nan")` unconditionally
# (curriculum_learning.py:1864, identical on main and origin/main since before
# this work). _is_stage_completed only checks that the KEY exists, not its
# value. Asserting test_loss is finite would make this test fail forever.
_tl="$(grep -oE 'train loss: [0-9.]+' "${LOG}" | tail -1 | awk '{print $NF}')"
if [ -z "${_tl}" ]; then
  echo "SMOKE FAIL: no epoch train loss found in the log" >&2
  exit 1
fi
if ! awk -v v="${_tl}" 'BEGIN{exit !(v+0 > 0 && v+0 < 1e6)}'; then
  echo "SMOKE FAIL: implausible final train loss ${_tl}" >&2
  exit 1
fi
echo "final epoch train loss: ${_tl}"
echo "SMOKE PASS: phase-2 completed real steps with a finite loss"
