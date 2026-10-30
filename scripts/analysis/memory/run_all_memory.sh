#!/bin/bash
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#SBATCH --job-name=mem_scaling
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

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
#
# ─────────────────────────────────────────────────────────────────────────────
# Memory-scaling sweep (paper Fig. 7 / Appendix "Computational Cost").
#
# For every backbone x model x (L, N) cell of the SimulationQADataset grid,
# runs scripts/analysis/memory/get_memory_use.py (a few training steps at
# batch 1) and appends one CSV row with the peak CUDA memory. A cell that OOMs
# is recorded with status=error and the sweep continues. Plot afterwards with
#   "${LOTUS_PYTHON}" scripts/analysis/memory/plot_memory_scaling.py --csv "$RESULTS_CSV"
#
# Environment overrides (all optional):
#   MEM_LLM_IDS      space-separated backbones    (default: google/gemma-3-270m google/gemma-3-1b-pt Qwen/Qwen3-0.6B)
#   MEM_MODELS       space-separated model types  (default: OpenTSLMSPFixPos2D OpenTSLMSP)
#                    OpenTSLMFlamingo is accepted too but needs the open_flamingo package.
#   MEM_LENGTHS      series lengths L             (default: 10 100 1000 10000)
#   MEM_NUM_SERIES   number of series N           (default: 1 2 3 4 5)
#   MEM_DATASETS     real datasets to include     (default: none; e.g. "HARCoTQADataset SleepEDFCoTQADataset")
#   MEM_STEPS        training steps per cell      (default: 20)
#   MEM_EXTRA        extra flags for get_memory_use.py (e.g. "--enc_dec_type transformer --lora_r 8 --lora_alpha 16"
#                    for the 1B / Qwen3 LOTUS encoder, see configs/*.env)
#   RESULTS_CSV      output CSV                   (default: $LOTUS_RESULTS_ROOT/memory/memory_use.csv)
#   DEVICE           torch device                 (default: cuda)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p output_job
cd "${LOTUS_REPO_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -a LLM_IDS <<< "${MEM_LLM_IDS:-google/gemma-3-270m google/gemma-3-1b-pt Qwen/Qwen3-0.6B}"
read -r -a MODELS <<< "${MEM_MODELS:-OpenTSLMSPFixPos2D OpenTSLMSP}"
read -r -a SIMULATION_LENGTHS <<< "${MEM_LENGTHS:-10 100 1000 10000}"
read -r -a SIMULATION_NUM_SERIES <<< "${MEM_NUM_SERIES:-1 2 3 4 5}"
read -r -a DATASETS <<< "${MEM_DATASETS:-}"
STEPS="${MEM_STEPS:-20}"
DEVICE="${DEVICE:-cuda}"
RESULTS_CSV="${RESULTS_CSV:-${LOTUS_RESULTS_ROOT}/memory/memory_use.csv}"
mkdir -p "$(dirname "${RESULTS_CSV}")"

SCRIPT="${LOTUS_REPO_ROOT}/scripts/analysis/memory/get_memory_use.py"
echo "Writing results to: ${RESULTS_CSV}"
echo "Backbones: ${LLM_IDS[*]}  Models: ${MODELS[*]}  L: ${SIMULATION_LENGTHS[*]}  N: ${SIMULATION_NUM_SERIES[*]}"

run_cell() {
  # usage: run_cell <llm> <model> <dataset> [extra get_memory_use.py flags...]
  local llm="$1" model="$2" dataset="$3"; shift 3
  echo "[RUN] llm_id=${llm} model=${model} dataset=${dataset} $*"
  set +e
  # shellcheck disable=SC2086
  "${LOTUS_PYTHON}" "${SCRIPT}" --llm_id "${llm}" --model "${model}" --dataset "${dataset}" \
      --device "${DEVICE}" --steps "${STEPS}" --results_csv "${RESULTS_CSV}" "$@" ${MEM_EXTRA:-}
  local status=$?
  set -e
  if [ ${status} -ne 0 ]; then
    echo "[ERROR] llm_id=${llm} model=${model} dataset=${dataset} $* (exit ${status})"
  fi
}

for llm in "${LLM_IDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      [ -n "${dataset}" ] || continue
      run_cell "${llm}" "${model}" "${dataset}"
    done
    for length in "${SIMULATION_LENGTHS[@]}"; do
      for num_series in "${SIMULATION_NUM_SERIES[@]}"; do
        run_cell "${llm}" "${model}" SimulationQADataset --length "${length}" --num_series "${num_series}"
      done
    done
  done
done

echo "All runs completed. -> ${RESULTS_CSV}"
