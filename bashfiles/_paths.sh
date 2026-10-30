# shellcheck shell=bash
# Shared path resolution for every launcher in bashfiles/.
#
# SOURCE this file, do not execute it. Use the preamble every launcher carries:
#     _d=""
#     for _c in "${LOTUS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
#               "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
#       _p="$_c"
#       while [ -n "$_p" ] && [ "$_p" != "/" ]; do
#         if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
#         _p="$(dirname "$_p")"
#       done
#       [ -n "$_d" ] && break
#     done
#     source "$_d/bashfiles/_paths.sh"
#
# Why the candidate list rather than just BASH_SOURCE: under sbatch, SLURM
# executes a COPY of the launcher from /var/spool/slurm/..., so BASH_SOURCE is
# NOT inside the repo and a walk-up from it reaches / and fails (the job dies
# within a second). SLURM_SUBMIT_DIR is the repo when you sbatch from the
# checkout, which is the normal workflow.
#
# Why this exists: the launchers used to hard-code one contributor's absolute
# paths, so they only ran on that account. Everything below resolves
# dynamically and every value is overridable from the environment, so the same
# script runs on any account.
#
# Overrides (export before sbatch, or set in your shell rc):
#   LOTUS_REPO_ROOT       repo checkout                (default: auto-detected)
#   LOTUS_PYTHON          python interpreter           (default: ./.env/bin/python, else conda env opentslm, else python3)
#   LOTUS_TORCHRUN        torchrun binary              (default: alongside LOTUS_PYTHON)
#   LOTUS_CONDA_ENV       conda env name or path       (default: opentslm)
#   LOTUS_SCRATCH         writable scratch base        (default: /scratch/$USER -- the authors' cluster layout; set it)
#   LOTUS_VAST            large/fast storage base      (default: /vast/$USER -- likewise)
#   LOTUS_RESULTS_ROOT    checkpoints / results        (default: $LOTUS_SCRATCH/Lotus/results)
#   RESULTS_BASE_DIR      where curriculum_learning.py writes a run (default: $LOTUS_RESULTS_ROOT;
#                         Phase-2 launchers append a per-experiment suffix)
#   LOTUS_PRETRAINED_DIR  released phase-1 checkpoints (default: $LOTUS_SCRATCH/Lotus/pretrained_weights)
#   LOTUS_DATA_DIR        downstream raw-data root; auto-downloaded corpora
#                       (HAR, Sleep-EDF, ECG-QA, M4, TSQA, ...) land here
#                                                    (default: $LOTUS_SCRATCH/Lotus/data)
#   LOTUS_SYNTHETIC_DIR   synthetic MV arrow dir(s)    (colon-separated; no default)
#   LOTUS_TSUMM_DIR       TSUMM arrow dir              (no default)
#   LOTUS_GIFT_DATA       GIFT-Eval data root          (no default)
#   LOTUS_ETT_DIR         ETT/Weather/Traffic/Electricity csv dir (default: $LOTUS_DATA_DIR/ett)
#   LOTUS_CHATTS_DIR      ChatTS corpus root (sft/, ift/ and process/align_256 under it)
#                                                    (no default)
#   TIMEMQA_OPEN_ENDED_CSV  Time-MQA open_ended_QA.csv (default: $LOTUS_DATA_DIR/timemqa/open_ended_QA.csv)
#   CHRONOS_UV_BASE_PATH    Chronos arrow datasets written by scripts/prefetch_chronos_uv.py
#                                                    (default: $LOTUS_VAST/Lotus/_chronos_uv_datasets)
#   LOTUS_HF_HUB          HF hub cache dir             (default: $HF_HOME/hub, else ~/.cache/huggingface/hub)
#   LOTUS_TSREASON_DIR    TS-Reasoning raw data        (no default)
#   LOTUS_OUTPUT_DIR      Slurm stdout/stderr logs     (default: $LOTUS_REPO_ROOT/output_job)
#   LOTUS_OFFICIAL_VARIANTS  dir with train_official_{sp,flamingo}.py for the OpenTSLM
#                         baselines (default: $LOTUS_REPO_ROOT/baselines/official_variants)
#   LOTUS_TIMEOMNI_PYTHON python with TimeOmni-1's deps (no default; only train_joint5_timeomni_4b.sh)
#   LOTUS_GIFT_RESULTS_BASE  GIFT-Eval results dir read by evaluation/forecasting/compare_gift_eval.py
#   LOTUS_GIFT_OURS_CSV   override for the merged all_results.csv path (same script)
#   LOTUS_CIK_VARIANTS    dir with the OpenTSLM-Flamingo eval helpers (default: evaluation/official)
#   LOTUS_GEMMA_LOCAL     local gemma-3-270m snapshot  (no default; falls back to the HF id)

# ── repo root ───────────────────────────────────────────────────────────────
if [ -z "${LOTUS_REPO_ROOT:-}" ]; then
  _lotus_self="${BASH_SOURCE[0]}"
  LOTUS_REPO_ROOT="$(cd "$(dirname "$_lotus_self")/.." && pwd)"
  # Under SLURM the script may run from a spool copy; prefer the submit dir
  # when it actually looks like this repo.
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/curriculum_learning.py" ]; then
    LOTUS_REPO_ROOT="${SLURM_SUBMIT_DIR}"
  fi
fi
export LOTUS_REPO_ROOT

# ── python / torchrun ───────────────────────────────────────────────────────
export LOTUS_CONDA_ENV="${LOTUS_CONDA_ENV:-opentslm}"
if [ -z "${LOTUS_PYTHON:-}" ]; then
  for _cand in \
      "${LOTUS_REPO_ROOT}/.env/bin/python" \
      "${HOME}/.conda/envs/${LOTUS_CONDA_ENV}/bin/python" \
      "$(command -v python3 2>/dev/null)"; do
    if [ -x "$_cand" ]; then LOTUS_PYTHON="$_cand"; break; fi
  done
fi
export LOTUS_PYTHON="${LOTUS_PYTHON:-python3}"
if [ -z "${LOTUS_TORCHRUN:-}" ]; then
  _lotus_bin="$(dirname "${LOTUS_PYTHON}")"
  if [ -x "${_lotus_bin}/torchrun" ]; then LOTUS_TORCHRUN="${_lotus_bin}/torchrun"
  else LOTUS_TORCHRUN="$(command -v torchrun 2>/dev/null || echo torchrun)"; fi
fi
export LOTUS_TORCHRUN

# ── writable locations ──────────────────────────────────────────────────────
export LOTUS_SCRATCH="${LOTUS_SCRATCH:-/scratch/${USER}}"
export LOTUS_VAST="${LOTUS_VAST:-/vast/${USER}}"
export LOTUS_RESULTS_ROOT="${LOTUS_RESULTS_ROOT:-${LOTUS_SCRATCH}/Lotus/results}"
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${LOTUS_RESULTS_ROOT}}"
export LOTUS_OUTPUT_DIR="${LOTUS_OUTPUT_DIR:-${LOTUS_REPO_ROOT}/output_job}"
export LOTUS_PRETRAINED_DIR="${LOTUS_PRETRAINED_DIR:-${LOTUS_SCRATCH}/Lotus/pretrained_weights}"
export LOTUS_DATA_DIR="${LOTUS_DATA_DIR:-${LOTUS_SCRATCH}/Lotus/data}"

# ── external datasets (no silent wrong default: empty unless you set them) ──
export LOTUS_SYNTHETIC_DIR="${LOTUS_SYNTHETIC_DIR:-}"
export LOTUS_TSUMM_DIR="${LOTUS_TSUMM_DIR:-}"
export LOTUS_GIFT_DATA="${LOTUS_GIFT_DATA:-}"
export LOTUS_TSREASON_DIR="${LOTUS_TSREASON_DIR:-}"
export LOTUS_GEMMA_LOCAL="${LOTUS_GEMMA_LOCAL:-}"
export LOTUS_ETT_DIR="${LOTUS_ETT_DIR:-${LOTUS_DATA_DIR}/ett}"
export LOTUS_CHATTS_DIR="${LOTUS_CHATTS_DIR:-}"
export TIMEMQA_OPEN_ENDED_CSV="${TIMEMQA_OPEN_ENDED_CSV:-}"
export CHRONOS_UV_BASE_PATH="${CHRONOS_UV_BASE_PATH:-${LOTUS_VAST}/Lotus/_chronos_uv_datasets}"

# Local HuggingFace hub cache (snapshot dirs live under $LOTUS_HF_HUB/models--*/snapshots/*)
if [ -z "${LOTUS_HF_HUB:-}" ]; then
  if [ -n "${HF_HOME:-}" ] && [ -d "${HF_HOME}/hub" ]; then LOTUS_HF_HUB="${HF_HOME}/hub"
  else LOTUS_HF_HUB="${HOME}/.cache/huggingface/hub"; fi
fi
export LOTUS_HF_HUB

# ── helper: fail fast with an actionable message instead of a silent skip ───
lotus_require_dir() {
  # usage: lotus_require_dir LOTUS_SYNTHETIC_DIR "synthetic MV arrow corpus"
  local _var="$1" _what="${2:-path}" _val="${!1:-}"
  if [ -z "${_val}" ]; then
    echo "[_paths.sh] ERROR: ${_var} is not set (${_what})." >&2
    echo "[_paths.sh]        export ${_var}=/path/to/data before submitting." >&2
    return 1
  fi
  local _first="${_val%%:*}"
  if [ ! -d "${_first}" ]; then
    echo "[_paths.sh] ERROR: ${_var}='${_first}' does not exist or is unreadable (${_what})." >&2
    return 1
  fi
  return 0
}

mkdir -p "${LOTUS_OUTPUT_DIR}" 2>/dev/null || true
