# shellcheck shell=bash
# Shared path resolution for every launcher in bashfiles/.
#
# SOURCE this file, do not execute it. Use the preamble every launcher carries:
#     _d=""
#     for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
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
# NOT inside the repo and a walk-up from it reaches / and fails. Job 3960424
# died that way in 1 second. SLURM_SUBMIT_DIR is the repo when you sbatch from
# the checkout, which is the normal workflow.
#
# Why this exists: the launchers used to hard-code one contributor's absolute
# paths (/scratch/s225250685/..., /home/s223540177/...), so they only ran on
# that account. Everything below resolves dynamically and every value is
# overridable from the environment, so the same script runs on any account.
#
# Overrides (export before sbatch, or set in your shell rc):
#   FTS_REPO_ROOT       repo checkout                (default: auto-detected)
#   FTS_PYTHON          python interpreter           (default: conda env opentslm, else python3)
#   FTS_TORCHRUN        torchrun binary              (default: alongside FTS_PYTHON)
#   FTS_CONDA_ENV       conda env name or path       (default: opentslm)
#   FTS_SCRATCH         writable scratch base        (default: /scratch/$USER)
#   FTS_VAST            large/fast storage base      (default: /vast/$USER)
#   FTS_RESULTS_ROOT    checkpoints / results        (default: $FTS_SCRATCH/FoundationalTS/results)
#   FTS_SYNTHETIC_DIR   synthetic MV arrow dir(s)    (colon-separated; no default)
#   FTS_TSUMM_DIR       TSUMM arrow dir              (no default)
#   FTS_GIFT_DATA       GIFT-Eval data root          (no default)
#   FTS_ETT_DIR         ETT csv directory            (default: $FTS_SCRATCH/TimeSeriesECM/dataset)
#   FTS_HF_HUB          HF hub cache dir             (default: $HF_HOME/hub, else ~/.cache/huggingface/hub)
#   FTS_TSREASON_DIR    TS-Reasoning raw data        (no default)
#   FTS_GEMMA_LOCAL     local gemma-3-270m snapshot  (no default; falls back to the HF id)

# ── repo root ───────────────────────────────────────────────────────────────
if [ -z "${FTS_REPO_ROOT:-}" ]; then
  _fts_self="${BASH_SOURCE[0]}"
  FTS_REPO_ROOT="$(cd "$(dirname "$_fts_self")/.." && pwd)"
  # Under SLURM the script may run from a spool copy; prefer the submit dir
  # when it actually looks like this repo.
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/curriculum_learning.py" ]; then
    FTS_REPO_ROOT="${SLURM_SUBMIT_DIR}"
  fi
fi
export FTS_REPO_ROOT

# ── python / torchrun ───────────────────────────────────────────────────────
export FTS_CONDA_ENV="${FTS_CONDA_ENV:-opentslm}"
if [ -z "${FTS_PYTHON:-}" ]; then
  for _cand in \
      "${FTS_REPO_ROOT}/.env/bin/python" \
      "${HOME}/.conda/envs/${FTS_CONDA_ENV}/bin/python" \
      "$(command -v python3 2>/dev/null)"; do
    if [ -x "$_cand" ]; then FTS_PYTHON="$_cand"; break; fi
  done
fi
export FTS_PYTHON="${FTS_PYTHON:-python3}"
if [ -z "${FTS_TORCHRUN:-}" ]; then
  _fts_bin="$(dirname "${FTS_PYTHON}")"
  if [ -x "${_fts_bin}/torchrun" ]; then FTS_TORCHRUN="${_fts_bin}/torchrun"
  else FTS_TORCHRUN="$(command -v torchrun 2>/dev/null || echo torchrun)"; fi
fi
export FTS_TORCHRUN

# ── writable locations ──────────────────────────────────────────────────────
export FTS_SCRATCH="${FTS_SCRATCH:-/scratch/${USER}}"
export FTS_VAST="${FTS_VAST:-/vast/${USER}}"
export FTS_RESULTS_ROOT="${FTS_RESULTS_ROOT:-${FTS_SCRATCH}/FoundationalTS/results}"
export FTS_OUTPUT_DIR="${FTS_OUTPUT_DIR:-${FTS_REPO_ROOT}/output_job}"
export FTS_PRETRAINED_DIR="${FTS_PRETRAINED_DIR:-${FTS_SCRATCH}/FoundationalTS/pretrained_weights}"

# ── external datasets (no silent wrong default: empty unless you set them) ──
export FTS_SYNTHETIC_DIR="${FTS_SYNTHETIC_DIR:-}"
export FTS_TSUMM_DIR="${FTS_TSUMM_DIR:-}"
export FTS_GIFT_DATA="${FTS_GIFT_DATA:-}"
export FTS_TSREASON_DIR="${FTS_TSREASON_DIR:-}"
export FTS_GEMMA_LOCAL="${FTS_GEMMA_LOCAL:-}"
export FTS_ETT_DIR="${FTS_ETT_DIR:-${FTS_SCRATCH}/TimeSeriesECM/dataset}"

# Local HuggingFace hub cache (snapshot dirs live under $FTS_HF_HUB/models--*/snapshots/*)
if [ -z "${FTS_HF_HUB:-}" ]; then
  if [ -n "${HF_HOME:-}" ] && [ -d "${HF_HOME}/hub" ]; then FTS_HF_HUB="${HF_HOME}/hub"
  else FTS_HF_HUB="${HOME}/.cache/huggingface/hub"; fi
fi
export FTS_HF_HUB

# ── helper: fail fast with an actionable message instead of a silent skip ───
fts_require_dir() {
  # usage: fts_require_dir FTS_SYNTHETIC_DIR "synthetic MV arrow corpus"
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

mkdir -p "${FTS_OUTPUT_DIR}" 2>/dev/null || true
