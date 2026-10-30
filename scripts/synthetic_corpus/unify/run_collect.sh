#!/bin/bash
# Collect synthetic MTS arrow datasets into a unified schema and emit statistics.
# Requires the `uni2st` conda env (gluonts + pyarrow + numpy + tqdm).
# For interactive runs: `conda activate uni2st && bash scripts/run_collect.sh`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory (collect_synthetic.py lives here)
OUT_DIR="${OUT_DIR:-${LOTUS_SYNTHETIC_DIR:?set LOTUS_SYNTHETIC_DIR to the directory that will hold unified_*.arrow}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Constraint envelope. The script defaults (3<=C<=10, 1024<=L<=4096) would
# drop the univariate KernelSynth files and the 512-length / 8192-length
# sweeps; the released corpus was collected with the wide-open envelope
# recorded in reference_stats.json ("envelope"), reproduced here as defaults.
MIN_VARIATES="${MIN_VARIATES:-1}"
MAX_VARIATES="${MAX_VARIATES:-10000}"
MIN_LENGTH="${MIN_LENGTH:-1}"
MAX_LENGTH="${MAX_LENGTH:-100000000}"
# Optional: SOURCES_JSON=/path/to/sources.json   override DEFAULT_SOURCES in the .py
# Optional: ONLY="name1 name2"                   process only a subset

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

ARGS=(--out-dir "${OUT_DIR}" --batch-size "${BATCH_SIZE}"
      --min-variates "${MIN_VARIATES}" --max-variates "${MAX_VARIATES}"
      --min-length "${MIN_LENGTH}" --max-length "${MAX_LENGTH}")
if [[ -n "${SOURCES_JSON:-}" ]]; then
    ARGS+=(--sources-json "${SOURCES_JSON}")
fi
if [[ -n "${ONLY:-}" ]]; then
    # shellcheck disable=SC2206
    ONLY_ARR=(${ONLY})
    ARGS+=(--only "${ONLY_ARR[@]}")
fi

python -u scripts/collect_synthetic.py "${ARGS[@]}"

echo "[run_collect] wrote unified arrows + stats to ${OUT_DIR}"
