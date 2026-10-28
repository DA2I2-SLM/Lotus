#!/bin/bash
# Synthetic multivariate / covariate-informed data generation (Chronos-2 §4.2
# re-implementation).
#
# Override via env vars (all optional):
#   NUM_SERIES=100000 MAX_KERNELS=5 LENGTH=1024 \
#   MIN_VARIATES=2 MAX_VARIATES=8 \
#   MIN_SOURCES=2 MAX_SOURCES=8 \
#   MAX_LAG=32 \
#   COVARIATE_PROB=0.3 MAX_COVARIATE_FRAC=0.5 \
#   SEED=0 OUT=/path/to/multivariate-kernelsynth.arrow \
#       bash scripts/run_multivariatize.sh
set -euo pipefail

REPO=${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}
NUM_SERIES=${NUM_SERIES:-10000}
MAX_KERNELS=${MAX_KERNELS:-5}
# Per-item length sampling: leave LENGTH unset to use MIN_LENGTH/MAX_LENGTH.
MIN_LENGTH=${MIN_LENGTH:-1024}
MAX_LENGTH=${MAX_LENGTH:-4096}
MIN_VARIATES=${MIN_VARIATES:-3}
MAX_VARIATES=${MAX_VARIATES:-10}
MIN_SOURCES=${MIN_SOURCES:-3}
MAX_SOURCES=${MAX_SOURCES:-10}
MAX_LAG=${MAX_LAG:-32}
# Covariates split off variates from the target -- a target of 3 with 1 covariate
# would land at 2 variates, violating the constraint. Default to 0 to keep target
# strictly in [MIN_VARIATES, MAX_VARIATES].
COVARIATE_PROB=${COVARIATE_PROB:-0.0}
MAX_COVARIATE_FRAC=${MAX_COVARIATE_FRAC:-0.5}
SEED=${SEED:-0}
OUT=${OUT:-${REPO}/data/multivariate-kernelsynth.arrow}

mkdir -p "$(dirname "${OUT}")"

echo "[run_multivariatize] num_series=${NUM_SERIES} max_kernels=${MAX_KERNELS} length in [${MIN_LENGTH}, ${MAX_LENGTH}]"
echo "[run_multivariatize] variates in [${MIN_VARIATES}, ${MAX_VARIATES}], sources in [${MIN_SOURCES}, ${MAX_SOURCES}]"
echo "[run_multivariatize] max_lag=${MAX_LAG} covariate_prob=${COVARIATE_PROB} max_covariate_frac=${MAX_COVARIATE_FRAC}"
echo "[run_multivariatize] seed=${SEED} -> ${OUT}"

python "${REPO}/scripts/multivariatize.py" \
    --num-series "${NUM_SERIES}" \
    --max-kernels "${MAX_KERNELS}" \
    --min-length "${MIN_LENGTH}" \
    --max-length "${MAX_LENGTH}" \
    --min-variates "${MIN_VARIATES}" \
    --max-variates "${MAX_VARIATES}" \
    --min-sources "${MIN_SOURCES}" \
    --max-sources "${MAX_SOURCES}" \
    --max-lag "${MAX_LAG}" \
    --covariate-prob "${COVARIATE_PROB}" \
    --max-covariate-frac "${MAX_COVARIATE_FRAC}" \
    --seed "${SEED}" \
    --out "${OUT}"

echo "[run_multivariatize] done -> ${OUT}"
