#!/bin/bash
# Generate the LMC-Synth multivariate synthetic dataset for TimePFN pretraining
# under the new constraint envelope:
#   - n_variates (channels) sampled per-item uniformly in [3, 10]
#   - length     sampled per-item uniformly in [1024, 4096]
#
# Dirichlet/Weibull bounds control how diverse the channels are (see Figure 3-5
# in the TimePFN paper). The defaults below cover a broad range from strongly
# correlated to nearly independent variates; override via env vars if needed.
#
# Requires the `uni2st` conda env (has gluonts, joblib, sklearn, pyarrow, tqdm).
# For interactive runs: `conda activate uni2st && bash scripts/gen_coreg_15k.sh`.
# Note: LMC_Synth.py's KERNEL_BANK is built at import time with LENGTH=1024,
# so kernel periodicities scale proportionally with the sampled per-item length.

set -euo pipefail

REPO_ROOT="${SYNTH_TIMEPFN_REPO:?set SYNTH_TIMEPFN_REPO to the egetaga/TimePFN checkout}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/datasets/synthetic}"
OUT_FILE="${OUT_FILE:-lmc_synth_C3-10_L1024-4096.arrow}"

NUM_SERIES="${NUM_SERIES:-15000}"
MAX_KERNELS="${MAX_KERNELS:-5}"

MIN_CHANNELS="${MIN_CHANNELS:-3}"
MAX_CHANNELS="${MAX_CHANNELS:-10}"
MIN_LENGTH="${MIN_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SEED="${SEED:-0}"

DIRICHLET_MIN="${DIRICHLET_MIN:-0.1}"
DIRICHLET_MAX="${DIRICHLET_MAX:-2.0}"
WEIBULL_SHAPE="${WEIBULL_SHAPE:-2.0}"
WEIBULL_SCALE="${WEIBULL_SCALE:-10.0}"

mkdir -p "${OUT_DIR}"

cd "${REPO_ROOT}"

python -u synthetic_data_generation/LMC_Synth.py \
    -N "${NUM_SERIES}" \
    -J "${MAX_KERNELS}" \
    --min-channels "${MIN_CHANNELS}" \
    --max-channels "${MAX_CHANNELS}" \
    --min-length   "${MIN_LENGTH}" \
    --max-length   "${MAX_LENGTH}" \
    --seed "${SEED}" \
    -M "${DIRICHLET_MIN}" \
    -X "${DIRICHLET_MAX}" \
    -W "${WEIBULL_SHAPE}" \
    -Z "${WEIBULL_SCALE}" \
    -D "${OUT_DIR}/" \
    -O "${OUT_FILE}"

echo "[gen_coreg_15k] wrote ${OUT_DIR}/${OUT_FILE}"
