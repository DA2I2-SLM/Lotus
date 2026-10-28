#!/bin/bash
# Univariate KernelSynth generation (Chronos v1 paper, Section 4.1).
#
# Wraps scripts/kernel-synth.py. Builds N random GP-prior series of length 1024
# by sampling 1..MAX_KERNELS kernels from KERNEL_BANK and combining them with
# random +/*. Output is a GluonTS-compatible arrow file with 1D `target`.
#
# Override via env vars, e.g.:
#   NUM_SERIES=1000000 MAX_KERNELS=5 OUT=/path/to/kernelsynth-data.arrow \
#       bash scripts/run_kernel_synth.sh
set -euo pipefail

REPO=${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}
NUM_SERIES=${NUM_SERIES:-100000}
MAX_KERNELS=${MAX_KERNELS:-10}
OUT=${OUT:-${REPO}/data/kernelsynth-data.arrow}

# kernel-synth.py hard-codes the output to `<script_dir>/kernelsynth-data.arrow`,
# so we run from the target directory and then move the file if OUT differs.
mkdir -p "$(dirname "${OUT}")"
WORKDIR=$(mktemp -d)
cp "${REPO}/scripts/kernel-synth.py" "${WORKDIR}/kernel-synth.py"

echo "[run_kernel_synth] num_series=${NUM_SERIES} max_kernels=${MAX_KERNELS}"
echo "[run_kernel_synth] writing to ${OUT}"

cd "${WORKDIR}"
python kernel-synth.py --num-series "${NUM_SERIES}" --max-kernels "${MAX_KERNELS}"
mv "${WORKDIR}/kernelsynth-data.arrow" "${OUT}"
rm -rf "${WORKDIR}"

echo "[run_kernel_synth] done -> ${OUT}"
