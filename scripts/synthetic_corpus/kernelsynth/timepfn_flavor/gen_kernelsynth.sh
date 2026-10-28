#!/bin/bash
# Generate the KernelSynth univariate synthetic dataset used as the
# independent-variate augmentation in TimePFN's curriculum pretraining.
# Produces the file referenced as `ind-kernelsynth-largechannel.arrow`
# in pfn_scripts/train/train_w_LMC_curriculum.sh and
# data_provider/data_factory.py (the `combine_Gaussian_datasets` branch).
#
# Note: synthetic_data_generation/kernel-synth.py hardcodes the output path
# to its own directory, so we run it in place and then move the file to
# OUT_DIR alongside the LMC-Synth arrow.
#
# Requires the `uni2st` conda env (has gluonts, joblib, sklearn, pyarrow, tqdm).
# For interactive runs: `conda activate uni2st && bash scripts/gen_kernelsynth.sh`.

set -euo pipefail

REPO_ROOT="${SYNTH_TIMEPFN_REPO:?set SYNTH_TIMEPFN_REPO to the egetaga/TimePFN checkout}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/datasets/synthetic}"
OUT_FILE="${OUT_FILE:-ind-kernelsynth-largechannel.arrow}"

NUM_SERIES="${NUM_SERIES:-20000}"
MAX_KERNELS="${MAX_KERNELS:-5}"

mkdir -p "${OUT_DIR}"

cd "${REPO_ROOT}"

python -u synthetic_data_generation/kernel-synth.py \
    -N "${NUM_SERIES}" \
    -J "${MAX_KERNELS}"

SRC="${REPO_ROOT}/synthetic_data_generation/ind-kernelsynth-largechannel.arrow"
DST="${OUT_DIR}/${OUT_FILE}"
mv -f "${SRC}" "${DST}"

echo "[gen_kernelsynth] wrote ${DST}"
