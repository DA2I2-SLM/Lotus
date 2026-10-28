#!/bin/bash
# Render 20 sample PNGs per synthetic-data file to figures/<dataset>/.
set -euo pipefail

REPO=${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}
N=${N:-20}
SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-${REPO}/data}
FIG_DIR=${FIG_DIR:-${REPO}/figures}

declare -A FILES=(
    ["kernelsynth"]="${DATA_DIR}/kernelsynth-data.arrow"
    ["multivariate"]="${DATA_DIR}/multivariate-kernelsynth.arrow"
    ["covariate"]="${DATA_DIR}/multivariate-kernelsynth.cov.arrow"
)

for name in "${!FILES[@]}"; do
    arrow=${FILES[$name]}
    out_dir=${FIG_DIR}/${name}
    if [ ! -f "${arrow}" ]; then
        echo "[run_visualize] skip ${name}: ${arrow} not found"
        continue
    fi
    echo "[run_visualize] ${name}: ${arrow} -> ${out_dir} (${N} samples)"
    python "${REPO}/scripts/visualize_samples.py" \
        --arrow "${arrow}" \
        --out-dir "${out_dir}" \
        --num-samples "${N}" \
        --seed "${SEED}"
done

echo "[run_visualize] done. Figures under ${FIG_DIR}/"
