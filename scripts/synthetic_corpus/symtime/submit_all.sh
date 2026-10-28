#!/bin/bash
# Submits the full S2Generator pre-train data generation (1M seeds) as a slurm array.
# Adjust --array in s2generate.sh to change the total seed range.
set -euo pipefail

cd "${SYNTH_SYMTIME_REPO:?set SYNTH_SYMTIME_REPO to the wwhenxuan/SymTime checkout}"

mkdir -p slurm/logs

sbatch slurm/s2generate.sh
