#!/bin/bash
#SBATCH --job-name kernel_synth
#SBATCH --output=slurm/logs/kernel_synth.out
#SBATCH --error=slurm/logs/kernel_synth.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time 1-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Univariate KernelSynth (Chronos v1). CPU-bound (sklearn GP).
# `uni2st` env carries gluonts + joblib + sklearn + tqdm needed by kernel-synth.py.

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}"

# Override defaults at sbatch time via --export, e.g.:
#   sbatch --export=ALL,NUM_SERIES=1000000,MAX_KERNELS=5 slurm/kernel_synth.slurm
export NUM_SERIES=${NUM_SERIES:-10000}
export MAX_KERNELS=${MAX_KERNELS:-5}
export OUT=${OUT:-${SYNTH_CHRONOS_REPO}/data/kernelsynth-data.arrow}

bash scripts/run_kernel_synth.sh
