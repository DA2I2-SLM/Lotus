#!/bin/bash
#SBATCH --job-name=ks_sweep
#SBATCH --output=slurm/logs/ks_sweep_%A_%a.out
#SBATCH --error=slurm/logs/ks_sweep_%A_%a.err
#SBATCH --array=1-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Diversity sweep for upstream KernelSynth (univariate, Chronos v1).
# Three array tasks emit three arrow files under data/ that together form a
# diverse univariate pretraining corpus:
#   task 1 (short):   L=1024, MAX_KERNELS=3
#   task 2 (medium):  L=2048, MAX_KERNELS=5
#   task 3 (long):    L=4096, MAX_KERNELS=7
#
# Each item's `target` is a 1D array of length L. To include these in the
# `unified/` corpus, run the collector with `--min-variates 1` (the default
# envelope drops univariate items).
#
# Override NUM_SERIES per array task at sbatch time:
#   sbatch --export=ALL,NUM_SERIES=1000000 slurm/kernel_synth_sweep.slurm

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

REPO=${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}
cd "${REPO}"

NUM_SERIES=${NUM_SERIES:-50000}

case "${SLURM_ARRAY_TASK_ID}" in
    1) TAG=short;  LENGTH=1024; MAX_KERNELS=3; SEED=1 ;;
    2) TAG=medium; LENGTH=2048; MAX_KERNELS=5; SEED=2 ;;
    3) TAG=long;   LENGTH=4096; MAX_KERNELS=7; SEED=3 ;;
    *) echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

OUT="${REPO}/data/ks_${TAG}.arrow"

echo "[ks_sweep ${TAG}] L=${LENGTH} max_kernels=${MAX_KERNELS} N=${NUM_SERIES} seed=${SEED} -> ${OUT}"
python scripts/kernel_synth_sweep.py \
    --num-series "${NUM_SERIES}" \
    --max-kernels "${MAX_KERNELS}" \
    --length "${LENGTH}" \
    --seed "${SEED}" \
    --out "${OUT}"
