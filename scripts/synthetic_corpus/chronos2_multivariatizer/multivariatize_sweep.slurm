#!/bin/bash
#SBATCH --job-name=mv_sweep
#SBATCH --output=slurm/logs/mv_sweep_%A_%a.out
#SBATCH --error=slurm/logs/mv_sweep_%A_%a.err
#SBATCH --array=1-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=5-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Diversity sweep for the chronos-forecasting multivariate pipeline.
# Three array tasks emit three arrow files under data/ that together form a
# diverse pretraining corpus:
#   task 1 (short):   L=1024, low D/K/lag,  MAX_KERNELS=3
#   task 2 (medium):  L=2048, mid D/K/lag,  MAX_KERNELS=5
#   task 3 (long):    L=4096, high D/K/lag, MAX_KERNELS=7
# D ranges are kept inside the unified-collector envelope [3,10] so no items
# are dropped during collection. COVARIATE_PROB=0 across the sweep since the
# unified schema is target-only.
#
# Override NUM_SERIES per array task at sbatch time:
#   sbatch --export=ALL,NUM_SERIES=300000 slurm/multivariatize_sweep.slurm

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

REPO=${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}
cd "${REPO}"

NUM_SERIES=${NUM_SERIES:-100000}

case "${SLURM_ARRAY_TASK_ID}" in
    1)
        TAG=short
        export MIN_LENGTH=512 MAX_LENGTH=1024
        export MAX_KERNELS=3
        export MIN_VARIATES=3
        export MAX_VARIATES=5
        export MIN_SOURCES=2
        export MAX_SOURCES=5
        export MAX_LAG=16
        export SEED=1
        ;;
    2)
        TAG=medium
        export MIN_LENGTH=1024 MAX_LENGTH=3096
        export MAX_KERNELS=5
        export MIN_VARIATES=4
        export MAX_VARIATES=8
        export MIN_SOURCES=3
        export MAX_SOURCES=8
        export MAX_LAG=64
        export SEED=2
        ;;
    3)
        TAG=long
        export MIN_LENGTH=4096 MAX_LENGTH=8192
        export MAX_KERNELS=7
        export MIN_VARIATES=5
        export MAX_VARIATES=10
        export MIN_SOURCES=4
        export MAX_SOURCES=10
        export MAX_LAG=128
        export SEED=3
        ;;
    *)
        echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

export NUM_SERIES
export COVARIATE_PROB=0.0
export OUT="${REPO}/data/mv_${TAG}.arrow"

echo "[mv_sweep ${TAG}] L=${MIN_LENGTH} D=[${MIN_VARIATES},${MAX_VARIATES}] K=[${MIN_SOURCES},${MAX_SOURCES}] MAX_LAG=${MAX_LAG} N=${NUM_SERIES}"
bash scripts/run_multivariatize.sh
