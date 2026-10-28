#!/bin/bash
#SBATCH --job-name=lmc_sweep
#SBATCH --output=slurm/logs/lmc_sweep_%A_%a.out
#SBATCH --error=slurm/logs/lmc_sweep_%A_%a.err
#SBATCH --array=1-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=5-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Diversity sweep for the TimePFN LMC-Synth pipeline.
# Three array tasks emit three arrow files under datasets/synthetic/ that
# together form a diverse multivariate pretraining corpus. Axes varied per task:
#
#   - length  L                  (short=1024, medium=2048, long=4096; pinned)
#   - channels C range           (short=[3,5], medium=[4,8], long=[5,10])
#   - max_kernels                (3, 5, 7)
#   - Dirichlet alpha range      (sparse, balanced, dense channel mixing)
#   - Weibull (shape, scale)     (fewer / mid / more latent functions)
#
# Dirichlet small alpha ⇒ each channel is mostly one latent (near-independent
# variates); large alpha ⇒ channels mix many latents (highly correlated).
# Weibull scale controls the expected number of latent functions per series.
#
# Override NUM_SERIES per array task at sbatch time:
#   sbatch --export=ALL,NUM_SERIES=200000 slurm/gen_lmc_sweep.slurm

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

REPO=${SYNTH_TIMEPFN_REPO:?set SYNTH_TIMEPFN_REPO to the egetaga/TimePFN checkout}
cd "${REPO}"

export NUM_SERIES=${NUM_SERIES:-200000}

case "${SLURM_ARRAY_TASK_ID}" in
    1)
        TAG=short
        export MIN_LENGTH=512 MAX_LENGTH=1024
        export MIN_CHANNELS=3  MAX_CHANNELS=5
        export MAX_KERNELS=3
        export DIRICHLET_MIN=0.05 DIRICHLET_MAX=0.5   # sparse channel mixing
        export WEIBULL_SHAPE=2.0  WEIBULL_SCALE=5.0   # fewer latents
        export SEED=1
        ;;
    2)
        TAG=medium
        export MIN_LENGTH=1024 MAX_LENGTH=3096
        export MIN_CHANNELS=4  MAX_CHANNELS=8
        export MAX_KERNELS=5
        export DIRICHLET_MIN=0.1  DIRICHLET_MAX=2.0   # balanced (paper default)
        export WEIBULL_SHAPE=2.0  WEIBULL_SCALE=10.0  # mid
        export SEED=2
        ;;
    3)
        TAG=long
        export MIN_LENGTH=4096 MAX_LENGTH=8192
        export MIN_CHANNELS=5  MAX_CHANNELS=10
        export MAX_KERNELS=7
        export DIRICHLET_MIN=0.5  DIRICHLET_MAX=5.0   # dense channel mixing
        export WEIBULL_SHAPE=2.0  WEIBULL_SCALE=15.0  # more latents
        export SEED=3
        ;;
    *)
        echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

export OUT_FILE="lmc_${TAG}.arrow"

echo "[lmc_sweep ${TAG}] L=${MIN_LENGTH} C=[${MIN_CHANNELS},${MAX_CHANNELS}] MK=${MAX_KERNELS} alpha=[${DIRICHLET_MIN},${DIRICHLET_MAX}] weibull=(${WEIBULL_SHAPE},${WEIBULL_SCALE}) N=${NUM_SERIES} -> ${OUT_FILE}"
bash scripts/gen_coreg_15k.sh
