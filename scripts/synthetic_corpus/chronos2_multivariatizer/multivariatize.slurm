#!/bin/bash
#SBATCH --job-name mv_kernel_synth
#SBATCH --output=slurm/logs/multivariatize.out
#SBATCH --error=slurm/logs/multivariatize.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time 1-00:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Synthetic multivariate / covariate-informed data (Chronos-2 §4.2
# re-implementation). CPU-bound; `uni2st` env carries gluonts + joblib +
# sklearn + tqdm.

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}"

# Override defaults at sbatch time, e.g.:
#   sbatch --export=ALL,NUM_SERIES=200000,MAX_VARIATES=16,COVARIATE_PROB=0.5 \
#          slurm/multivariatize.slurm
export NUM_SERIES=${NUM_SERIES:-10000}
export MAX_KERNELS=${MAX_KERNELS:-5}
export MIN_LENGTH=${MIN_LENGTH:-1024}
export MAX_LENGTH=${MAX_LENGTH:-4096}
export MIN_VARIATES=${MIN_VARIATES:-3}
export MAX_VARIATES=${MAX_VARIATES:-10}
export MIN_SOURCES=${MIN_SOURCES:-3}
export MAX_SOURCES=${MAX_SOURCES:-10}
export MAX_LAG=${MAX_LAG:-32}
export COVARIATE_PROB=${COVARIATE_PROB:-0.0}
export MAX_COVARIATE_FRAC=${MAX_COVARIATE_FRAC:-0.5}
export SEED=${SEED:-0}
export OUT=${OUT:-${SYNTH_CHRONOS_REPO}/data/multivariate-kernelsynth.arrow}

bash scripts/run_multivariatize.sh
