#!/bin/bash
#SBATCH --job-name viz_synth
#SBATCH --output=slurm/logs/visualize.out
#SBATCH --error=slurm/logs/visualize.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time 0-01:00:00
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

# Render 20 sample PNGs per synthetic-data file (kernelsynth, multivariate,
# covariate-informed) to figures/<dataset>/. CPU-only matplotlib job.
# `uni2st` env carries matplotlib + gluonts + numpy.

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_CHRONOS_REPO:?set SYNTH_CHRONOS_REPO to the amazon-science/chronos-forecasting checkout}"

# Override defaults at sbatch time, e.g.:
#   sbatch --export=ALL,N=50,SEED=7 slurm/visualize.slurm
export N=${N:-20}
export SEED=${SEED:-0}
export DATA_DIR=${DATA_DIR:-${SYNTH_CHRONOS_REPO}/data}
export FIG_DIR=${FIG_DIR:-${SYNTH_CHRONOS_REPO}/figures}

bash scripts/run_visualize.sh
