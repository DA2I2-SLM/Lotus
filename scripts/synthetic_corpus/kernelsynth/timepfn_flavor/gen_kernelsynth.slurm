#!/bin/bash
#SBATCH --job-name=tpfn_kern
#SBATCH --output=slurm/logs/gen_kernelsynth.out
#SBATCH --error=slurm/logs/gen_kernelsynth.err
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=12:00:00

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
# uni2st carries gluonts + joblib + sklearn + pyarrow needed by kernel-synth.py;
# the project's default `time-series` env is missing gluonts.
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_TIMEPFN_REPO:?set SYNTH_TIMEPFN_REPO to the egetaga/TimePFN checkout}"
bash scripts/gen_kernelsynth.sh
