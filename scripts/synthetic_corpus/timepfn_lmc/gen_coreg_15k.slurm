#!/bin/bash
#SBATCH --job-name=tpfn_coreg
#SBATCH --output=slurm/logs/gen_coreg_15k.out
#SBATCH --error=slurm/logs/gen_coreg_15k.err
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
# uni2st carries gluonts + joblib + sklearn + pyarrow needed by LMC_Synth.py;
# the project's default `time-series` env is missing gluonts.
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_TIMEPFN_REPO:?set SYNTH_TIMEPFN_REPO to the egetaga/TimePFN checkout}"
bash scripts/gen_coreg_15k.sh
