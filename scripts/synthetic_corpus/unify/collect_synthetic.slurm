#!/bin/bash
#SBATCH --job-name=tpfn_collect
#SBATCH --output=logs/collect_synthetic.out
#SBATCH --error=logs/collect_synthetic.err
#SBATCH --partition=cpu              # adjust to your cluster
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=12:00:00

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
# uni2st has gluonts + pyarrow + numpy + tqdm; the project's default `time-series`
# env lacks gluonts.
conda activate "${SYNTH_CONDA_ENV:-uni2st}"

cd "${SYNTH_UNIFY_DIR:-${SLURM_SUBMIT_DIR:-.}}"   # the directory holding collect_synthetic.py (submit from there)
bash scripts/run_collect.sh
