#!/bin/bash
#SBATCH --job-name s2gen_test
#SBATCH --output=slurm/logs/s2gen_test.out
#SBATCH --error=slurm/logs/s2gen_test.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=32gb
#SBATCH --time 0-02:00:00
#SBATCH --partition=gpu              # adjust to your cluster
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
##SBATCH --qos=batch-short      # site-specific QoS: enable/adjust for your cluster

module load Anaconda3 2>/dev/null || true   # site-specific; any env with the deps below works
source activate
conda activate "${SYNTH_SYMTIME_ENV:-./.env}"   # SymTime venv with s2generator installed

cd "${SYNTH_SYMTIME_REPO:?set SYNTH_SYMTIME_REPO to the wwhenxuan/SymTime checkout}"

ROOT_PATH=${SYNTH_SYMTIME_REPO}/data/datasets/pretrain_data_test/
mkdir -p "${ROOT_PATH}"

echo "[$(date)] running small test (100 seeds, 8 threads)"

python -u data_provider/s2generator_main.py \
  --root_path "${ROOT_PATH}" \
  --start_seed 0 \
  --end_seed   100 \
  --max_input_dim 6 \
  --max_output_dim 6 \
  --length 768 \
  --num_threads 8

echo "[$(date)] test done"
