#!/bin/bash
#SBATCH --job-name=eval-timer-etth1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source activate
conda activate opentslm

cd /home/s223540177/FoundationalTS

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

MODEL_ID="${MODEL_ID:-thuml/timer-base-84m}"
ETTH1_CSV="${ETTH1_CSV:-/scratch/s223540177/TimeSeriesECM/dataset/ETTh1.csv}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
HISTORY_LEN="${HISTORY_LEN:-480}"
PRED_LEN="${PRED_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ETTH1_STRIDE="${ETTH1_STRIDE:-1}"
PLOT_EVERY="${PLOT_EVERY:-0}"
OUT_DIR="${OUT_DIR:-./timer_etth1_eval/${EVAL_SPLIT}_h${HISTORY_LEN}_p${PRED_LEN}}"

CMD=(
  python /home/s223540177/FoundationalTS/evaluation/forecasting/inference_timer.py
  --model_id "${MODEL_ID}"
  --device cuda
  --dtype float32
  --etth1_csv "${ETTH1_CSV}"
  --eval_split "${EVAL_SPLIT}"
  --history_len "${HISTORY_LEN}"
  --pred_len "${PRED_LEN}"
  --batch_size "${BATCH_SIZE}"
  --etth1_stride "${ETTH1_STRIDE}"
  --plot_every "${PLOT_EVERY}"
  --out_dir "${OUT_DIR}"
)

echo "Launching Timer ETTh1 inference:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
