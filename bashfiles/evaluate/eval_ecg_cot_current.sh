#!/bin/bash
#SBATCH --job-name=ecg_cot_eval
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

# Re-eval the CURRENT best_model.pt of each ECG-sweep config on HAR+Sleep+ECG CoT.
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/${USER}_ecgcot_${SLURM_JOB_ID}
cd /home/s223540177/FoundationalTS
SNAP=/home/${USER}/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
B=/scratch/s223540177/FoundationalTS
REL=sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt
OUTB=/scratch/s223540177/cik_work/ecg_eval_current

for cfg in natural sqrt3 balanced txlora; do
  ck="$B/results_harsleep_ecg_$cfg/$REL"
  echo "############ EVAL $cfg ($(date +%H:%M)) ############"
  [ -f "$ck" ] || { echo "MISSING $ck"; continue; }
  python evaluation/cot/eval_cot_phase2.py \
    --ckpt "$ck" --llm_id "$SNAP" --tasks har sleep ecg --split test --max_samples 300 \
    --out_dir "$OUTB/$cfg" 2>&1 | grep -E "Stage:|✓ |accuracy=|epoch=|Error|OOM"
  echo "==== DONE $cfg ($(date +%H:%M)) ===="
done
echo "ALL_ECG_COT_CURRENT_COMPLETE"
