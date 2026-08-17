#!/bin/bash
#SBATCH --job-name=ecg_bal_only
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]
set -uo pipefail
mkdir -p output_job
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opentslm
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/${USER}_bal_${SLURM_JOB_ID}
cd /home/s223540177/FoundationalTS
SNAP=/home/${USER}/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
CK=/scratch/s223540177/FoundationalTS/results_harsleep_ecg_balanced/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt
python evaluation/cot/eval_cot_phase2.py --ckpt "$CK" --llm_id "$SNAP" \
  --tasks har sleep ecg --split test --max_samples 300 \
  --out_dir /scratch/s223540177/cik_work/ecg_eval_current/balanced 2>&1 | grep -E "Stage:|✓ |accuracy=|epoch=|Error|OOM"
echo "BALANCED_ONLY_COMPLETE"
