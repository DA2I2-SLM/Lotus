#!/bin/bash
# Language eval (TSQA MCQ accuracy + M4 captioning ROUGE) for the TRANSFORMER
# joint_phase1_extend_balanced checkpoint trained by
# bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_nofreq.sh.
#
# Outputs:
#   eval_results/trans_joint_phase1_extbal_nofreq/language/tsqa/
#   eval_results/trans_joint_phase1_extbal_nofreq/language/m4/
#
#SBATCH --job-name=eval-lang-trans-joint-extbal-nofreq
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=output_job/%x.out
#SBATCH --error=output_job/%x.err
#SBATCH --constraint=gpu-v100
#SBATCH --exclude=v100l-f-[01-06]

set -euo pipefail
mkdir -p output_job

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /scratch/s225250685/project/FoundationalTS
conda activate ./.env

ENV_PY="$(pwd)/.env/bin/python"
# Sanity check: fail fast if the conda env did not actually activate.
"${ENV_PY}" -c "import torch" || { echo "ERROR: torch not importable from ${ENV_PY}"; exit 1; }

LLM=google/gemma-3-270m
TSQA_EVAL=evaluation/baseline/evaluate_tsqa_opentslm.py
M4_EVAL=evaluation/captioning/m4_eval.py

CKPT=/scratch/s225250685/project/FoundationalTS/results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_nofreq/checkpoints/best_model.pt
OUT_BASE=./eval_results/trans_joint_phase1_extbal_nofreq/language

echo "CKPT=${CKPT}"

echo "================================================================"
echo "[eval_language] TSQA MCQ -> ${OUT_BASE}/tsqa"
echo "================================================================"
${ENV_PY} ${TSQA_EVAL} \
  --ckpt       ${CKPT} \
  --llm_id     ${LLM} \
  --tasks      tsqa \
  --patch_size 32 \
  --pe_type    sincos \
  --enc_dec_type transformer \
  --device     cuda \
  --out_dir    ${OUT_BASE}/tsqa

echo "================================================================"
echo "[eval_language] M4 captioning -> ${OUT_BASE}/m4"
echo "================================================================"
${ENV_PY} ${M4_EVAL} \
  --ckpt           ${CKPT} \
  --llm_id         ${LLM} \
  --device         cuda \
  --batch_size     32 \
  --patch_size     32 \
  --pe_type        sincos \
  --enc_dec_type   transformer \
  --max_new_tokens 128 \
  --out_dir        ${OUT_BASE}/m4
