#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=eval_out/evs_%x_%j.out
#SBATCH --error=eval_out/evs_%x_%j.err
#
# ═══════════════════════════════════════════════════════════════════════════
# THE evaluation entry point for every baseline checkpoint.
#
#   sbatch --job-name=<tag> --time=HH:MM:SS bashfiles/evaluate/eval_baselines.sh \
#       <runner> <checkpoint> <out_dir> <task> [task ...]
#
# runner:
#   sp      OpenTSLM Soft-Prompt checkpoints  (opentslm-sp_*.pt)
#   fl      OpenTSLM Flamingo checkpoints     (opentslm-flamingo_*.pt)
#   joint   FoundationalTS SP+FixPos2D checkpoints (freq encoder)
# task: har | sleep | ecg | pamap2 | timemqa
#
# Examples:
#   sbatch --job-name=sp.har --time=06:00:00 bashfiles/evaluate/eval_baselines.sh \
#     sp  ./opentslm-sp_gemma-3-270m_joint-5task-finetune.pt  ./eval_out/sp/har  har
#   sbatch --job-name=fl.ecg --time=06:00:00 bashfiles/evaluate/eval_baselines.sh \
#     fl  ./opentslm-flamingo_gemma-3-270m_joint-5task-finetune.pt ./eval_out/fl/ecg ecg
#
# Notes that matter for correctness:
#   * "ecg" as the sole task automatically applies the PINNED 5000-sample
#     manifest (eval_manifests/ecg_test_first5000.json); it hard-fails on any
#     gold-hash mismatch so every model is scored on identical rows.
#   * SP generation is ~10 s/sample: shard long tasks with
#     --export=ALL,EVAL_SHARD_START=<lo>,EVAL_SHARD_END=<hi>  (row range).
#   * Output: <out_dir>/<task>_test_predictions.jsonl with
#     idx/gold/pred/label_gold/label_pred; accuracy = case-insensitive exact
#     match of label_pred vs label_gold.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail
RUNNER=$1; CK=$2; OUT=$3; shift 3; TASKS="$@"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO/eval_out"
module load Anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opentslm
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
cd "$REPO"
case $RUNNER in
  joint) SCRIPT=evaluation/cot/eval_cot_phase2.py;      EXTRA="--use_freq_embed";;
  sp)    SCRIPT=evaluation/official/eval_official_sp.py; EXTRA="";;
  fl)    SCRIPT=evaluation/official/eval_official_fl.py; EXTRA="";;
  *) echo "runner must be joint|sp|fl"; exit 1;;
esac
MFLAG=""
[ "$TASKS" = "ecg" ] && MFLAG="--sample_manifest $REPO/eval_manifests/ecg_test_first5000.json"
python -u "$SCRIPT" --ckpt "$CK" --llm_id google/gemma-3-270m \
  --tasks $TASKS --split test --max_samples 0 $MFLAG $EXTRA \
  --out_dir "$OUT"
echo "EVAL_DONE $RUNNER $TASKS"
