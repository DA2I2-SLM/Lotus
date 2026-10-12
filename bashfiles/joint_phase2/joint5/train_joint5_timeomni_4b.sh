#!/bin/bash
#SBATCH --job-name=j5_timeomni4b
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gpus=4
#SBATCH --constraint="gpu-l40s|gpu-rtxp"
#SBATCH --output=output_job/%x_%j.out
#SBATCH --error=output_job/%x_%j.err

# Resolve repo root + interpreter + data locations dynamically so this script
# runs on any account. Override any FTS_* var in your environment.
# NOTE: under sbatch SLURM executes a COPY of this script from its spool dir
# (/var/spool/slurm/...), so ${BASH_SOURCE[0]} is NOT inside the repo. Try
# SLURM_SUBMIT_DIR and $PWD as well, and fail with an actionable message
# instead of an unbound-variable error further down.
_d=""
for _c in "${FTS_REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" "$PWD"; do
  _p="$_c"
  while [ -n "$_p" ] && [ "$_p" != "/" ]; do
    if [ -f "$_p/bashfiles/_paths.sh" ]; then _d="$_p"; break; fi
    _p="$(dirname "$_p")"
  done
  [ -n "$_d" ] && break
done
[ -n "$_d" ] || { echo "ERROR: cannot locate bashfiles/_paths.sh - set FTS_REPO_ROOT" >&2; exit 2; }
# shellcheck source=/dev/null
source "$_d/bashfiles/_paths.sh"
#
# ─────────────────────────────────────────────────────────────────────────────
# TimeOmni-1-4B fine-tuned with OUR recipe: same five-task mixture, same
# serialisation, same sampling, same selection metric, same LoRA rank/alpha as
# every other adapter arm. Answers a question none of the other rows can:
# does time-series-SPECIFIC pretraining help the serialised-digits interface?
#
# NOT a matched-backbone comparison. At 4.21B this is ~4x ours-1b, so it belongs
# beside the context rows, not in the like-for-like block. Label it as such.
#
# ── CONTAMINATION AUDIT (done before running this; see the paper footnote) ────
# TimeOmni-1's training corpus is enumerated and CLOSED in arXiv:2509.24803
# Appendix D.1: synthetic scenario data (Merrill et al.), CausalRivers, NYC
# Barclays-Center taxi, CityLearn (+ EWELD as OOD *test* only). The per-task
# counts in Appendix D.2 Table 3 sum to exactly the 23,605 QA pairs the abstract
# claims, leaving no unaccounted budget. String counts across arXiv v1 and v3:
# HAR 0, Sleep 0, ECG 0, PTB 0, PAMAP 0, UCI 0, EEG 0. Time-MQA appears 11 times
# but only as the motivating foil and as baseline MODEL names, never as their
# training data. So none of our six columns are in their corpus.
# THE REAL CAVEAT IS PROVENANCE, NOT OVERLAP: the paper trains Qwen2.5-7B
# (grep Qwen3 = 0 in all versions); this 4B checkpoint is an undocumented
# Apr-2026 release on Qwen3.5-4B with no training-data section and no released
# training data (authors: public "second half of 2026"), so the clean-corpus
# finding is authoritative for the paper's model and presumptive for these
# weights. Do not describe this row as "verified uncontaminated".
#
# ── ENVIRONMENT ──────────────────────────────────────────────────────────────
# Needs transformers >=5.x: the config is model_type qwen3_5, written by
# transformers 5.3.0.dev0, and 4.57.3 (the opentslm env) does not know that
# model type and there is no auto_map for trust_remote_code. A SEPARATE venv is
# used deliberately: upgrading the shared env would change the library under
# every eval shard launched afterwards and silently alter finished rows.
# It is SELF-CONTAINED (no --system-site-packages) on purpose: an inheriting
# venv picked up the conda env's peft 0.17.1 (imports HybridCache, removed in
# transformers 5.x) and torchao 0.15.0 (rejected by peft>=0.18, while
# torchao>0.16 itself needs torch>=2.11). Three-way deadlock; isolating ends it.
#   ${FTS_SCRATCH}/venv_timeomni2  (self-contained: own torch 2.9.1+cu128, transformers 5.15.0, peft 0.20.0, NO torchao)
#
# ── LoRA TARGETS ─────────────────────────────────────────────────────────────
# r=8, alpha=16, dropout 0 — identical to ours-1b and to the tokenized-1b LoRA
# arm. The TARGET LIST is extended, and that is deliberate. Qwen3.5 is a hybrid:
# text_config.full_attention_interval=4, layer_types = 3x linear_attention then
# 1x full_attention, so only 8 of 32 layers even have q/k/v/o_proj; the other 24
# use linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.
# Passing the bare seven names would adapt 8/32 attention blocks (10.62M params,
# 71.7% of linear params) where the same list adapts every block on gemma-3-1b.
# Adding in_proj_qkv / in_proj_z / out_proj restores full per-layer coverage
# (200 modules, 15.24M params). in_proj_a and in_proj_b are 32x2560 gating
# vectors and are left alone. Measured, not assumed — see the sizing run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm          # for the dataset code on PYTHONPATH
cd ${FTS_REPO_ROOT}

PY=${FTS_SCRATCH}/venv_timeomni2/bin/python
MODE="${1:-full}"
RES=${FTS_RESULTS_ROOT}_joint5_timeomni4b

MODEL=$(ls -d ${FTS_VAST}/.cache/huggingface/hub/models--TimeOmni-1--TimeOmni-1-4B/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && MODEL=$(ls -d ${FTS_HF_HUB}/models--TimeOmni-1--TimeOmni-1-4B/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then echo "FATAL: no local TimeOmni-1-4B snapshot"; exit 1; fi
echo "[timeomni] snapshot: $MODEL"

if [ "$MODE" = "smoke" ]; then
  EPOCHS=1; NPROC=1; GEN_EM_N=4; ACCUM=4; EVAL_EVERY=2
  BALANCE_TARGET=50; RES="${RES}_smoke"; rm -rf "$RES"
else
  EPOCHS=20; NPROC=4; GEN_EM_N=192; ACCUM=32; EVAL_EVERY=500
  BALANCE_TARGET=40000
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${FTS_SCRATCH}/hf_cache
export HF_DATASETS_CACHE=${FTS_SCRATCH}/hf_cache/datasets
export PYTHONPATH=${FTS_REPO_ROOT}/src:${PYTHONPATH:-}

# ---- identical sampling to every other joint5 run ----
export PHASE2_COT_ONLY=1
export PHASE2_INCLUDE_ECG=1
export PHASE2_INCLUDE_PAMAP2=1
export PHASE2_COT_TASKS=har,sleep,ecg,pamap2,timemqa
export PHASE2_COT_MIX=balanced_all
export PHASE2_BALANCE_TARGET=$BALANCE_TARGET

echo "=== [TimeOmni-1-4B] JOINT-5 balanced_all | mode=$MODE | LoRA r=8 a=16 | results=$RES ==="
$PY -c "import transformers,torch;print('  transformers',transformers.__version__,'torch',torch.__version__)"

# MUST launch via the venv python: a bare `torchrun` resolves to the conda env's
# entry point and would run transformers 4.57.3, which cannot load qwen3_5.
$PY -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
  baselines/tokenized_llm/tokenized_sft.py \
  --mode train \
  --model_id "$MODEL" \
  --out_dir "$RES" \
  --epochs $EPOCHS \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum $ACCUM \
  --max_len 4096 \
  --max_values 512 \
  --gen_em_n $GEN_EM_N \
  --eval_every $EVAL_EVERY \
  --patience 8 \
  --lora_r 8 --lora_alpha 16 \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj

echo "DONE_TIMEOMNI4B_JOINT5 best=${RES}/best"
