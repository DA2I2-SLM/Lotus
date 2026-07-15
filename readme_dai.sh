#!/bin/bash
# =============================================================================
# readme_dai.sh — How to train Phase 2 (downstream) for OpenTSLMSPFixPos2D
# =============================================================================
# This is a documented, runnable how-to. Read the comments to understand the
# knobs, then either copy the sbatch line at the bottom or run one of the
# ready-made launchers in bashfiles/joint_phase2/patch32/.
#
# Phase 2 = downstream stage. It ALWAYS warm-starts from a Phase-1 checkpoint
# and continues training on the downstream tasks (HAR / Sleep / ECG CoT, plus
# optionally the numeric forecasting + ChatTS objectives).
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Environment
# -----------------------------------------------------------------------------
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opentslm
cd "/home/$(whoami)/FoundationalTS"

# -----------------------------------------------------------------------------
# 1. Warm start (REQUIRED for Phase 2)
# -----------------------------------------------------------------------------
# Phase 2 loads a Phase-1 checkpoint via `--warm_start_from pretrained`, which
# reads PRETRAINED_WARM_START_CKPT (defined in curriculum_learning.py):
#   /scratch/$USER/FoundationalTS/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt
#
# To point at a DIFFERENT Phase-1 checkpoint (e.g. a text-only phase-1), export:
#   export WARM_START_PRETRAINED_CKPT=/scratch/$USER/FoundationalTS/pretrained_weights/<...>/checkpoints/best_model.pt

# -----------------------------------------------------------------------------
# 2. What data Phase 2 trains on — controlled by env flags
# -----------------------------------------------------------------------------
# PHASE2_COT_ONLY   1 -> train ONLY on HAR/Sleep(/ECG) CoT   (no forecasting, no ChatTS)
#                   0 -> ALSO load numeric forecasting samples + ChatTS  (recovers the
#                        forecast head; use this if you care about forecasting quality)
#
# PHASE2_INCLUDE_ECG 1 -> add ECG-QA CoT as a third downstream task (lazy adapter,
#                         appended LAST for the curriculum). Default OFF.
#
# PHASE2_COT_MIX    natural | sqrt | balanced   (only when ECG is enabled)
#                   natural  -> tasks at natural size (ECG dominates the mix)
#                   sqrt     -> counts proportional to sqrt(size), normalized to ECG
#                   balanced -> HAR = Sleep = ECG  (1:1:1)
#
# SLEEP_UPSAMPLE N  repeat Sleep CoT N times (HAR+Sleep-only runs; e.g. 9 -> ~1:1 vs HAR)
# HAR_UPSAMPLE  N   repeat HAR  CoT N times
#
# RESULTS_BASE_DIR  where checkpoints/logs are written (best_model.pt lives under
#                   $RESULTS_BASE_DIR/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/
#                   joint_phase2_downstream_from_pretrained_nofreq/checkpoints/)

# -----------------------------------------------------------------------------
# 3. Recipes (pick ONE — these mirror the bashfiles/joint_phase2/patch32 launchers)
# -----------------------------------------------------------------------------
# (a) HAR+Sleep, balanced, CoT-only  ->  train_har_sleep_balanced.sh
#     export PHASE2_COT_ONLY=1 ; export SLEEP_UPSAMPLE=9
#     export RESULTS_BASE_DIR=/scratch/$USER/FoundationalTS/results_harsleep_balanced
#
# (b) HAR+Sleep+ECG sweep            ->  train_har_sleep_ecg_{natural,sqrt3,balanced,txlora}.sh
#     export PHASE2_COT_ONLY=1 ; export PHASE2_INCLUDE_ECG=1 ; export PHASE2_COT_MIX=balanced
#     export RESULTS_BASE_DIR=/scratch/$USER/FoundationalTS/results_harsleep_ecg_balanced
#
# (c) Level-1 fix: re-enable forecasting + ChatTS  ->  train_har_sleep_balanced_forecast.sh
#     export PHASE2_COT_ONLY=0 ; export SLEEP_UPSAMPLE=9
#     export RESULTS_BASE_DIR=/scratch/$USER/FoundationalTS/results_harsleep_balanced_fc

# -----------------------------------------------------------------------------
# 4. Launch (recommended: submit a ready-made bashfile with sbatch)
# -----------------------------------------------------------------------------
#   sbatch bashfiles/joint_phase2/patch32/train_har_sleep_balanced.sh
#   sbatch bashfiles/joint_phase2/patch32/train_har_sleep_ecg_balanced.sh
#   sbatch bashfiles/joint_phase2/patch32/train_har_sleep_balanced_forecast.sh
#
# Those scripts request 2x2 V100 (2 nodes x 2 GPUs), 20 epochs, and wrap the
# torchrun call in a restart loop. The core training command they run is below —
# set the env flags from section 3 first if you invoke it directly.

export PHASE2_COT_ONLY="${PHASE2_COT_ONLY:-1}"
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-/scratch/$(whoami)/FoundationalTS/results_harsleep_balanced}"
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=2 \
  curriculum_learning.py \
    --model OpenTSLMSPFixPos2D \
    --stages joint_phase2_downstream \
    --llm_id google/gemma-3-270m \
    --num_epochs 20 \
    --pe_type sincos \
    --patch_size 32 \
    --enc_dec_type mlp \
    --mlp_enc_hidden_dim 64 --mlp_enc_ff_dim 256 \
    --mlp_dec_hidden_dim 64 --mlp_dec_ff_dim 256 \
    --batch_size 2 \
    --val_batch_size 1 \
    --max_tokens_per_batch 1200 \
    --grad_accum_steps 8 \
    --mse_alpha 1.0 \
    --windows_per_col_train 0 \
    --dataloader_workers 0 \
    --save_every_n_steps 800 \
    --gradient_checkpointing \
    --warm_start_from pretrained

# -----------------------------------------------------------------------------
# 5. Evaluate the resulting checkpoint
# -----------------------------------------------------------------------------
#   CoT (HAR/Sleep acc + ECG exact-match/ROUGE-L):
#     python evaluation/cot/eval_cot_phase2.py \
#       --ckpt $RESULTS_BASE_DIR/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt \
#       --tasks har sleep ecg --split test --max_samples 300
# =============================================================================
