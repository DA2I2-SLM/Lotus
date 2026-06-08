#!/bin/bash
# Use a separate Chronos preprocessed-splits cache so this run does not
# race with concurrent jobs writing to the default
# _preprocessed_chronos_splits_mv directory. Read by ChronosDataset.py at
# import time; must be exported before torchrun launches.
export CHRONOS_PREPROCESSED_SPLITS_DIR="_preprocessed_chronos_splits_mv_freq"

torchrun \
      curriculum_learning.py \
        --model OpenTSLMSPFixPos2D \
        --stages phase1_chronos_mse \
        --llm_id google/gemma-3-270m \
        --num_epochs 20 \
        --pe_type sincos \
        --patch_size 32 \
        --enc_dec_type mlp \
        --chronos_window_stride 32 \
        --val_batch_size 4 \
        --batch_size 64 \
        --grad_accum_steps 8 \
        --mlp_enc_hidden_dim 64 \
        --mlp_enc_ff_dim 256 \
        --mlp_dec_hidden_dim 64 \
        --mlp_dec_ff_dim 256 \
        --use_freq_embed
