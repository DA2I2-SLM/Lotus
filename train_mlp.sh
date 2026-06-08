torchrun \
  /home/s223540177/FoundationalTS/curriculum_learning.py \
    --model OpenTSLMSPFixPos2D \
    --stages phase1_chronos_mse \
    --llm_id google/gemma-3-270m \
    --num_epochs 20 \
    --pe_type sincos \
    --patch_size 32 \
    --enc_dec_type mlp \
    --chronos_window_stride 32 \
    --val_batch_size 32 \
    --grad_accum_steps 4 \
    --base_batch_size 128 \
    --mlp_enc_hidden_dim 64 \
    --mlp_enc_ff_dim 256 \
    --mlp_dec_hidden_dim 64 \
    --mlp_dec_ff_dim 256