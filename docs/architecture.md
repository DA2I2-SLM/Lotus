Input timeseries [C, L]  (per-channel z-normed history; L = N_patches * 32 + 32 for one future patch)
        │
        ▼
[MLPEncoder2D]  src/opentslm/model/encoder/MLPEncoder2D.py
   Conv1d(1, 64, k=32, s=32)        →  [C, T, 64]   patchify (T = L/32)
   + 2D sincos PE (time D/2 ⊕ chan D/2)
   Position-wise MLP 64→256→64
        │  output: [C, T, enc_dim]
        ▼
[Projector]  64 → 3584 → 3584   (encoder embeddings → LLM hidden dim)
        │
        ▼
[LLM — Gemma-3-270M, FROZEN]  (called via _llm_last_hidden_only at OpenTSLMSPFixPos2D.py:268)
   Returns last_hidden [B, L_tokens, H=3584]   no CE / no logits
        │
        ▼
Extract hidden states at TS-patch positions via ts_mask
   ts_hidden [Σ patch_counts, 3584]
        │
        ▼
[Decoder Projector]  3584 → 3584 → enc_dim,  then LayerNorm
   pad_sequence by series → [N_ts, N_patches, enc_dim]
        │
        ▼
[MLPDecoder2D]  src/opentslm/model/decoder/MLPDecoder2D.py
   Position-wise MLP enc_dim → 64 → 256 → 64
   Patch head 64 → 32
   Output: [N_ts, N_patches, 32]   (per-patch reconstruction)
        │
        ▼
SHIFTED next-patch MSE   (compute_pretrain_loss, OpenTSLMSPFixPos2D.py:1989–2053)
   pred at position t   ⇄   target patch at t+1
   F.mse_loss(pred[valid], target[valid])
