Dataset sample (Phase1JointDataset.py:1000–1064)
   series      [T, C]        per-channel z-normed
   ts_future   [C, P]        ground-truth next patch, per channel
        │
        ▼  split into C rows  (OpenTSLMSPFixPos2D.py:536–564)
ts_list   :  C rows of shape (1, L), accumulated across the batch
chan_id  :  [0, 1, …, C-1]   (resets per sample; not a global vocab)
        │
        ▼
Encoder MLPEncoder2D.forward(x=(C,L), chan_id=(C,))
   Conv1d(1→D, k=P, s=P) applied per channel independently
   + 2D sincos PE: D/2 dims for time, D/2 dims for channel index
   → tokens [C, T, D]      (no cross-channel mixing in encoder)
        │
        ▼  time-major flatten  (OpenTSLMSPFixPos2D.py:727–737)
LLM token sequence per sample:
   [ t₀c₀, t₀c₁, …, t₀c_{C-1},  t₁c₀, …,  t_{T-1}c_{C-1} ]   length T·C
        │
        ▼
LLM (Gemma-3-270M, frozen) — self-attention here is the ONLY place
   channels actually exchange information
   last_hidden  [T·C, H=3584]   per sample
        │
        ▼  unflatten back to channel-major  (OpenTSLMSPFixPos2D.py:2006–2023)
   reshape [T, C, H] → permute → [C, T, H]
        │
        ▼  decoder_projector + LayerNorm → split by channel-row
   ts_hidden_3d  [C, T, enc_dim]
        │
        ▼
Decoder MLPDecoder2D
   position-wise MLP (no cross-channel, no cross-time mixing)
   → [C, T, P]    one patch prediction per (channel, time-step)
        │
        ▼  shifted next-patch MSE  (OpenTSLMSPFixPos2D.py:2041–2051)
   target = patch at t+1 of the same channel
   loss = F.mse_loss(pred[valid], target[valid])
        ── channel-agnostic average over all valid (channel, patch) pairs
