#!/usr/bin/env python3
"""
Probe what the model generates at the ETTh1 forecasting prompt WITHOUT any logit boost.

Shows:
  1. Top-10 logits at the first generation step (where does <ts> rank?)
  2. 30-token free generation (greedy, no forcing, no boost)
"""
import os, sys
import pandas as pd
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

# ── Config ────────────────────────────────────────────────────────────────────
CKPT = (
    "/scratch/s223540177/FoundationalTS/result/sincos_patch32/gemma_3_270m"
    "/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend"
    "/checkpoints/best_model.pt"
)
CSV         = "/scratch/s223540177/TimeSeriesECM/dataset/ETTh1.csv"
PATCH_SIZE  = 32
PRED_LEN    = 96
HISTORY_LEN = 512
SPLIT       = (0.6, 0.2, 0.2)
LORA_R, LORA_ALPHA = 32, 64
DEVICE      = "cuda"

# ── Build one test window (same as example_etth1_forecast.py) ─────────────────
df = pd.read_csv(CSV)
df = df.select_dtypes(include=[float, int])
col_names = list(df.columns)
C, N = df.shape[1], len(df)
tr = int(N * SPLIT[0])
va = int(N * (SPLIT[0] + SPLIT[1]))

train_t     = torch.tensor(df.iloc[:tr].values, dtype=torch.float32)
global_mean = train_t.mean(0)
global_std  = train_t.std(0).clamp(min=1e-5)
test_t      = torch.tensor(df.iloc[va:].values, dtype=torch.float32)

window  = test_t[:HISTORY_LEN + PRED_LEN]
hist    = (window[:HISTORY_LEN] - global_mean) / global_std
fut     = (window[HISTORY_LEN:] - global_mean) / global_std

n_per_ch = PRED_LEN // PATCH_SIZE
sample = {
    "type":       "chronos",
    "pre_prompt": f"This is ETTh1 data. You are given {HISTORY_LEN} timesteps across {C} channels:",
    "post_prompt": f". Predict the next {PRED_LEN} timesteps as {n_per_ch} patches per channel",
    "time_series": [hist[:, c].tolist() for c in range(C)],
    "time_series_text": [
        f"This is the time series ({col_names[c]}), "
        f"mean {global_mean[c].item():.4f} std {global_std[c].item():.4f}. <ts>"
        for c in range(C)
    ],
    "ts_future":             fut.T.clone(),
    "ts_already_normalized": True,
    "n_channels":            C,
    "_n_channels":           C,
}

# ── Load model ────────────────────────────────────────────────────────────────
cfg = SimpleNamespace(
    is_pretraining=True, use_only_time_pe=False,
    use_only_channel_pe=False, use_freq_embed=False,
    patch_size=PATCH_SIZE, pe_type="sincos", enc_dec_type="transformer",
    mlp_dec_hidden_dim=64, mlp_dec_ff_dim=256,
    mlp_enc_hidden_dim=64, mlp_enc_ff_dim=256,
)
model = OpenTSLMSPFixPos2D(cfg, llm_id="google/gemma-3-270m", device=DEVICE)
if "llm_state" not in torch.load(CKPT, map_location="cpu", weights_only=False).keys():
    model.enable_lora(lora_r=LORA_R, lora_alpha=LORA_ALPHA)
model.load_from_file(CKPT)
model.eval()

ts_id  = model.ts_start_token_id
ts_end = model.ts_end_token_id
fp_id  = model.future_patch_token_id
print(f"\nSpecial token IDs:  <ts>={ts_id}  </ts>={ts_end}  <fp>={fp_id}\n")

# ── Prompt forward pass ───────────────────────────────────────────────────────
batch = extend_time_series_to_match_patch_size_and_aggregate([sample], patch_size=PATCH_SIZE)

with torch.no_grad():
    inputs_embeds, attn_mask, _, _, _, _, _ = model.pad_and_apply_batch(
        batch, return_targets=True
    )
    prompt_out  = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        return_dict=True,
    )
    valid_lens  = attn_mask.sum(1).long().clamp(min=1)
    next_logits = prompt_out.logits[0, valid_lens[0] - 1, :]   # [vocab]

# ── 1. Top-10 candidates at first generation step ────────────────────────────
print("=" * 60)
print("  Top-10 tokens the model naturally predicts (no boost)")
print("=" * 60)
ts_logit = next_logits[ts_id].item()
print(f"  <ts>  logit = {ts_logit:.2f}\n")

top = torch.topk(next_logits, 10)
for rank, (score, idx) in enumerate(zip(top.values, top.indices)):
    tok = model.tokenizer.decode([idx.item()])
    flag = ""
    if idx.item() == ts_id:    flag = "  ← <ts>"
    elif idx.item() == ts_end: flag = "  ← </ts>"
    elif idx.item() == fp_id:  flag = "  ← <fp>"
    print(f"  rank {rank+1:2d}  logit={score:8.2f}  {repr(tok):20s}{flag}")

# ── 2. Free generation: 30 tokens, greedy, no boost ──────────────────────────
print("\n" + "=" * 60)
print("  Free generation: 30 tokens, greedy, no logit boost")
print("=" * 60)
prompt_len = inputs_embeds.shape[1]

with torch.no_grad():
    out_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=30,
        do_sample=False,
    )

generated = out_ids[0, prompt_len:].tolist()
decoded   = model.tokenizer.decode(generated, skip_special_tokens=False)
print(f"\n  Raw decoded: {repr(decoded)}")
print(f"\n  Token-by-token:")
for i, tid in enumerate(generated):
    tok = model.tokenizer.decode([tid])
    flag = ""
    if tid == ts_id:    flag = "  ← <ts>"
    elif tid == ts_end: flag = "  ← </ts>"
    elif tid == fp_id:  flag = "  ← <fp>"
    print(f"    [{i:2d}] id={tid:6d}  {repr(tok):20s}{flag}")
