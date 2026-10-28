#!/usr/bin/env python3
"""
ETT forecasting example — ETTh1, ETTh2, ETTm1, ETTm2.

Loads one test window from each dataset, runs the Phase-2 model, and prints
per-channel MSE/MAE plus a plot of the first channel for each dataset.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

# ── Config ────────────────────────────────────────────────────────────────────

CKPT = (
    os.environ.get("FTS_RESULTS_ROOT", "results")
    + "/sincos_patch32/gemma_3_270m"
    "/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend"
    "/checkpoints/best_model.pt"
)
DATA_ROOT = os.environ.get("FTS_ETT_DIR", "data/ETT")

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

PATCH_SIZE  = 32
HISTORY_LEN = 512   # 16 patches per channel (512 / 32)
PRED_LEN    = 96    # 3 patches per channel (96 / 32)
WINDOW_IDX  = 0

# Informer standard fixed-count boundaries (matches Phase2DownstreamDataset)
_TRAIN_END = {"ETTh1": 8640,  "ETTh2": 8640,  "ETTm1": 34560, "ETTm2": 34560}
_VAL_END   = {"ETTh1": 11520, "ETTh2": 11520, "ETTm1": 46080, "ETTm2": 46080}
_TEST_END  = {"ETTh1": 14400, "ETTh2": 14400, "ETTm1": 57600, "ETTm2": 57600}

LORA_R     = 32
LORA_ALPHA = 64
DEVICE     = "cuda"

# ── Load model once ───────────────────────────────────────────────────────────

cfg = SimpleNamespace(
    is_pretraining=True,
    use_only_time_pe=False,
    use_only_channel_pe=False,
    use_freq_embed=False,
    patch_size=PATCH_SIZE,
    pe_type="sincos",
    enc_dec_type="transformer",
    mlp_dec_hidden_dim=64,
    mlp_dec_ff_dim=256,
    mlp_enc_hidden_dim=64,
    mlp_enc_ff_dim=256,
)

model = OpenTSLMSPFixPos2D(cfg, llm_id="google/gemma-3-270m", device=DEVICE)

ckpt_keys = torch.load(CKPT, map_location="cpu", weights_only=False).keys()
if "llm_state" not in ckpt_keys:
    model.enable_lora(lora_r=LORA_R, lora_alpha=LORA_ALPHA)
    print(f"LoRA enabled: r={LORA_R}  alpha={LORA_ALPHA}")

model.load_from_file(CKPT)
model.eval()
print(f"Loaded: {CKPT}\n")

# ── Per-dataset loop ──────────────────────────────────────────────────────────

summary = []   # [(dataset, mse, mae)]

for ds_name in DATASETS:
    csv_path = os.path.join(DATA_ROOT, f"{ds_name}.csv")
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[float, int])
    col_names = list(df.columns)
    C = df.shape[1]
    N = len(df)

    tr = _TRAIN_END[ds_name]
    va = _VAL_END[ds_name]

    te = _TEST_END[ds_name]
    train_tensor = torch.tensor(df.iloc[:tr].values, dtype=torch.float32)
    global_mean  = train_tensor.mean(dim=0)
    global_std   = train_tensor.std(dim=0).clamp(min=1e-5)
    test_tensor  = torch.tensor(df.iloc[va:te].values, dtype=torch.float32)

    print(f"  rows={N}  train=0:{tr}  test={va}:{te}  C={C}")

    window_len = HISTORY_LEN + PRED_LEN
    start      = WINDOW_IDX * PRED_LEN
    assert start + window_len <= len(test_tensor), (
        f"Window {WINDOW_IDX} out of range for {ds_name}: "
        f"test has {len(test_tensor)} rows, need {start + window_len}"
    )

    window   = test_tensor[start : start + window_len]
    hist_raw = window[:HISTORY_LEN]
    fut_raw  = window[HISTORY_LEN:]

    hist_norm = (hist_raw - global_mean) / global_std   # [H, C]
    fut_norm  = (fut_raw  - global_mean) / global_std   # [P, C]

    n_per_ch = PRED_LEN // PATCH_SIZE
    sample = {
        "type":       "chronos",
        "pre_prompt": (
            f"This is {ds_name} data. "
            f"You are given {HISTORY_LEN} timesteps across {C} channels:"
        ),
        "post_prompt": (
            f". Predict the next {PRED_LEN} timesteps "
            f"as {n_per_ch} patch{'es' if n_per_ch != 1 else ''} per channel"
        ),
        "time_series": [hist_norm[:, c].tolist() for c in range(C)],
        "time_series_text": [
            f"This is the time series ({col_names[c]}), "
            f"mean {global_mean[c].item():.4f} std {global_std[c].item():.4f}. <ts>"
            for c in range(C)
        ],
        "ts_future":             fut_norm.T.clone(),
        "ts_already_normalized": True,
        "n_channels":            C,
        "_n_channels":           C,
    }

    batch = extend_time_series_to_match_patch_size_and_aggregate(
        [sample], patch_size=PATCH_SIZE
    )

    with torch.no_grad():
        preds = model.forecast_phase2(
            batch, pred_len=PRED_LEN, max_new_tokens=512, logit_boost=50.0
        )

    if preds[0] is None:
        print(f"  ERROR: model returned None for {ds_name}")
        summary.append((ds_name, float("nan"), float("nan")))
        continue

    pred = preds[0].cpu().float()   # [C, PRED_LEN]
    gt   = fut_norm.T.float()       # [C, PRED_LEN]

    print(f"  {'channel':>12}   MSE      MAE")
    print(f"  {'-'*35}")
    for c, name in enumerate(col_names):
        p   = pred[c].numpy()
        g   = gt[c].numpy()
        mse = float(((p - g) ** 2).mean())
        mae = float(abs(p - g).mean())
        print(f"  {name:>12}   {mse:.4f}   {mae:.4f}")
    print(f"  {'-'*35}")
    all_mse = float(((pred - gt) ** 2).mean())
    all_mae = float((pred - gt).abs().mean())
    print(f"  {'ALL':>12}   {all_mse:.4f}   {all_mae:.4f}")
    summary.append((ds_name, all_mse, all_mae))

    # ── Plot channel 0 ────────────────────────────────────────────────────────
    ch   = 0
    h_np = hist_norm[:, ch].numpy()
    p_np = pred[ch].numpy()
    g_np = gt[ch].numpy()

    t_hist = np.arange(-HISTORY_LEN, 0)
    t_fut  = np.arange(0, PRED_LEN)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    ax = axes[0]
    ax.plot(t_hist, h_np, color="steelblue", linewidth=1.2, label=f"History ({HISTORY_LEN} steps)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.plot(t_fut, g_np, color="green", linewidth=1.5, label="Ground truth")
    ax.plot(t_fut, p_np, color="red",   linewidth=1.5, linestyle="--", label="Prediction")
    ax.set_title(f"{ds_name} — channel '{col_names[ch]}' — window {WINDOW_IDX}")
    ax.set_ylabel("Normalised value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_fut, g_np, color="green", linewidth=1.5, label="Ground truth")
    ax.plot(t_fut, p_np, color="red",   linewidth=1.5, linestyle="--", label="Prediction")
    ax.fill_between(t_fut, g_np, p_np, alpha=0.15, color="red", label="Error")
    ax.set_title(
        f"Forecast region — MSE={((p_np-g_np)**2).mean():.4f}  "
        f"MAE={abs(p_np-g_np).mean():.4f}"
    )
    ax.set_xlabel("Time step (0 = forecast start)")
    ax.set_ylabel("Normalised value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{ds_name.lower()}_window{WINDOW_IDX}_h{HISTORY_LEN}_p{PRED_LEN}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Plot → {out_path}")
    plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Summary  (history={HISTORY_LEN}  pred={PRED_LEN}  window={WINDOW_IDX})")
print(f"{'='*60}")
print(f"  {'Dataset':>8}   MSE      MAE")
print(f"  {'-'*30}")
for ds_name, mse, mae in summary:
    print(f"  {ds_name:>8}   {mse:.4f}   {mae:.4f}")
print(f"{'='*60}\n")
