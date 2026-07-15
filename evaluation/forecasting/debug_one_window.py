#!/usr/bin/env python3
"""Run forecast_phase2 on one ETTh1 test window and print history / pred / gt."""

import argparse
import getpass
import os
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/evaluation/forecasting")

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from eval_phase2_downstream import forecast_phase2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--hist_len", type=int, default=512)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--window_idx", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0, help="which channel to print")
    ap.add_argument("--show_n", type=int, default=30, help="how many timesteps to display")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = SimpleNamespace(
        pe_type="sincos",
        is_pretraining=False,
        gradient_checkpointing=False,
        patch_size=args.patch_size,
        enc_dec_type="mlp",
        mlp_enc_hidden_dim=64,
        mlp_enc_ff_dim=256,
        mlp_dec_hidden_dim=64,
        mlp_dec_ff_dim=256,
        use_only_time_pe=False,
        use_only_channel_pe=False,
    )
    print(f"Loading model from {args.ckpt}")
    model = OpenTSLMSPFixPos2D(args=cfg, llm_id=args.llm_id, device=device).to(device)
    model.enable_lora(lora_r=4, lora_alpha=8)
    model.load_from_file(args.ckpt)
    model.eval()

    # Load CSV → drop date column, take numeric, train/test split following ETTh convention
    df = pd.read_csv(args.csv)
    df = df.select_dtypes(include="number")
    n = len(df)
    # ETTh: train=8640, val=2880, test=2880
    if n > 14000:  # ETTh1/2
        train_end, val_end = 8640, 8640 + 2880
    else:
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.1)
    train_df = df.iloc[:train_end]
    test_df  = df.iloc[val_end:]

    mu  = torch.tensor(train_df.mean().values, dtype=torch.float32)
    std = torch.tensor(train_df.std().values,  dtype=torch.float32).clamp(min=1e-5)

    test_arr = torch.tensor(test_df.values, dtype=torch.float32)  # [T, C]
    T, C = test_arr.shape
    start = args.window_idx * args.pred_len
    if start + args.hist_len + args.pred_len > T:
        print(f"window_idx out of range; max={(T - args.hist_len - args.pred_len) // args.pred_len}")
        return

    hist_raw = test_arr[start : start + args.hist_len]            # [hist, C]
    fut_raw  = test_arr[start + args.hist_len : start + args.hist_len + args.pred_len]  # [pred, C]

    hist_n = ((hist_raw - mu) / std).T  # [C, hist]
    fut_n  = ((fut_raw  - mu) / std).T  # [C, pred]

    n_per_ch = (args.pred_len + args.patch_size - 1) // args.patch_size
    means = mu.unsqueeze(1).expand(-1, args.pred_len).cpu().numpy()
    stds  = std.unsqueeze(1).expand(-1, args.pred_len).cpu().numpy()

    sample = {
        "pre_prompt": (
            "Forecast the next "
            f"{args.pred_len} timesteps for each channel. "
            f"You will see {C} channels of historical data, each normalised to zero mean and unit std."
        ),
        "post_prompt": (
            f". Predict the next {args.pred_len} timesteps "
            f"as {n_per_ch} patch{'es' if n_per_ch != 1 else ''} per channel"
        ),
        "time_series":      [hist_n[c].tolist() for c in range(C)],
        "time_series_text": [
            f"Channel {c}, mean {means[c, 0]:.4f} std {stds[c, 0]:.4f}. <ts>"
            for c in range(C)
        ],
        "ts_already_normalized": True,
        "n_channels":  C,
        "_n_channels": C,
        "answer":      "",
    }

    print(f"\nWindow: idx={args.window_idx}  hist={args.hist_len}  pred={args.pred_len}  C={C}")
    with torch.no_grad():
        preds = forecast_phase2(
            model, [sample], pred_len=args.pred_len, patch_size=args.patch_size,
            max_new_tokens=C * n_per_ch * 2 + 64,
        )

    if preds[0] is None:
        print("forecast_phase2 returned None — generation failed")
        return

    pred_n = preds[0].cpu().float().numpy()        # [C, pred]
    fut_n_np = fut_n.cpu().numpy()                 # [C, pred]

    ch = args.channel
    show = args.show_n
    print(f"\n--- channel {ch} (showing first {show} of {args.pred_len} pred timesteps, normalised) ---")
    print(f"history end ({show} steps): {hist_n[ch, -show:].cpu().numpy().round(3).tolist()}")
    print(f"gt          (first {show}): {fut_n_np[ch, :show].round(3).tolist()}")
    print(f"pred        (first {show}): {pred_n[ch, :show].round(3).tolist()}")
    print()
    diff = pred_n[ch, :show] - fut_n_np[ch, :show]
    print(f"pred - gt   (first {show}): {diff.round(3).tolist()}")
    print()
    sample_mse = float(((pred_n - fut_n_np) ** 2).mean())
    sample_mae = float(np.abs(pred_n - fut_n_np).mean())
    print(f"This window MSE (all channels, normalised): {sample_mse:.4f}")
    print(f"This window MAE (all channels, normalised): {sample_mae:.4f}")


if __name__ == "__main__":
    main()
