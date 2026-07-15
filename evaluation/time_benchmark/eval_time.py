#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
TIME benchmark zero-shot evaluation for OpenTSLMSPFixPos2D.

Uses forecast_phase2() for inference on all 49 TIME datasets.
Point forecasts are tiled to 9 quantile levels; MASE and CRPS are
computed by timebench.evaluation.saver.save_window_predictions().

Prerequisites
-------------
  1. Install timebench:
       git clone https://github.com/zqiao11/TIME.git
       cd TIME && pip install -e .
  2. Download TIME datasets from HuggingFace and set the path:
       echo "TIME_DATASET=/path/to/time_datasets" >> .env

Usage
-----
  # All 49 datasets, short-horizon term:
  python eval_time.py \\
    --ckpt /scratch/.../best_model.pt \\
    --llm_id google/gemma-3-270m

  # Single dataset, all horizon terms:
  python eval_time.py --ckpt ... --datasets Global_Influenza --term all

  # Phase-1 checkpoint (no LoRA):
  python eval_time.py --ckpt .../phase1.pt --lora_r 0 --llm_id ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from gluonts.time_feature import get_seasonality
from tqdm import tqdm

# timebench — install from https://github.com/zqiao11/TIME
from timebench.evaluation.data import Dataset, load_dataset_config, get_dataset_settings
from timebench.evaluation.saver import save_window_predictions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONTEXT_LEN = 512   # fixed history window; multiple of patch_size ≤ 32


@torch.no_grad()
def predict_window(
    model: OpenTSLMSPFixPos2D,
    context: np.ndarray,   # (D, T) float32, raw scale
    pred_len: int,
    patch_size: int,
    logit_boost: float,
) -> np.ndarray:
    """Return point forecast (D, pred_len) in original (unnormalized) scale."""
    D, T = context.shape

    # Truncate to last CONTEXT_LEN steps, or zero-pad if series is shorter
    if T >= CONTEXT_LEN:
        ctx = context[:, -CONTEXT_LEN:]
    else:
        ctx = np.pad(context, ((0, 0), (CONTEXT_LEN - T, 0)))

    # Per-channel z-score normalization
    means = ctx.mean(axis=1, keepdims=True)
    stds  = ctx.std(axis=1, keepdims=True).clip(min=1e-8)
    ctx_n = (ctx - means) / stds

    n_per_ch = math.ceil(pred_len / patch_size)
    eos = model.get_eos_token()

    sample = {
        "type":       "chronos",
        "pre_prompt": f"You are given {CONTEXT_LEN} timesteps across {D} channels:",
        "post_prompt": (
            f". Predict the next {pred_len} timesteps "
            f"as {n_per_ch} patch{'es' if n_per_ch != 1 else ''} per channel"
        ),
        "time_series":      [ctx_n[c].tolist() for c in range(D)],
        "time_series_text": [
            f"Channel {c}, mean {means[c, 0]:.4f} std {stds[c, 0]:.4f}. <ts>"
            for c in range(D)
        ],
        "ts_already_normalized": True,
        "n_channels":  D,
        "_n_channels": D,
        "answer":      eos,
    }

    max_new = D * n_per_ch * 2 + 64
    preds = model.forecast_phase2(
        [sample], pred_len=pred_len, max_new_tokens=max_new, logit_boost=logit_boost,
    )

    if preds[0] is None:
        return np.broadcast_to(ctx[:, -1:], (D, pred_len)).copy()

    pred_norm = preds[0].cpu().float().numpy()   # (D, pred_len)
    return pred_norm * stds + means              # denormalize to original scale


def eval_one_dataset(
    model: OpenTSLMSPFixPos2D,
    ds_name: str,
    term: str,
    args: argparse.Namespace,
    ds_settings: dict,
) -> None:
    ds = Dataset(
        name=ds_name,
        term=term,
        to_univariate=False,
        prediction_length=ds_settings["prediction_length"],
        test_length=ds_settings["test_length"],
        val_length=ds_settings.get("val_length") or 0,
    )

    pred_len    = ds.prediction_length
    windows     = list(ds.test_data.input)
    n_instances = len(windows)
    season_len  = get_seasonality(ds.freq)

    print(
        f"  {ds_name}/{term}  pred={pred_len}  ctx={CONTEXT_LEN}  "
        f"freq={ds.freq}  seasonality={season_len}  n_windows={n_instances}"
    )

    if n_instances == 0:
        print(f"  [skip] {ds_name}/{term}: no test windows")
        return

    all_preds = []
    for d in tqdm(windows, desc=f"  {ds_name}/{term}", leave=False):
        target = np.asarray(d["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target[np.newaxis, :]   # (1, T) — univariate

        pred = predict_window(
            model, target, pred_len, args.patch_size, args.logit_boost
        )   # (D, pred_len)
        all_preds.append(pred)

    fc           = np.stack(all_preds, axis=0)      # (N, D, pred_len)
    fc_quantiles = np.stack([fc] * 9, axis=1)       # (N, 9, D, pred_len)

    ds_config = f"{ds_name}/{term}"
    metadata  = save_window_predictions(
        dataset         = ds,
        fc_quantiles    = fc_quantiles,
        ds_config       = ds_config,
        output_base_dir = args.out_dir,
        seasonality     = season_len,
        model_hyperparams = {
            "patch_size":  args.patch_size,
            "lora_r":      args.lora_r,
            "lora_alpha":  args.lora_alpha,
            "logit_boost": args.logit_boost,
            "llm_id":      args.llm_id,
            "context_len": CONTEXT_LEN,
        },
        quantile_levels = QUANTILE_LEVELS,
    )
    print(f"  saved → {metadata}")


def load_model(args: argparse.Namespace) -> OpenTSLMSPFixPos2D:
    cfg = SimpleNamespace(
        is_pretraining      = True,
        use_only_time_pe    = False,
        use_only_channel_pe = False,
        use_freq_embed      = False,
        patch_size          = args.patch_size,
        pe_type             = args.pe_type,
        enc_dec_type        = args.enc_dec_type,
        mlp_dec_hidden_dim  = args.mlp_dec_hidden_dim,
        mlp_dec_ff_dim      = args.mlp_dec_ff_dim,
        mlp_enc_hidden_dim  = args.mlp_enc_hidden_dim,
        mlp_enc_ff_dim      = args.mlp_enc_ff_dim,
    )
    model = OpenTSLMSPFixPos2D(cfg, llm_id=args.llm_id, device=args.device)

    ckpt_keys = torch.load(args.ckpt, map_location="cpu", weights_only=False).keys()
    if "llm_state" not in ckpt_keys and args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print(f"LoRA enabled: r={args.lora_r}  alpha={args.lora_alpha}")

    model.load_from_file(args.ckpt)
    model.eval()
    print(f"Loaded: {args.ckpt}")
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="TIME benchmark zero-shot evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt",     required=True,
                    help="Path to best_model.pt checkpoint.")
    ap.add_argument("--llm_id",   default="google/gemma-3-270m")
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--patch_size",         type=int,   default=32)
    ap.add_argument("--lora_r",             type=int,   default=32)
    ap.add_argument("--lora_alpha",         type=int,   default=64)
    ap.add_argument("--mlp_dec_hidden_dim", type=int,   default=None)
    ap.add_argument("--mlp_dec_ff_dim",     type=int,   default=None)
    ap.add_argument("--mlp_enc_hidden_dim", type=int,   default=None)
    ap.add_argument("--mlp_enc_ff_dim",     type=int,   default=None)
    ap.add_argument("--enc_dec_type",       default="transformer")
    ap.add_argument("--pe_type",            default="sincos")
    ap.add_argument("--logit_boost",        type=float, default=50.0)
    ap.add_argument("--term",   default="short",
                    choices=["short", "medium", "long", "all"],
                    help="Forecast horizon term. 'all' runs short+medium+long.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of dataset names. Default: all datasets in datasets.yaml.")
    ap.add_argument("--out_dir",  default="./eval_results/time_benchmark")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args)

    cfg      = load_dataset_config()
    ds_names = args.datasets or list(cfg.get("datasets", {}).keys())
    terms    = ["short", "medium", "long"] if args.term == "all" else [args.term]

    print(f"\nTIME benchmark  term(s)={terms}  n_datasets={len(ds_names)}\n")

    for ds_name in ds_names:
        for term in terms:
            try:
                settings = get_dataset_settings(ds_name, term, cfg)
            except Exception:
                continue
            try:
                eval_one_dataset(model, ds_name, term, args, settings)
            except Exception as exc:
                print(f"  [error] {ds_name}/{term}: {exc}")

    print(f"\nAll results → {args.out_dir}")


if __name__ == "__main__":
    main()
