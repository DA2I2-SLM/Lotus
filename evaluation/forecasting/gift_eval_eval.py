#!/usr/bin/env python3
"""
Zero-shot GIFT-Eval evaluation of OpenTSLMSPFixPos2D.

Protocol:
  - gift_eval.data.Dataset for data loading (to_univariate=True)
  - GluonTS evaluate_model for metrics
  - All 55 dataset variants × 3 terms (short/medium/long) = 165 configs
  - Results saved to <out_dir>/results_task<N>.csv (merged into all_results.csv)

Usage:
    python evaluation/forecasting/gift_eval_eval.py \
        --ckpt results/.../best_model.pt \
        --gift_eval_data /path/to/gift_eval_data \
        --task_id 0 --n_tasks 5
"""

import argparse
import csv
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

# ── Constants ──────────────────────────────────────────────────────────────────
PATCH_SIZE      = 32
MAX_CTX_STEPS   = 512
NORM_STD_FLOOR  = 1e-3
NUM_SAMPLES     = 20
INNER_BS        = 16
OUTER_BS        = 256
MODEL_NAME      = "OpenTSLMSPFixPos2D"

METRIC_NAMES = [
    "MSE[mean]", "MSE[0.5]", "MAE[mean]", "MAE[0.5]",
    "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS",
    "RMSE[mean]", "NRMSE[mean]", "ND[mean]",
    "mean_weighted_sum_quantile_loss",
]
CSV_HEADER = ["dataset", "model"] + METRIC_NAMES + ["domain", "num_variates"]


# ── Build config list from disk ────────────────────────────────────────────────
def enumerate_configs(gift_eval_data: str):
    """Enumerate all (ds_name, term) pairs from datasets on disk."""
    base = Path(gift_eval_data)
    names = []
    for p in sorted(base.iterdir()):
        if not p.is_dir() or p.name.startswith("."): continue
        subdirs = sorted(s for s in p.iterdir() if s.is_dir())
        if subdirs:
            for freq_dir in subdirs:
                names.append(f"{p.name}/{freq_dir.name}")
        else:
            names.append(p.name)

    configs = []
    for name in names:
        for term in ["short", "medium", "long"]:
            configs.append((name, term))
    return configs


# ── Model setup ────────────────────────────────────────────────────────────────
def build_model_args():
    return SimpleNamespace(
        is_pretraining=True,
        patch_size=PATCH_SIZE,
        enc_dec_type="transformer",
        trans_enc_dim=192,
        trans_enc_num_layers=6,
        trans_enc_num_heads=8,
        trans_enc_ff_dim=896,
        trans_dec_dim=192,
        trans_dec_num_layers=4,
        trans_dec_num_heads=8,
        trans_dec_ff_dim=896,
        use_freq_embed=True,
        llm_mv_position_ids=True,
        llm_dtype="auto",
        gradient_checkpointing=False,
        mlp_enc_hidden_dim=None,
        mlp_enc_ff_dim=None,
        mlp_dec_hidden_dim=None,
        mlp_dec_ff_dim=None,
        use_only_time_pe=False,
        use_only_channel_pe=False,
    )


def _make_uv_item(ctx_1d, eos_token, pre_norm_mean=0.0, pre_norm_std=1.0):
    T = len(ctx_1d)
    prompt = (
        f"This is a univariate time series with 1 channel "
        f"and {T} time steps. "
        f"It has mean {pre_norm_mean:.4f} and std {pre_norm_std:.4f}. <ts>"
    )
    return {
        "type":                   "chronos",
        "source":                 "benchmark",
        "answer":                 eos_token,
        "post_prompt":            "",
        "pre_prompt":             "",
        "time_series":            [np.asarray(ctx_1d, dtype=np.float32)],
        "time_series_text":       [prompt],
        "ts_already_normalized":  True,
        "time_series_norm_stats": [(pre_norm_mean, pre_norm_std)],
        "norm_mean":              float(pre_norm_mean),
        "norm_std":               float(pre_norm_std),
    }


@torch.no_grad()
def _predict_next_patch(model, batch_items, device):
    inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
        batch_items, return_targets=True
    )
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]
    ts_hidden = last_hidden[ts_mask]
    ts_hidden = model.decoder_projector(ts_hidden.float())
    ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

    B = len(batch_items)
    per_item_counts = [int(ts_mask[b].sum()) for b in range(B)]
    parts = list(torch.split(ts_hidden, per_item_counts))

    preds = []
    for part in parts:
        dec_out = model.decoder(part.unsqueeze(0))
        preds.append(dec_out[0, -1, :])
    return torch.stack(preds, dim=0)


def forecast_batch(model, eos_token, contexts, n_patches, device):
    """contexts: float32 [B, ctx_len]. Returns: float32 [B, n_patches*PATCH_SIZE]."""
    B = len(contexts)
    # Use nan-safe stats so datasets with missing values don't produce all-NaN forecasts.
    means = np.nanmean(contexts, axis=-1, keepdims=True)
    stds  = np.maximum(np.nanstd(contexts, axis=-1, keepdims=True), NORM_STD_FLOOR)
    ctx_norm = (contexts - means) / stds
    ctx_norm = np.nan_to_num(ctx_norm, nan=0.0)   # fill missing positions with 0
    per_mean = means[:, 0].tolist()
    per_std  = stds[:, 0].tolist()

    # Clip normalized context to prevent bfloat16 overflow in attention.
    ctx_norm = np.clip(ctx_norm, -10.0, 10.0)
    curr_ctx = [ctx_norm[b].tolist() for b in range(B)]
    all_patches = []

    for _ in range(n_patches):
        batch_preds = []
        for s in range(0, B, INNER_BS):
            e = min(s + INNER_BS, B)
            items = [
                _make_uv_item(curr_ctx[b][-MAX_CTX_STEPS:], eos_token,
                              per_mean[b], per_std[b])
                for b in range(s, e)
            ]
            pred = _predict_next_patch(model, items, device)
            batch_preds.append(pred.cpu().float().numpy())
        next_patch = np.concatenate(batch_preds, axis=0)   # [B, PATCH_SIZE]
        # Prevent NaN from cascading into subsequent autoregressive context.
        next_patch = np.nan_to_num(next_patch, nan=0.0, posinf=10.0, neginf=-10.0)
        all_patches.append(next_patch)
        for b in range(B):
            curr_ctx[b].extend(all_patches[-1][b].tolist())

    pred_norm = np.concatenate(all_patches, axis=-1)       # [B, n_patches*PATCH_SIZE]
    return (pred_norm * stds + means).astype(np.float32)


# ── GluonTS-compatible predictor ───────────────────────────────────────────────
class OurModelPredictor:
    """
    `predict` receives an iterable of input dicts (from test_data.input).
    Each dict has keys: item_id, start, freq, target.
    target shape: [T] (univariate, after to_univariate=True).
    """
    def __init__(self, model, eos_token, prediction_length, device):
        self.model = model
        self.eos_token = eos_token
        self.prediction_length = prediction_length
        self.device = device
        self.n_patches = math.ceil(prediction_length / PATCH_SIZE)

    def predict(self, dataset) -> Iterator:
        from gluonts.model.forecast import SampleForecast

        entries = list(dataset)
        amp_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        for start in range(0, len(entries), OUTER_BS):
            batch = entries[start : start + OUTER_BS]
            contexts = []
            for entry in batch:
                tgt = np.asarray(entry["target"], dtype=np.float32)
                if tgt.ndim > 1:
                    tgt = tgt.mean(axis=0)
                ctx_len = min(MAX_CTX_STEPS, len(tgt))
                ctx = tgt[-ctx_len:]
                if len(ctx) < MAX_CTX_STEPS:
                    ctx = np.pad(ctx, (MAX_CTX_STEPS - len(ctx), 0))
                contexts.append(ctx)

            contexts = np.stack(contexts, axis=0)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(self.device == "cuda")):
                preds = forecast_batch(
                    self.model, self.eos_token, contexts,
                    self.n_patches, self.device
                )

            preds = preds[:, :self.prediction_length]

            for i, entry in enumerate(batch):
                tgt = np.asarray(entry["target"], dtype=np.float32)
                fc_start = entry["start"] + len(tgt)
                samples = np.tile(preds[i], (NUM_SAMPLES, 1))
                yield SampleForecast(
                    samples=samples,
                    start_date=fc_start,
                    item_id=entry.get("item_id"),
                )


# ── Evaluation ─────────────────────────────────────────────────────────────────
def get_metrics():
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type="0.5"),
        MAE(forecast_type="mean"),
        MAE(forecast_type="0.5"),
        MASE(forecast_type="0.5"),
        MAPE(forecast_type="0.5"),
        SMAPE(forecast_type="0.5"),
        MSIS(alpha=0.05),
        RMSE(forecast_type="mean"),
        NRMSE(forecast_type="mean"),
        ND(forecast_type="mean"),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def domain_of(ds_name: str) -> str:
    dn = ds_name.lower()
    if "m4" in dn:                                          return "m4"
    if "electricity" in dn:                                 return "energy"
    if "solar" in dn:                                       return "energy"
    if "traffic" in dn or "loop" in dn or "sz_taxi" in dn: return "transport"
    if any(k in dn for k in ("weather", "jena", "kdd", "temperature")): return "weather"
    if any(k in dn for k in ("ett", "m_dense")):            return "etdataset"
    if any(k in dn for k in ("bizitobs", "bitbrains")):     return "cloud"
    if any(k in dn for k in ("hospital", "covid", "us_births")): return "health"
    return "other"


def evaluate_one(model, eos_token, ds_name, term, device):
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    try:
        # Load without to_univariate first to check target_dim.
        # MultivariateToUnivariate applied to an already-1D dataset collapses
        # targets to scalars and breaks gluonts split — only use it for MV data.
        _ds = Dataset(name=ds_name, term=term, to_univariate=False)
        use_uv = _ds.target_dim > 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=use_uv) if use_uv else _ds
    except Exception as e:
        print(f"  [SKIP] {ds_name}/{term}: {e}", flush=True)
        return None

    pred_len   = dataset.prediction_length
    season_len = get_seasonality(dataset.freq)

    predictor = OurModelPredictor(
        model=model, eos_token=eos_token,
        prediction_length=pred_len, device=device,
    )

    try:
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=get_metrics(),
            batch_size=None,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=True,
            seasonality=season_len,
        )
    except Exception as e:
        print(f"  [ERROR] {ds_name}/{term}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None

    num_variates = dataset.target_dim if hasattr(dataset, "target_dim") else 1
    ds_key = f"{ds_name}/{term}"

    row = [ds_key, MODEL_NAME]
    for name in METRIC_NAMES:
        col = res.get(name)
        val = float(col.iloc[0]) if col is not None and len(col) > 0 else ""
        row.append(val)
    row += [domain_of(ds_name), num_variates]

    mse_str  = f"{float(res['MSE[mean]'].iloc[0]):.4f}"  if res.get('MSE[mean]')  is not None else "N/A"
    mae_str  = f"{float(res['MAE[mean]'].iloc[0]):.4f}"  if res.get('MAE[mean]')  is not None else "N/A"
    mase_str = f"{float(res['MASE[0.5]'].iloc[0]):.4f}"  if res.get('MASE[0.5]') is not None else "N/A"
    print(f"  {ds_key}: MSE={mse_str}  MAE={mae_str}  MASE={mase_str}", flush=True)
    return row


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--llm_id",         default="/vast/s225250685/models/gemma-3-270m")
    ap.add_argument("--gift_eval_data", required=True)
    ap.add_argument("--task_id",  type=int, default=0)
    ap.add_argument("--n_tasks",  type=int, default=1)
    ap.add_argument("--out_dir",  default="results/OpenTSLMSPFixPos2D")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.environ["GIFT_EVAL"] = args.gift_eval_data

    all_configs = enumerate_configs(args.gift_eval_data)
    my_configs  = [c for i, c in enumerate(all_configs) if i % args.n_tasks == args.task_id]

    print(f"Task {args.task_id}/{args.n_tasks}: {len(my_configs)} configs", flush=True)
    print(f"Device: {args.device}", flush=True)

    print(f"Loading model from {args.llm_id} ...", flush=True)
    model = OpenTSLMSPFixPos2D(
        args=build_model_args(),
        llm_id=args.llm_id,
        device=args.device,
    )
    model.load_from_file(args.ckpt)
    model.eval().to(args.device)
    eos_token = model.get_eos_token()
    print("Model loaded.", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"results_task{args.task_id}.csv")

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for idx, (ds_name, term) in enumerate(my_configs):
            print(f"\n[{idx+1}/{len(my_configs)}] {ds_name}/{term}", flush=True)
            row = evaluate_one(model, eos_token, ds_name, term, args.device)
            if row is not None:
                writer.writerow(row)
                f.flush()

    print(f"\nDone. Results → {out_file}", flush=True)


if __name__ == "__main__":
    main()
