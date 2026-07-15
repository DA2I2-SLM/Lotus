#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""Phase-1 long-horizon forecasting evaluation (foundational MSE path).

Evaluates an OpenTSLMSPFixPos2D phase-1 checkpoint on the standard long-
horizon benchmarks (ETTh1/h2, ETTm1/m2, Weather, Electricity, Traffic)
using only the foundational MSE path:

  encoder → projector → layer_norm → decoder_projector → layer_norm → decoder

No LLM generation is used; the frozen LLM backbone is not run at all.

Protocol
--------
* Channel-independent evaluation: each column is treated as a univariate
  series, consistent with how phase-1 was trained on univariate Chronos data.
* Per-channel RevIN normalisation using the context window's mean / std.
* Autoregressive patch-by-patch prediction (outer re-encode loop) for
  pred_len > patch_size.  The context window slides forward one patch at a
  time so the model always sees exactly history_len timesteps — within its
  training distribution.
* Non-overlapping test windows (stride = pred_len), matching phase-2 eval.
* Metrics: MSE and MAE in the normalised scale (industry standard for these
  benchmarks — RevIN eliminates scale so cross-dataset comparison is fair).

If pred_len is not divisible by patch_size the last forecast patch is
trimmed to fit; a warning is printed.

Results are written to --out_dir as:
  results_<timestamp>.csv   — one row per (dataset, pred_len)
  results_<timestamp>.json  — same, plus meta (args, n_windows)

Usage
-----
  python eval_phase1_forecasting.py \\
    --ckpt /path/to/best_model.pt \\
    --llm_id google/gemma-3-270m \\
    --patch_size 96 \\
    --enc_dec_type transformer \\
    --data_root /scratch/.../dataset \\
    --datasets etth1 etth2 weather \\
    --pred_len 96 192 336 720 \\
    --out_dir ./eval_results/phase1_forecasting

  # all datasets at once:
  python eval_phase1_forecasting.py \\
    --ckpt ...pt --llm_id ... --patch_size 96 \\
    --data_root /scratch/.../dataset \\
    --all_datasets --pred_len 96 192 336 720
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT  = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D  # noqa: E402
from opentslm.time_series_datasets.util import (  # noqa: E402
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ── Dataset configs ──────────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "etth1":       {"file": "ETTh1.csv",       "split": (0.6, 0.2, 0.2)},
    "etth2":       {"file": "ETTh2.csv",       "split": (0.6, 0.2, 0.2)},
    "ettm1":       {"file": "ETTm1.csv",       "split": (0.6, 0.2, 0.2)},
    "ettm2":       {"file": "ETTm2.csv",       "split": (0.6, 0.2, 0.2)},
    "weather":     {"file": "weather.csv",     "split": (0.7, 0.1, 0.2)},
    "electricity": {"file": "electricity.csv", "split": (0.7, 0.1, 0.2)},
    "traffic":     {"file": "traffic.csv",     "split": (0.7, 0.1, 0.2)},
}

# Context lengths mirror phase-2 downstream eval and common long-horizon papers.
_CONTEXT_BY_PRED: Dict[int, int] = {96: 512, 192: 512, 336: 672, 720: 720}


# ── Dataset ──────────────────────────────────────────────────────────────────────

class Phase1ForecastDataset(Dataset):
    """Univariate sliding-window dataset for phase-1 forecasting evaluation.

    Each sample is one (channel, window) pair normalised independently via
    per-channel RevIN (mean/std from the context window only, never the future).
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "test",
        patch_size: int = 96,
        pred_len: int = 96,
        history_len: Optional[int] = None,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        eos_token: str = "",
        ds_name: str = "",
    ):
        if history_len is None:
            history_len = _CONTEXT_BY_PRED.get(pred_len, 512)
        # Ensure history_len is a multiple of patch_size (round down, min 1 patch).
        history_len = max(patch_size, (history_len // patch_size) * patch_size)

        self.history_len = history_len
        self.pred_len    = pred_len
        self.patch_size  = patch_size

        df = pd.read_csv(csv_path)
        df = df.select_dtypes(include=[float, int])

        N  = len(df)
        tr = int(N * split_ratios[0])
        va = int(N * (split_ratios[0] + split_ratios[1]))
        s, e = {"train": (0, tr), "val": (tr, va), "test": (va, N)}.get(mode, (va, N))
        data      = torch.tensor(
            df.iloc[s:e].reset_index(drop=True).values, dtype=torch.float32
        )
        col_names = list(df.columns)
        T, C      = data.shape

        window_len = history_len + pred_len
        stride     = pred_len   # non-overlapping windows on val/test

        self.samples: List[dict] = []
        for start in range(0, T - window_len + 1, stride):
            for c in range(C):
                hist = data[start : start + history_len, c]
                fut  = data[start + history_len : start + window_len, c]

                mean = hist.mean().item()
                std  = max(hist.std().item(), 1e-8)

                self.samples.append({
                    "type":                   "chronos",
                    "pre_prompt":             "",
                    "time_series":            [hist.tolist()],
                    "time_series_text":       [
                        f"This is the time series ({col_names[c]}), "
                        f"mean {mean:.4f} std {std:.4f}. <ts>"
                    ],
                    "post_prompt":            eos_token,
                    "answer":                 eos_token,
                    "norm_mean":              mean,
                    "norm_std":               std,
                    "time_series_norm_stats": [(mean, std)],
                    "ts_already_normalized":  True,
                    "n_patches":              history_len // patch_size,
                    # Normalised ground truth: [pred_len]; stored for metric computation.
                    "ts_future":              ((fut - mean) / std).tolist(),
                })

        print(
            f"[Phase1ForecastDataset] {ds_name:12s} {mode}  "
            f"pred={pred_len:4d}  hist={history_len:4d}  C={C:3d}  "
            f"stride={stride:4d}  n_samples={len(self.samples):6d}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ── Forward pass (foundational MSE path only) ────────────────────────────────────

@torch.no_grad()
def _predict_next_patch(
    model: OpenTSLMSPFixPos2D,
    batch: List[dict],
    patch_size: int,
) -> torch.Tensor:
    """Encode a batch of univariate contexts and return the next-patch prediction.

    Runs:  encoder → projector → layer_norm → decoder_projector → layer_norm
           → decoder → last-position output

    No LLM forward pass is performed.

    Returns: [N, patch_size] float32, normalised.
    """
    collated = extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=patch_size
    )
    if not collated:
        raise RuntimeError("Collate produced an empty batch.")

    model.pad_and_apply_batch(collated, return_targets=True)

    if model._last_ts_enc is None:
        raise RuntimeError(
            "_last_ts_enc was not populated after pad_and_apply_batch. "
            "Check that the batch contains valid time_series entries."
        )

    hidden = model.build_foundational_history_hidden()  # [N, T_patches, enc_dim]
    pred   = model.decoder(hidden)                      # [N, T_patches, patch_size]
    return pred[:, -1, :].float()                       # [N, patch_size]


@torch.no_grad()
def forecast_autoregressive(
    model: OpenTSLMSPFixPos2D,
    initial_batch: List[dict],
    n_patches: int,
    patch_size: int,
) -> torch.Tensor:
    """Forecast n_patches future patches autoregressively for a batch of samples.

    At each step the predicted patch is appended to the context and the oldest
    patch is dropped, keeping the context length constant at history_len.
    This ensures the model always receives a context within its training
    distribution.

    Returns: [N, n_patches * patch_size] float32, normalised.
    """
    N = len(initial_batch)
    history_len = len(initial_batch[0]["time_series"][0])

    # Working histories in normalised space: list of 1-D numpy arrays.
    histories = [
        np.array(s["time_series"][0], dtype=np.float32) for s in initial_batch
    ]
    norms = [(s["norm_mean"], s["norm_std"]) for s in initial_batch]
    eos   = initial_batch[0]["post_prompt"]

    all_patches: List[torch.Tensor] = []  # list of [N, patch_size] tensors

    for step in range(n_patches):
        step_batch = [
            {
                "type":                   "chronos",
                "pre_prompt":             "",
                "time_series":            [histories[i].tolist()],
                "time_series_text":       [
                    f"mean {norms[i][0]:.4f} std {norms[i][1]:.4f}. <ts>"
                ],
                "post_prompt":            eos,
                "answer":                 eos,
                "norm_mean":              norms[i][0],
                "norm_std":               norms[i][1],
                "time_series_norm_stats": [norms[i]],
                "ts_already_normalized":  True,
                "n_patches":              len(histories[i]) // patch_size,
            }
            for i in range(N)
        ]

        next_patches = _predict_next_patch(model, step_batch, patch_size)
        # [N, patch_size] on device

        all_patches.append(next_patches.cpu())

        # Slide context window: drop oldest patch, append predicted patch.
        for i in range(N):
            p = next_patches[i].cpu().numpy()
            histories[i] = np.concatenate([histories[i], p])[-history_len:]

    return torch.cat(all_patches, dim=1)  # [N, n_patches * patch_size]


# ── Metric helpers ───────────────────────────────────────────────────────────────

def _mse_mae(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred   = pred.float()
    target = target.float().to(pred.device)
    return (
        float(F.mse_loss(pred, target).item()),
        float(F.l1_loss(pred, target).item()),
    )


# ── Per-dataset evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(
    model: OpenTSLMSPFixPos2D,
    csv_path: str,
    ds_name: str,
    pred_len: int,
    patch_size: int,
    split_ratios: Tuple[float, float, float],
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    eos = model.get_eos_token()

    n_patches = math.ceil(pred_len / patch_size)
    if pred_len % patch_size != 0:
        print(
            f"  [warn] pred_len={pred_len} not divisible by patch_size={patch_size}; "
            f"predicting {n_patches} patches and trimming to {pred_len} timesteps."
        )

    ds = Phase1ForecastDataset(
        csv_path     = csv_path,
        mode         = "test",
        patch_size   = patch_size,
        pred_len     = pred_len,
        split_ratios = split_ratios,
        eos_token    = eos,
        ds_name      = ds_name,
    )

    if len(ds) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "n_windows": 0}

    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = list,  # keep as plain list; collation is done inside forecast
    )

    total_mse = 0.0
    total_mae = 0.0
    n_total   = 0

    for raw_batch in tqdm(loader, desc=f"  {ds_name} H={pred_len}", leave=False):
        preds = forecast_autoregressive(
            model, raw_batch, n_patches=n_patches, patch_size=patch_size
        )  # [N, n_patches * patch_size]

        targets = torch.tensor(
            np.stack([s["ts_future"] for s in raw_batch]), dtype=torch.float32
        )  # [N, pred_len]

        # Trim preds to exact pred_len in case of rounding.
        preds = preds[:, :pred_len]

        mse, mae = _mse_mae(preds, targets)
        N = len(raw_batch)
        total_mse += mse * N
        total_mae += mae * N
        n_total   += N

    return {
        "mse":       total_mse / n_total if n_total > 0 else float("nan"),
        "mae":       total_mae / n_total if n_total > 0 else float("nan"),
        "n_windows": n_total,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required
    parser.add_argument("--ckpt",      required=True,
                        help="Path to phase-1 checkpoint (.pt).")
    parser.add_argument("--llm_id",    default="google/gemma-3-270m",
                        help="LLM model ID or local snapshot path.")
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing dataset CSV files.")

    # Dataset selection
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument(
        "--all_datasets", action="store_true",
        help="Evaluate on all seven standard benchmarks."
    )
    ds_group.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_CONFIGS),
        help="Subset of datasets to evaluate."
    )

    # Evaluation
    parser.add_argument("--pred_len",   nargs="+", type=int,
                        default=[96, 192, 336, 720],
                        help="Prediction horizons.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Samples per forward pass (each sample = one channel window).")

    # Model architecture — must match the training run exactly.
    parser.add_argument("--patch_size",         type=int, default=96)
    parser.add_argument("--pe_type",            default="sincos")
    parser.add_argument("--enc_dec_type",       default="transformer",
                        choices=["transformer", "mlp"])
    parser.add_argument("--mlp_enc_hidden_dim", type=int, default=None)
    parser.add_argument("--mlp_enc_ff_dim",     type=int, default=None)
    parser.add_argument("--mlp_dec_hidden_dim", type=int, default=None)
    parser.add_argument("--mlp_dec_ff_dim",     type=int, default=None)

    parser.add_argument("--device",  default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out_dir", default="./eval_results/phase1_forecasting")
    return parser.parse_args()


def _make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    ns = SimpleNamespace(
        is_pretraining       = True,
        patch_size           = args.patch_size,
        pe_type              = args.pe_type,
        use_only_time_pe     = False,
        use_only_channel_pe  = False,
        use_freq_embed       = False,
        enc_dec_type         = args.enc_dec_type,
    )
    for attr in (
        "mlp_enc_hidden_dim", "mlp_enc_ff_dim",
        "mlp_dec_hidden_dim", "mlp_dec_ff_dim",
    ):
        v = getattr(args, attr)
        if v is not None:
            setattr(ns, attr, v)
    return ns


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[eval] device={device}")
    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] patch_size={args.patch_size}  enc_dec_type={args.enc_dec_type}")
    print(f"[eval] pred_len={args.pred_len}")

    model = OpenTSLMSPFixPos2D(
        args   = _make_model_args(args),
        llm_id = args.llm_id,
        device = device,
    ).to(device)
    model.load_from_file(args.ckpt)
    model.eval()

    datasets = list(DATASET_CONFIGS) if args.all_datasets else args.datasets

    rows: List[dict] = []
    all_results: Dict[str, dict] = {}

    for ds_name in datasets:
        cfg      = DATASET_CONFIGS[ds_name]
        csv_path = os.path.join(args.data_root, cfg["file"])
        if not os.path.exists(csv_path):
            print(f"[skip] {ds_name}: {csv_path} not found")
            continue

        all_results[ds_name] = {}
        for pred_len in args.pred_len:
            print(f"\n[eval] dataset={ds_name}  pred_len={pred_len}")
            res = evaluate_dataset(
                model        = model,
                csv_path     = csv_path,
                ds_name      = ds_name,
                pred_len     = pred_len,
                patch_size   = args.patch_size,
                split_ratios = tuple(cfg["split"]),
                batch_size   = args.batch_size,
            )
            mse = res["mse"]
            mae = res["mae"]
            nw  = res["n_windows"]
            print(f"  MSE={mse:.4f}  MAE={mae:.4f}  n_windows={nw}")

            all_results[ds_name][pred_len] = res
            rows.append({
                "dataset":   ds_name,
                "pred_len":  pred_len,
                "mse":       round(mse, 4),
                "mae":       round(mae, 4),
                "n_windows": nw,
            })

    # ── Summary table ─────────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"{'Dataset':12s}  {'H':>5s}  {'MSE':>8s}  {'MAE':>8s}  {'N':>7s}")
    print(sep)
    for row in rows:
        mse_s = f"{row['mse']:.4f}" if not math.isnan(row["mse"]) else "    nan"
        mae_s = f"{row['mae']:.4f}" if not math.isnan(row["mae"]) else "    nan"
        print(
            f"{row['dataset']:12s}  {row['pred_len']:>5d}  "
            f"{mse_s:>8s}  {mae_s:>8s}  {row['n_windows']:>7d}"
        )
    print(sep)

    # ── Save ──────────────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out  = os.path.join(args.out_dir, f"results_{ts}.csv")
    json_out = os.path.join(args.out_dir, f"results_{ts}.json")

    pd.DataFrame(rows).to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": all_results, "rows": rows}, fh, indent=2)

    print(f"\n[eval] saved → {csv_out}")
    print(f"[eval] saved → {json_out}")


if __name__ == "__main__":
    main()
