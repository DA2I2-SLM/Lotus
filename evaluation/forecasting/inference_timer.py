#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""Zero-shot Timer forecasting inference on ETTh1-style windows.

This script evaluates the official Timer Hugging Face checkpoints with the
same ETTh1 sliding-window setup used elsewhere in this repo, so the metrics are
easy to compare against OpenTSLM runs.

Official Timer quickstart:
  AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)
  output = model.generate(seqs, max_new_tokens=prediction_length)
"""

from __future__ import annotations

import argparse
import json
import math
import getpass
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM


@dataclass
class TimerBatch:
    history: torch.Tensor          # [B, L_hist]
    future: torch.Tensor           # [B, pred_len]
    series_name: List[str]


class ETTh1TimerEvalDataset(Dataset):
    """ETTh1 sliding-window dataset in normalized forecasting space."""

    TARGET_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    def __init__(
        self,
        csv_path: str,
        mode: str = "test",
        pred_len: int = 96,
        history_len: int = 480,
        stride: int = 1,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    ) -> None:
        self.pred_len = int(pred_len)
        self.history_len = int(history_len)

        df = pd.read_csv(csv_path)
        n_rows = len(df)
        train_end = int(n_rows * split_ratios[0])
        val_end = int(n_rows * (split_ratios[0] + split_ratios[1]))

        if mode == "train":
            df_split = df.iloc[:train_end]
        elif mode == "val":
            df_split = df.iloc[train_end:val_end]
        else:
            df_split = df.iloc[val_end:]

        window_len = self.history_len + self.pred_len
        self.samples: List[Tuple[torch.Tensor, str]] = []

        for col in self.TARGET_COLS:
            if col not in df_split.columns:
                continue
            series = torch.tensor(df_split[col].values, dtype=torch.float32)
            total_len = len(series)
            for start in range(0, total_len - window_len + 1, stride):
                self.samples.append((series[start : start + window_len].clone(), col))

        print(
            f"[ETTh1TimerEvalDataset] mode={mode} split_rows={len(df_split)} | "
            f"{len(self.samples)} windows "
            f"(history={self.history_len} pred={self.pred_len} stride={stride})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        window, col_name = self.samples[idx]
        history = window[: self.history_len]
        future = window[self.history_len :]

        mean = history.mean()
        std = history.std().clamp(min=1e-8)
        history_norm = (history - mean) / std
        future_norm = (future - mean) / std

        return {
            "history": history_norm,
            "future": future_norm,
            "series_name": col_name,
        }


def collate_timer(batch: List[Dict[str, torch.Tensor | str]]) -> TimerBatch:
    histories = torch.stack([item["history"] for item in batch], dim=0)
    futures = torch.stack([item["future"] for item in batch], dim=0)
    names = [str(item["series_name"]) for item in batch]
    return TimerBatch(history=histories, future=futures, series_name=names)


def _resolve_model_context(model, fallback: int = 2880) -> int:
    token_len = int(getattr(model.config, "input_token_len", 96))
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is None:
        return fallback
    return int(max_positions) * token_len


def _prepare_history_for_timer(
    history: torch.Tensor,
    *,
    token_len: int,
    max_context: int,
) -> torch.Tensor:
    """Crop to the most recent context and left-pad to a token boundary."""
    if history.dim() != 2:
        raise ValueError(f"Expected [B, L], got {tuple(history.shape)}")

    if history.size(1) > max_context:
        history = history[:, -max_context:]

    remainder = history.size(1) % token_len
    if remainder == 0:
        return history

    pad_len = token_len - remainder
    left_pad = history[:, :1].expand(-1, pad_len)
    return torch.cat([left_pad, history], dim=1)


def _extract_generated_forecast(output, pred_len: int) -> torch.Tensor:
    if torch.is_tensor(output):
        seq = output
    elif hasattr(output, "sequences"):
        seq = output.sequences
    else:
        raise TypeError(f"Unsupported Timer generate output type: {type(output)!r}")

    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    if seq.size(-1) < pred_len:
        raise RuntimeError(
            f"Timer output length {seq.size(-1)} is shorter than pred_len={pred_len}"
        )
    return seq[:, -pred_len:]


def _save_plot(
    out_dir: str,
    sample_idx: int,
    history: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
        hist = history.detach().cpu().numpy()
        pr = pred.detach().cpu().numpy()
        gt = target.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(hist)), hist, label="history", color="steelblue")
        ax.plot(
            range(len(hist), len(hist) + len(gt)),
            gt,
            label="target future",
            color="seagreen",
            linewidth=2,
        )
        ax.plot(
            range(len(hist), len(hist) + len(pr)),
            pr,
            label="timer forecast",
            color="darkorange",
            linewidth=2,
        )
        ax.set_title(f"Timer forecast | sample={sample_idx} | series={name}")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"forecast_{sample_idx:06d}_{name}.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting should not kill eval
        print(f"[plot warning] Could not save plot for sample {sample_idx}: {type(exc).__name__}: {exc}")


@torch.no_grad()
def run_timer_forecast(args) -> Dict[str, float | int]:
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    print(f"[timer] loading model_id={args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    token_len = int(getattr(model.config, "input_token_len", 96))
    max_context = args.max_context or _resolve_model_context(model)
    print(
        f"[timer] token_len={token_len} "
        f"max_context={max_context} "
        f"timer_revin={args.timer_revin}"
    )

    dataset = ETTh1TimerEvalDataset(
        csv_path=args.etth1_csv,
        mode=args.eval_split,
        pred_len=args.pred_len,
        history_len=args.history_len,
        stride=args.etth1_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_timer,
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    total_mse = 0.0
    total_mae = 0.0
    total_n = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="timer-inference")):
        if batch_idx >= args.max_batches:
            break

        history = _prepare_history_for_timer(
            batch.history.to(device=device, dtype=torch.float32),
            token_len=token_len,
            max_context=max_context,
        )
        future = batch.future.to(device=device, dtype=torch.float32)

        generated = model.generate(
            history,
            max_new_tokens=args.pred_len,
            revin=args.timer_revin,
        )
        pred = _extract_generated_forecast(generated, args.pred_len).to(future.dtype)

        mse = F.mse_loss(pred, future, reduction="mean")
        mae = F.l1_loss(pred, future, reduction="mean")
        batch_size = future.size(0)

        total_mse += float(mse.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_n += batch_size

        base_idx = batch_idx * args.batch_size
        if args.plot_every > 0:
            for local_idx in range(batch_size):
                global_idx = base_idx + local_idx
                if global_idx % args.plot_every == 0:
                    _save_plot(
                        os.path.join(out_dir, "sample_plots"),
                        global_idx,
                        batch.history[local_idx],
                        pred[local_idx],
                        future[local_idx],
                        batch.series_name[local_idx],
                    )

    if total_n == 0:
        raise RuntimeError("No evaluation samples were processed.")

    metrics: Dict[str, float | int | str] = {
        "model_id": args.model_id,
        "eval_split": args.eval_split,
        "history_len": args.history_len,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "timer_token_len": token_len,
        "timer_max_context": max_context,
        "timer_revin": bool(args.timer_revin),
        "mean_mse": total_mse / total_n,
        "mean_mae": total_mae / total_n,
        "n": total_n,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Timer zero-shot ETTh1 inference")
    ap.add_argument("--model_id", default="thuml/timer-base-84m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--etth1_csv", default=os.path.join(os.environ.get("FTS_ETT_DIR", "data/ETT"), "ETTh1.csv"))
    ap.add_argument("--eval_split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--history_len", type=int, default=480)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--etth1_stride", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=10_000_000)
    ap.add_argument("--max_context", type=int, default=0,
                    help="Optional hard cap on raw context length. 0 = use Timer config.")
    ap.add_argument("--timer_revin", action="store_true",
                    help="Pass revin=True into Timer generate(). Disabled by default because "
                         "this evaluator already uses history-normalized ETTh1 windows.")
    ap.add_argument("--plot_every", type=int, default=0)
    ap.add_argument("--out_dir", default="./timer_etth1_eval")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_timer_forecast(args)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "metrics_timer.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n{'=' * 60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")
    print(f"{'=' * 60}")
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
