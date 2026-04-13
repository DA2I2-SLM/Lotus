# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Phase2DownstreamDataset
-----------------------
Standalone PyTorch Dataset combining two sources for Phase-2 fine-tuning:

  1. Downstream forecasting benchmarks (type="chronos")
     ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity
     → MSE path in compute_joint_phase2_loss (decoder-only, LLM frozen)

  2. ChatTS sft + ift instruction-following (type="chatts")
     → CE path in compute_joint_phase2_loss (text generation, LLM + LoRA)

Sample dict format is compatible with:
  - extend_time_series_to_match_patch_size_and_aggregate (collate)
  - model.pad_and_apply_batch
  - model.compute_joint_phase2_loss

Usage (standalone sanity check):
    cd /home/s223540177/FoundationalTS
    conda run -n opentslm python \\
        src/opentslm/time_series_datasets/Phase2DownstreamDataset.py
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOWNSTREAM_DATA_DIR = "/scratch/s223540177/TimeSeriesECM/dataset"
CHATTS_DIR = "/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS"

# ---------------------------------------------------------------------------
# Dataset split ratios
# ETT: 60/20/20 (standard in PatchTST, TimesNet, iTransformer, …)
# Others: 70/10/20
# ---------------------------------------------------------------------------

_FORECAST_DATASETS: Dict[str, Dict] = {
    # name          csv filename          split style    max cols (None = all)
    "ETTh1":       {"file": "ETTh1.csv",       "split": "ett",    "max_cols": None},
    "ETTh2":       {"file": "ETTh2.csv",       "split": "ett",    "max_cols": None},
    "ETTm1":       {"file": "ETTm1.csv",       "split": "ett",    "max_cols": None},
    "ETTm2":       {"file": "ETTm2.csv",       "split": "ett",    "max_cols": None},
    "weather":     {"file": "weather.csv",     "split": "other",  "max_cols": None},
    "traffic":     {"file": "traffic.csv",     "split": "other",  "max_cols": 16},
    "electricity": {"file": "electricity.csv", "split": "other",  "max_cols": 16},
}


# ---------------------------------------------------------------------------
# Multi-horizon config (Time-MoE convention)
#
# We train/evaluate on the four standard long-horizon forecasting targets and
# map each horizon to a Time-MoE-style context length target:
#   pred=96  -> context=512
#   pred=192 -> context=512
#   pred=336 -> context=672
#   pred=720 -> context=720
#
# The active history length is aligned to the runtime patch_size when needed.
# Forecast horizons themselves must remain divisible by patch_size because the
# Phase-2 decoder loss predicts exactly `pred_len // patch_size` future patches.
#
# Training samples windows randomly for each (series, horizon) pair.
# Validation/test enumerate windows deterministically.
# ---------------------------------------------------------------------------

_DEFAULT_PRED_LENS = [96, 192, 336, 720]

_TIME_MOE_CONTEXT_BY_PRED: Dict[int, int] = {
    96: 512,
    192: 512,
    336: 672,
    720: 720,
}

_NORM_STD_FLOOR = 1e-3
_NORM_CLIP = 20.0


def _align_down_to_patch_size(length: int, patch_size: int) -> int:
    """Round a positive length down to the nearest patch boundary."""
    return (length // patch_size) * patch_size


def _resolve_horizon_pairs(
    patch_size: int,
    pred_lens: Optional[List[int]] = None,
    horizon_pairs: Optional[List[Tuple[int, int]]] = None,
    history_len: Optional[int] = None,
    pred_len: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Resolve the active (history_len, pred_len) pairs for this dataset.

    Priority:
      1. Explicit `horizon_pairs`
      2. Backward-compatible single `(history_len, pred_len)` pair
      3. Time-MoE defaults driven by `pred_lens`
    """
    if horizon_pairs is not None:
        requested_pairs = [(int(h), int(p)) for h, p in horizon_pairs]
    elif history_len is not None or pred_len is not None:
        if history_len is None or pred_len is None:
            raise ValueError(
                "history_len and pred_len must be provided together for single-horizon compatibility mode."
            )
        requested_pairs = [(int(history_len), int(pred_len))]
    else:
        if pred_lens is None:
            pred_lens = _DEFAULT_PRED_LENS
        elif isinstance(pred_lens, (int, float)):
            pred_lens = [int(pred_lens)]
        else:
            pred_lens = [int(p) for p in pred_lens]

        unsupported = [p for p in pred_lens if p not in _TIME_MOE_CONTEXT_BY_PRED]
        if unsupported:
            raise ValueError(
                f"Unsupported pred_lens {unsupported}. "
                f"Supported Time-MoE horizons are {sorted(_TIME_MOE_CONTEXT_BY_PRED)} "
                "unless explicit horizon_pairs are provided."
            )
        requested_pairs = [(_TIME_MOE_CONTEXT_BY_PRED[p], p) for p in pred_lens]

    resolved_pairs: List[Tuple[int, int]] = []
    seen = set()
    skipped = []

    for requested_history_len, requested_pred_len in requested_pairs:
        if requested_pred_len <= 0 or requested_history_len <= 0:
            skipped.append(
                f"(history={requested_history_len}, pred={requested_pred_len}) -> lengths must be positive"
            )
            continue

        if requested_pred_len % patch_size != 0:
            skipped.append(
                f"(history={requested_history_len}, pred={requested_pred_len}) -> "
                f"pred_len not divisible by patch_size={patch_size}"
            )
            continue

        aligned_history_len = requested_history_len
        if aligned_history_len % patch_size != 0:
            aligned_history_len = _align_down_to_patch_size(aligned_history_len, patch_size)

        if aligned_history_len <= 0:
            skipped.append(
                f"(history={requested_history_len}, pred={requested_pred_len}) -> "
                f"history aligned to 0 for patch_size={patch_size}"
            )
            continue

        pair = (aligned_history_len, requested_pred_len)
        if pair not in seen:
            seen.add(pair)
            resolved_pairs.append(pair)

    if not resolved_pairs:
        details = "; ".join(skipped) if skipped else "no valid horizon pairs remained"
        raise ValueError(
            f"No valid horizon pairs for patch_size={patch_size}. {details}"
        )

    return resolved_pairs


def _split_indices(n: int, split_style: str, mode: str) -> Tuple[int, int]:
    """Return (start, end) row indices for the requested split."""
    if split_style == "ett":
        boundaries = (int(n * 0.6), int(n * 0.8))
    else:  # "other"
        boundaries = (int(n * 0.7), int(n * 0.8))
    train_end, val_end = boundaries
    if mode == "train":
        return 0, train_end
    elif mode == "val":
        return train_end, val_end
    else:  # test
        return val_end, n


def _make_sample(
    df_window: "torch.Tensor",   # [history_len + pred_len, C]  all columns
    col_names: List[str],
    history_len: int,
    pred_len: int,
    ds_name: str,
    eos_token: str,
) -> dict:
    """Slice one multivariate window and build the sample dict.

    Each column is normalised independently with history-only z-norm
    (RevIN convention).  All channels are kept in a single sample so that
    the 2-D encoder can do cross-channel attention.
    """
    C = df_window.shape[1]
    history = df_window[:history_len]          # [history_len, C]
    future  = df_window[history_len:]          # [pred_len,    C]

    time_series      = []
    time_series_text = []
    futures_norm     = []
    norm_stats       = []

    for c in range(C):
        h = history[:, c]
        f = future[:, c]
        mean = h.mean().item()
        std  = max(h.std().item(), _NORM_STD_FLOOR)
        h_norm = ((h - mean) / std).clamp(-_NORM_CLIP, _NORM_CLIP)
        f_norm = ((f - mean) / std).clamp(-_NORM_CLIP, _NORM_CLIP)
        time_series.append(h_norm.tolist())
        time_series_text.append(
            f"This is the time series ({col_names[c]}), "
            f"mean {mean:.4f} std {std:.4f}. <ts>"
        )
        futures_norm.append(f_norm)
        norm_stats.append((mean, std))

    # ts_future: [C, pred_len] — one future per channel
    ts_future = torch.stack(futures_norm, dim=0)   # [C, pred_len]

    return {
        "type":                   "chronos",
        "batch_group":            f"chronos:{ds_name}:{C}:{history_len}:{pred_len}",
        "pre_prompt":             (
            f"This is {ds_name} data. "
            f"You are given {history_len} timesteps across {C} channels:"
        ),
        "post_prompt":            f". Predict the next {pred_len} timesteps",
        "answer":                 f"<ts></ts>{eos_token}",
        "time_series":            time_series,
        "time_series_text":       time_series_text,
        "ts_future":              ts_future,
        "ts_already_normalized":  True,
        "is_forecasting":         True,
        "n_channels":             C,
        "time_series_norm_stats": norm_stats,
        "_dataset":               ds_name,
        "_history_len":           history_len,
        "_pred_len":              pred_len,
    }


def _load_forecast_samples(
    mode: str,
    patch_size: int,
    max_cols_wide: Optional[int],
    datasets_dir: str,
    eos_token: str,
    horizon_pairs: List[Tuple[int, int]],
    windows_per_col_train: int,
    seed: int,
) -> List[dict]:
    """Build multi-horizon forecasting samples from all downstream CSVs.

    Training:   windows_per_col_train == 0  →  enumerate ALL windows with stride=1
                                               (standard PatchTST/iTransformer convention).
                windows_per_col_train  > 0  →  sample that many random start positions
                                               per (series, horizon) pair.
    Val / test: enumerate all non-overlapping windows with stride=pred_len, deterministic.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for Phase2DownstreamDataset. "
            "Install with: pip install pandas"
        )

    # Filter horizon pairs to those divisible by patch_size
    valid_pairs = [
        (h, p) for h, p in horizon_pairs
        if h % patch_size == 0 and p % patch_size == 0
    ]
    if not valid_pairs:
        raise ValueError(
            f"No horizon pairs in {horizon_pairs} are divisible by patch_size={patch_size}. "
            "Adjust pred_lens/horizon_pairs or reduce patch_size."
        )

    rng = random.Random(seed)
    samples: List[dict] = []

    for ds_name, cfg in _FORECAST_DATASETS.items():
        csv_path = os.path.join(datasets_dir, cfg["file"])
        if not os.path.exists(csv_path):
            print(f"[Phase2DownstreamDataset] {ds_name}: not found at {csv_path}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df = df.select_dtypes(include="number")
        if df.empty:
            print(f"[Phase2DownstreamDataset] {ds_name}: no numeric columns, skipping")
            continue

        cap = cfg.get("max_cols") if max_cols_wide is None else max_cols_wide
        if cap is not None and df.shape[1] > cap:
            df = df.iloc[:, :cap]

        N = len(df)
        start_row, end_row = _split_indices(N, cfg["split"], mode)
        df_split = df.iloc[start_row:end_row].reset_index(drop=True)
        T = len(df_split)

        # All columns stacked: [T, C]
        data = torch.tensor(df_split.values, dtype=torch.float32)
        col_names = list(df_split.columns)
        C = data.shape[1]

        n_before = len(samples)
        for history_len, pred_len in valid_pairs:
            window_len = history_len + pred_len
            if T < window_len:
                continue  # series too short for this horizon

            max_start = T - window_len  # inclusive upper bound

            if mode == "train":
                if windows_per_col_train == 0:
                    # All windows, stride=1 — matches PatchTST/iTransformer convention
                    starts = list(range(0, max_start + 1))
                else:
                    # Random sampling: `windows_per_col_train` draws per horizon
                    starts = [rng.randint(0, max_start) for _ in range(windows_per_col_train)]
            else:
                # Deterministic non-overlapping windows (stride = pred_len)
                starts = list(range(0, max_start + 1, pred_len))

            for win_start in starts:
                df_window = data[win_start : win_start + window_len]   # [window_len, C]
                samples.append(_make_sample(
                    df_window, col_names, history_len, pred_len,
                    ds_name, eos_token,
                ))

        n_added = len(samples) - n_before
        print(
            f"[Phase2DownstreamDataset] {ds_name} {mode}: "
            f"{C} cols × {len(valid_pairs)} horizons → {n_added} samples"
        )

    return samples


def _load_chatts_samples(
    mode: str,
    chatts_dir: str,
    eos_token: str,
    splits: Tuple[float, float, float],
    seed: int,
) -> List[dict]:
    """Load ChatTS sft + ift JSONL files and split into train/val/test."""
    rng = random.Random(seed)
    all_rows: List[dict] = []

    for subset in ("sft", "ift"):
        jsonl_path = os.path.join(chatts_dir, subset, "train.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"[Phase2DownstreamDataset] ChatTS {subset}: not found at {jsonl_path}, skipping")
            continue
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    row["_subset"] = subset
                    all_rows.append(row)
                except json.JSONDecodeError:
                    continue
        print(f"[Phase2DownstreamDataset] ChatTS {subset}: loaded {len(all_rows)} rows so far")

    if not all_rows:
        return []

    # Deterministic shuffle then split
    rng.shuffle(all_rows)
    n = len(all_rows)
    train_end = int(n * splits[0])
    val_end   = int(n * (splits[0] + splits[1]))

    if mode == "train":
        rows = all_rows[:train_end]
    elif mode == "val":
        rows = all_rows[train_end:val_end]
    else:
        rows = all_rows[val_end:]

    samples: List[dict] = []
    skipped = 0
    for row in rows:
        try:
            sample = _format_chatts_row(row, eos_token)
            if sample is not None:
                samples.append(sample)
            else:
                skipped += 1
        except Exception:
            skipped += 1
            continue

    print(
        f"[Phase2DownstreamDataset] ChatTS {mode}: "
        f"{len(samples)} samples ({skipped} skipped)"
    )
    return samples


def _format_chatts_row(row: dict, eos_token: str) -> Optional[dict]:
    """Convert a raw ChatTS JSONL row to a sample dict."""
    ts_raw = row.get("timeseries")
    if ts_raw is None:
        return None

    # timeseries is [[v0, v1, ...]] — shape [C, T]
    try:
        ts = torch.tensor(ts_raw, dtype=torch.float32)  # [C, T]
    except Exception:
        return None

    if ts.dim() == 1:
        ts = ts.unsqueeze(0)  # treat as [1, T]
    if ts.dim() != 2 or ts.numel() == 0:
        return None

    C, _ = ts.shape

    time_series      = []
    time_series_text = []
    for i in range(C):
        ch = ts[i]
        m  = ch.mean().item()
        s  = max(ch.std().item(), _NORM_STD_FLOOR)
        ch_norm = ((ch - m) / s).clamp(-_NORM_CLIP, _NORM_CLIP)
        time_series.append(ch_norm.tolist())
        time_series_text.append(
            f"This is time series {i}, mean {m:.4f} std {s:.4f}. <ts>"
        )

    answer = row.get("output", "")
    if not answer.endswith(eos_token):
        answer = answer + eos_token

    return {
        "type":             "chatts",
        "batch_group":      f"chatts:{row.get('_subset', 'unknown')}:{C}",
        "pre_prompt":       row.get("input", ""),
        "post_prompt":      "Answer:",
        "answer":           answer,
        "time_series":      time_series,
        "time_series_text": time_series_text,
        "ts_already_normalized": True,
        "n_channels":       C,
    }


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class Phase2DownstreamDataset(Dataset):
    """
    Joint Phase-2 downstream dataset combining:

      1. Downstream forecasting benchmarks (ETT, Weather, Traffic, Electricity)
         → type="chronos" → MSE path in compute_joint_phase2_loss

      2. ChatTS sft + ift instruction-following
         → type="chatts" → CE path in compute_joint_phase2_loss

    All samples are loaded eagerly at init and stored in a flat list.
    The dataset is compatible with extend_time_series_to_match_patch_size_and_aggregate
    and compute_joint_phase2_loss without any changes to the model or loss function.

    Args:
        mode:             "train", "val", or "test"
        EOS_TOKEN:        EOS token string from model.get_eos_token()
        patch_size:       Patch size (must match the model). History lengths are
                          aligned to patch boundaries; pred_len must be divisible by it.
        pred_len:         Optional single forecast horizon for backward compatibility.
        history_len:      Optional single history context for backward compatibility.
        forecast_stride:  Deprecated compatibility argument; ignored in Time-MoE mode.
        pred_lens:        Forecast horizons to train/evaluate jointly. Defaults to
                          Time-MoE horizons [96, 192, 336, 720].
        horizon_pairs:    Explicit (history_len, pred_len) overrides. If provided,
                          these take precedence over pred_lens and single-horizon args.
        windows_per_col_train:
                          Number of random windows to sample per (dataset column, horizon)
                          during training.
        max_cols_wide:    Max columns to use from wide datasets (traffic: 863, electricity: 321).
                          None = use all (memory-heavy for traffic/electricity).
        chatts_splits:    (train_frac, val_frac, test_frac) for splitting ChatTS JSONL files.
        datasets_dir:     Root directory containing the downstream CSV files.
        chatts_dir:       Root directory containing ChatTS sft/ and ift/ subdirectories.
        seed:             Random seed for shuffling.
    """

    def __init__(
        self,
        mode: str,
        EOS_TOKEN: str,
        patch_size: int = 16,
        pred_len: Optional[int] = None,
        history_len: Optional[int] = None,
        forecast_stride: Optional[int] = None,
        pred_lens: Optional[List[int]] = None,
        horizon_pairs: Optional[List[Tuple[int, int]]] = None,
        windows_per_col_train: int = 0,
        max_cols_wide: Optional[int] = 16,
        chatts_splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        datasets_dir: str = DOWNSTREAM_DATA_DIR,
        chatts_dir: str = CHATTS_DIR,
        seed: int = 42,
    ):
        if mode == "validation":
            mode = "val"
        assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
        assert patch_size > 0, f"patch_size must be positive, got {patch_size}"
        assert windows_per_col_train >= 0, (
            "windows_per_col_train must be non-negative, got "
            f"{windows_per_col_train}"
        )

        resolved_horizon_pairs = _resolve_horizon_pairs(
            patch_size=patch_size,
            pred_lens=pred_lens,
            horizon_pairs=horizon_pairs,
            history_len=history_len,
            pred_len=pred_len,
        )

        self.mode               = mode
        self.EOS_TOKEN          = EOS_TOKEN
        self.patch_size         = patch_size
        self.horizon_pairs      = resolved_horizon_pairs
        self.pred_lens          = [pred for _, pred in resolved_horizon_pairs]
        self.history_lens       = [hist for hist, _ in resolved_horizon_pairs]
        if len(resolved_horizon_pairs) == 1:
            self.history_len, self.pred_len = resolved_horizon_pairs[0]
        else:
            self.history_len, self.pred_len = None, None
        self.windows_per_col_train = int(windows_per_col_train)

        if windows_per_col_train == 0:
            windows_desc = "all_windows_stride1"
        else:
            windows_desc = str(windows_per_col_train)

        print(f"\n[Phase2DownstreamDataset] Loading {mode} split "
              f"(patch={patch_size}, horizon_pairs={resolved_horizon_pairs}, "
              f"windows_per_col_train={windows_desc})")
        if forecast_stride is not None:
            print(
                "[Phase2DownstreamDataset] forecast_stride is ignored in Time-MoE mode; "
                "validation/test use stride=pred_len."
            )

        # ── Forecasting samples ───────────────────────────────────────────
        forecast_samples = _load_forecast_samples(
            mode=mode,
            patch_size=patch_size,
            max_cols_wide=max_cols_wide,
            datasets_dir=datasets_dir,
            eos_token=EOS_TOKEN,
            horizon_pairs=resolved_horizon_pairs,
            windows_per_col_train=windows_per_col_train,
            seed=seed,
        )

        # ── ChatTS samples ────────────────────────────────────────────────
        chatts_samples = _load_chatts_samples(
            mode=mode,
            chatts_dir=chatts_dir,
            eos_token=EOS_TOKEN,
            splits=chatts_splits,
            seed=seed,
        )

        # ── Merge and shuffle ─────────────────────────────────────────────
        self.samples = forecast_samples + chatts_samples
        rng = random.Random(seed)
        rng.shuffle(self.samples)
        self.rows = self.samples

        n_forecast = len(forecast_samples)
        n_chatts   = len(chatts_samples)
        print(
            f"[Phase2DownstreamDataset] {mode}: "
            f"total={len(self.samples)} "
            f"(forecast={n_forecast}, chatts={n_chatts})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    EOS = "</s>"  # Gemma-style; replace with model.get_eos_token() during real training

    for split in ("train", "val", "test"):
        ds = Phase2DownstreamDataset(
            mode=split,
            EOS_TOKEN=EOS,
            patch_size=16,
            max_cols_wide=16,
        )
        print(f"\n  [{split}] total={len(ds)}")
        print(f"  horizon_pairs={ds.horizon_pairs}")

        # Sample one forecasting sample
        for i, s in enumerate(ds.samples):
            if s["type"] == "chronos":
                print(f"\n  Forecast sample [{i}] dataset={s.get('_dataset')} col={s.get('_col')}")
                print(f"    pre_prompt:   {s['pre_prompt']}")
                print(f"    post_prompt:  {s['post_prompt']}")
                print(f"    answer:       {s['answer']}")
                ts = s["time_series"][0]
                print(f"    time_series:  len={len(ts)}  first5={[round(v,3) for v in ts[:5]]}")
                print(f"    ts_future:    shape={s['ts_future'].shape}  first5={s['ts_future'][:5].tolist()}")
                print(f"    norm_mean={s['norm_mean']:.4f}  norm_std={s['norm_std']:.4f}")
                break

        # Sample one ChatTS sample
        for i, s in enumerate(ds.samples):
            if s["type"] == "chatts":
                print(f"\n  ChatTS sample [{i}]")
                print(f"    pre_prompt:   {s['pre_prompt'][:120]}...")
                print(f"    answer:       {s['answer'][:120]}...")
                print(f"    n_channels:   {len(s['time_series'])}")
                print(f"    ts len:       {len(s['time_series'][0])}")
                break

        if split == "train":
            break  # only full check on train for speed
