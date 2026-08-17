# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Phase2DownstreamDataset
-----------------------
Standalone PyTorch Dataset combining two sources for Phase-2 fine-tuning:

  1. Downstream forecasting benchmarks (type="chronos")
     ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity
     → MSE path in compute_joint_phase2_loss
       encoder+projector frozen; LLM LoRA + decoder_projector + decoder trainable.

  2. ChatTS sft + ift instruction-following (type="chatts")
     → CE path in compute_joint_phase2_loss
       encoder+projector frozen; LLM LoRA trainable.

Sample dict format is compatible with:
  - extend_time_series_to_match_patch_size_and_aggregate (collate)
  - model.pad_and_apply_batch
  - model.compute_joint_phase2_loss

Usage (standalone sanity check):
    cd /home/$(whoami)/FoundationalTS
    conda run -n opentslm python \\
        src/opentslm/time_series_datasets/Phase2DownstreamDataset.py
"""

import getpass
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOWNSTREAM_DATA_DIR = f"/scratch/{getpass.getuser()}/TimeSeriesECM/dataset"
CHATTS_DIR = "/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS"


# ---------------------------------------------------------------------------
# CoT integration (HAR / SleepEDF / ECG-QA) — added so phase-2 training also
# covers stages 3/4/5 jointly in a single joint_phase2_downstream pass.
# ---------------------------------------------------------------------------

# Per-task default _seq_len estimate (constant per dataset; avoids reading
# row data upfront). Used by the homogeneous batch sampler for bucketing only.
_COT_SEQ_LEN_ESTIMATE = {
    "har_cot":   3 * 4 + 500,
    "sleep_cot": 1 * 94 + 800,
    "ecg_cot":   12 * 156 + 800,
}
_COT_N_CH_DEFAULT = {"har_cot": 3, "sleep_cot": 1, "ecg_cot": 12}


class _LazyCoTAdapter:
    """Lazy wrapper around a QADataset subclass: keeps the raw HuggingFace
    Dataset on disk and formats individual samples on demand in __getitem__.

    Bypasses QADataset.__init__'s eager preload + format pass — that pass
    is what OOMs when loading HAR+Sleep+ECG-QA together (~235k formatted
    samples each holding numpy time series in memory).
    """

    def __init__(self, qa_dataset_cls, split, EOS_TOKEN, type_label, patch_size, **kw):
        instance = qa_dataset_cls.__new__(qa_dataset_cls)
        instance.EOS_TOKEN = EOS_TOKEN
        for k, v in kw.items():
            setattr(instance, k, v)
        for attr, default in (
            ("max_samples", None),
            ("exclude_comparison", False),
            ("preload_processed_data", False),
        ):
            if not hasattr(instance, attr):
                setattr(instance, attr, default)
        raw_train, raw_val, raw_test = instance._load_splits()
        self.raw = raw_val if split == "validation" else (raw_test if split == "test" else raw_train)
        self._instance = instance
        self.type_label = type_label
        self.patch_size = patch_size
        self.EOS_TOKEN = EOS_TOKEN

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        row = self.raw[int(idx)]
        formatted = self._instance._format_sample(row)
        ts_list = [(ts.tolist() if hasattr(ts, "tolist") else list(ts)) for ts in formatted["time_series"]]
        answer = formatted["answer"]
        if not answer.endswith(self.EOS_TOKEN):
            answer = answer + self.EOS_TOKEN
        C = len(ts_list)
        n_ts_patches = sum(math.ceil(len(ts) / self.patch_size) for ts in ts_list)
        text_len_est = (
            len(formatted.get("pre_prompt", ""))
            + len(formatted.get("post_prompt", ""))
            + len(answer)
        )
        return {
            "type":             self.type_label,
            "pre_prompt":       formatted.get("pre_prompt", ""),
            "post_prompt":      formatted.get("post_prompt", ""),
            "answer":           answer,
            "time_series":      ts_list,
            "time_series_text": formatted.get("time_series_text", []),
            "n_channels":       C,
            "_n_channels":      C,
            "_seq_len":         n_ts_patches + text_len_est,
        }


def _load_cot_lazy(qa_dataset_cls, mode, EOS_TOKEN, type_label, patch_size, **kw):
    """Return (lazy_adapter, proxy_rows). Proxy rows hold constant per-task
    _seq_len so the sampler can bucket without materializing samples."""
    split = "validation" if mode == "val" else mode
    try:
        adapter = _LazyCoTAdapter(qa_dataset_cls, split, EOS_TOKEN, type_label, patch_size, **kw)
    except Exception as e:
        print(f"[Phase2DownstreamDataset] {type_label} {mode}: skipping lazy ({type(e).__name__}: {e})")
        return None, []
    n = len(adapter)
    seq_len_est = _COT_SEQ_LEN_ESTIMATE.get(type_label, 1000)
    n_ch_default = _COT_N_CH_DEFAULT.get(type_label, 1)
    proxy_rows = [{
        "type":        type_label,
        "_n_channels": n_ch_default,
        "_seq_len":    seq_len_est,
    } for _ in range(n)]
    print(
        f"[Phase2DownstreamDataset] {type_label} {mode} (lazy): {n} samples "
        f"(proxy only; samples fetched on demand)"
    )
    return adapter, proxy_rows


def _resize_list(lst: List[dict], target_n: int, seed: int) -> List[dict]:
    """Resize a list of sample dicts to exactly target_n by repeating (upsample)
    or shuffled-truncating (downsample). Repeats share dict refs (no data copy)."""
    if not lst or target_n <= 0:
        return []
    if target_n <= len(lst):
        out = list(lst)
        random.Random(seed).shuffle(out)
        return out[:target_n]
    reps, rem = divmod(target_n, len(lst))
    out = lst * reps
    if rem:
        extra = list(lst)
        random.Random(seed).shuffle(extra)
        out += extra[:rem]
    return out


def _load_cot_samples(
    qa_dataset_cls,
    mode: str,
    EOS_TOKEN: str,
    type_label: str,
    patch_size: int,
    **extra_kwargs,
) -> List[dict]:
    """Adapt a QADataset subclass (HARCoTQADataset / SleepEDFCoTQADataset /
    ECGQACoTQADataset) into Phase2-format samples.

    Each output sample has the same dict layout as ChatTS samples so the
    existing compute_joint_phase2_loss CE path handles them uniformly.

    Args:
        qa_dataset_cls: The QADataset subclass to instantiate.
        mode:           "train", "val", or "test".
        EOS_TOKEN:      EOS token for the LLM (appended to each answer).
        type_label:     "har_cot" / "sleep_cot" / "ecg_cot" — used by the
                        homogeneous batch sampler to keep each batch
                        single-task.
        patch_size:     Patch size for the _seq_len estimate.
        extra_kwargs:   Extra kwargs forwarded to the dataset constructor
                        (e.g. preload_processed_data=False for ECG).
    """
    split = "validation" if mode == "val" else mode  # QADataset uses "validation"
    try:
        ds = qa_dataset_cls(split=split, EOS_TOKEN=EOS_TOKEN, **extra_kwargs)
    except Exception as e:
        print(
            f"[Phase2DownstreamDataset] {type_label} {mode}: skipping — "
            f"failed to load ({type(e).__name__}: {e})"
        )
        return []

    samples: List[dict] = []
    for row in ds.dataset:
        # row is PromptWithAnswer.to_dict() — already has the right schema
        # (pre_prompt, post_prompt, answer, time_series, time_series_text).
        time_series = row["time_series"]
        time_series_text = row["time_series_text"]
        # Coerce numpy arrays / tensors → Python lists for consistency with chatts.
        ts_list: List[List[float]] = []
        for ts in time_series:
            if hasattr(ts, "tolist"):
                ts_list.append(ts.tolist())
            else:
                ts_list.append(list(ts))
        C = len(ts_list)
        if C == 0:
            continue

        answer = row["answer"]
        if not answer.endswith(EOS_TOKEN):
            answer = answer + EOS_TOKEN

        n_ts_patches = sum(math.ceil(len(ts) / patch_size) for ts in ts_list)
        text_len_est = (
            len(row.get("pre_prompt", ""))
            + len(row.get("post_prompt", ""))
            + len(answer)
        )

        samples.append({
            "type":             type_label,
            "pre_prompt":       row.get("pre_prompt", ""),
            "post_prompt":      row.get("post_prompt", ""),
            "answer":           answer,
            "time_series":      ts_list,
            "time_series_text": time_series_text,
            "n_channels":       C,
            "_n_channels":      C,
            "_seq_len":         n_ts_patches + text_len_est,
        })

    print(
        f"[Phase2DownstreamDataset] {type_label} {mode}: "
        f"{len(samples)} samples loaded"
    )
    return samples

# ---------------------------------------------------------------------------
# Dataset split ratios
# ETTh: fixed Informer counts — 8640 train / 2880 val / 2880 test (12/4/4 months)
# ETTm: fixed Informer counts — 34560 train / 11520 val / 11520 test (4× ETTh)
# Others: 70/10/20 ratio
# ---------------------------------------------------------------------------

_FORECAST_DATASETS: Dict[str, Dict] = {
    # name          csv filename          split style    max cols (None = all)
    "ETTh1":       {"file": "ETTh1.csv",       "split": "etth",   "max_cols": None},
    "ETTh2":       {"file": "ETTh2.csv",       "split": "etth",   "max_cols": None},
    "ETTm1":       {"file": "ETTm1.csv",       "split": "ettm",   "max_cols": None},
    "ETTm2":       {"file": "ETTm2.csv",       "split": "ettm",   "max_cols": None},
    "weather":     {"file": "weather.csv",     "split": "other",  "max_cols": None},
    # traffic (862 ch) and electricity (321 ch) removed: even capped to 16 cols,
    # the multivariate samples at long horizons (pred_len=720) produce per-sample
    # token counts that don't fit on V100-32GB backward, even as B=1 batches.
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
# Forecast horizons need not be divisible by patch_size; the model generates
# ceil(pred_len / patch_size) patches and the last patch is trimmed at eval time.
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
    if split_style == "etth":
        train_end, val_end, test_end = 8640, 11520, 14400
    elif split_style == "ettm":
        train_end, val_end, test_end = 34560, 46080, 57600
    else:  # "other"
        train_end = int(n * 0.7)
        val_end   = int(n * 0.8)
        test_end  = n
    if mode == "train":
        return 0, train_end
    elif mode == "val":
        return train_end, val_end
    else:  # test
        return val_end, test_end


def _make_sample(
    df_window: "torch.Tensor",   # [history_len + pred_len, C]  all columns
    col_names: List[str],
    history_len: int,
    pred_len: int,
    ds_name: str,
    eos_token: str,
    ch_sep_token: str,
    fp_tokens: List[str],
    patch_size: int,
    global_mean: "torch.Tensor",  # [C]  training-split mean per channel
    global_std:  "torch.Tensor",  # [C]  training-split std per channel (clamped ≥ 1e-5)
) -> dict:
    """Slice one multivariate window and build the sample dict.

    Each column is normalised with global statistics (mean and std computed
    from the training split of each CSV), following the standard LTSF
    benchmark convention (PatchTST, iTransformer, TimeMixer).  Using global
    stats instead of per-window history stats prevents extreme normalised
    target values for sparse channels (e.g. rain) and avoids sentinel-value
    corruption from -9999 placeholders.

    Answer format:
        <ts> [<future_patch> × (N_per_ch × C), time-major] </ts> <eos>
    """
    C = df_window.shape[1]
    history = df_window[:history_len]          # [history_len, C]
    future  = df_window[history_len:]          # [pred_len,    C]

    time_series      = []
    time_series_text = []
    futures_norm     = []
    norm_stats       = []

    # Instance (per-window) normalization — RevIN-style. Default OFF (global-train
    # stats) for backward compat; PHASE2_INSTANCE_NORM=1 normalizes each channel by
    # its OWN history mean/std, so the model predicts residuals around the current
    # level instead of the train mean. This fixes the level-anchoring failure on
    # distribution-shifted series (ETT), matching how PatchTST/iTransformer/Chronos
    # handle it and mirroring the per-window norm the CiK forecaster already uses.
    _instance_norm = os.environ.get("PHASE2_INSTANCE_NORM") == "1"

    for c in range(C):
        h    = history[:, c]
        f    = future[:, c]
        if _instance_norm:
            mean = h.mean().item()
            std  = h.std().item()
            # Larger floor than global (1e-5): a near-constant window has tiny std,
            # which would explode f_norm. This guards the sparse/constant-channel
            # blow-up that motivated global norm in the first place.
            std  = std if std > 1e-3 else 1e-3
        else:
            mean = global_mean[c].item()
            std  = global_std[c].item()   # pre-clamped to min=1e-5
        h_norm = (h - mean) / std
        f_norm = (f - mean) / std
        time_series.append(h_norm.tolist())
        time_series_text.append(
            f"This is the time series ({col_names[c]}), "
            f"mean {mean:.4f} std {std:.4f}. <ts>"
        )
        futures_norm.append(f_norm)
        norm_stats.append((mean, std))

    # ts_future: [C, pred_len] — one future per channel
    ts_future = torch.stack(futures_norm, dim=0)   # [C, pred_len]

    n_per_ch = math.ceil(pred_len / patch_size)
    answer = "<ts>" + fp_tokens[0] * (n_per_ch * C) + "</ts>" + eos_token

    return {
        "type":                   "chronos",
        "pre_prompt":             (
            f"This is {ds_name} data. "
            f"You are given {history_len} timesteps across {C} channels:"
        ),
        "post_prompt":            (
            f". Predict the next {pred_len} timesteps "
            f"as {n_per_ch} patch{'es' if n_per_ch != 1 else ''} per channel"
        ),
        "answer":                 answer,
        "time_series":            time_series,
        "time_series_text":       time_series_text,
        "ts_future":              ts_future,
        "ts_already_normalized":  True,
        "is_forecasting":         True,
        "n_channels":             C,
        "_n_channels":            C,
        "_n_patches":             n_per_ch,
        "time_series_norm_stats": norm_stats,
        "_dataset":               ds_name,
        "_history_len":           history_len,
        "_pred_len":              pred_len,
        "_seq_len":               C * (history_len // patch_size + n_per_ch),
    }


def _load_forecast_samples(
    mode: str,
    patch_size: int,
    max_cols_wide: Optional[int],
    datasets_dir: str,
    eos_token: str,
    ch_sep_token: str,
    fp_tokens: List[str],
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

        # Replace sentinel values (e.g. -9999 missing-data placeholders) with
        # NaN, then forward/backward fill so no windows contain sentinel noise.
        df = df.mask(df < -100).ffill().bfill().fillna(0.0)

        # Global per-channel statistics from the TRAINING split only.
        # All modes (train/val/test) normalise with the same training-split
        # stats — this is the standard LTSF benchmark convention and prevents
        # extreme normalised targets for sparse channels (rain, raining, etc.).
        N = len(df)
        _train_s, _train_e = _split_indices(N, cfg["split"], "train")
        _train_df  = df.iloc[_train_s:_train_e]
        global_mean = torch.tensor(_train_df.mean().values, dtype=torch.float32)
        global_std  = torch.tensor(_train_df.std().values,  dtype=torch.float32).clamp(min=1e-5)

        start_row, end_row = _split_indices(N, cfg["split"], mode)
        df_split = df.iloc[start_row:end_row].reset_index(drop=True)
        T = len(df_split)

        # All columns stacked: [T, C]
        data = torch.tensor(df_split.values, dtype=torch.float32)
        col_names = list(df_split.columns)
        C = data.shape[1]

        n_before = len(samples)
        for history_len, pred_len in horizon_pairs:
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
                    ch_sep_token, fp_tokens, patch_size,
                    global_mean, global_std,
                ))

        n_added = len(samples) - n_before
        print(
            f"[Phase2DownstreamDataset] {ds_name} {mode}: "
            f"{C} cols × {len(horizon_pairs)} horizons → {n_added} samples"
        )

    return samples


def _build_lazy_entries(
    patch_size: int,
    max_cols_wide: Optional[int],
    datasets_dir: str,
    eos_token: str,
    ch_sep_token: str,
    fp_tokens: List[str],
    horizon_pairs: List[Tuple[int, int]],
) -> List[dict]:
    """Build lazy series entries for phase-1-style random-crop training.

    Returns one entry per (dataset, horizon_pair).  No windows are
    pre-computed; _make_sample is called with a fresh random start at
    __getitem__ time, giving different windows every epoch automatically.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. pip install pandas")

    entries: List[dict] = []
    for ds_name, cfg in _FORECAST_DATASETS.items():
        csv_path = os.path.join(datasets_dir, cfg["file"])
        if not os.path.exists(csv_path):
            print(f"[Phase2DownstreamDataset] {ds_name}: not found at {csv_path}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df = df.select_dtypes(include="number")
        if df.empty:
            continue

        cap = cfg.get("max_cols") if max_cols_wide is None else max_cols_wide
        if cap is not None and df.shape[1] > cap:
            df = df.iloc[:, :cap]

        df = df.mask(df < -100).ffill().bfill().fillna(0.0)

        N = len(df)
        _train_s, _train_e = _split_indices(N, cfg["split"], "train")
        _train_df   = df.iloc[_train_s:_train_e]
        global_mean = torch.tensor(_train_df.mean().values, dtype=torch.float32)
        global_std  = torch.tensor(_train_df.std().values,  dtype=torch.float32).clamp(min=1e-5)

        data      = torch.tensor(_train_df.values, dtype=torch.float32)  # [T_train, C]
        col_names = list(_train_df.columns)
        T         = len(data)

        n_before = len(entries)
        for history_len, pred_len in horizon_pairs:
            window_len = history_len + pred_len
            if T < window_len:
                continue
            entries.append({
                "data":         data,
                "max_start":    T - window_len,
                "col_names":    col_names,
                "global_mean":  global_mean,
                "global_std":   global_std,
                "history_len":  history_len,
                "pred_len":     pred_len,
                "ds_name":      ds_name,
                "eos_token":    eos_token,
                "ch_sep_token": ch_sep_token,
                "fp_tokens":    fp_tokens,
                "patch_size":   patch_size,
            })

        n_added = len(entries) - n_before
        print(
            f"[Phase2DownstreamDataset] {ds_name} train (lazy): "
            f"{n_added} entries (random crop per call)"
        )

    return entries


def _load_chatts_samples(
    mode: str,
    chatts_dir: str,
    eos_token: str,
    splits: Tuple[float, float, float],
    seed: int,
    patch_size: int = 32,
) -> List[dict]:
    """Load ChatTS sft + ift JSONL files and split into train/val/test."""
    rng = random.Random(seed)
    all_rows: List[dict] = []

    for subset in ("sft",):  # ift removed: instruction-tuned subset has long prompts that OOM V100-32GB
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
            sample = _format_chatts_row(row, eos_token, patch_size=patch_size)
            if sample is not None:
                sample["_subset"] = row.get("_subset", "unknown")
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


def _format_chatts_row(row: dict, eos_token: str, patch_size: int = 32) -> Optional[dict]:
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
        s  = max(ch.std().item(), 1e-2)
        ch_norm = ((ch - m) / s)
        time_series.append(ch_norm.tolist())
        time_series_text.append(
            f"This is time series {i}, mean {m:.4f} std {s:.4f}. <ts>"
        )

    answer = row.get("output", "")
    if not answer.endswith(eos_token):
        answer = answer + eos_token

    n_ts_patches = sum(math.ceil(len(ts) / patch_size) for ts in time_series)
    text_len_est = (len(row.get("input", "")) + len(answer))

    return {
        "type":             "chatts",
        "pre_prompt":       row.get("input", ""),
        "post_prompt":      "Answer:",
        "answer":           answer,
        "time_series":      time_series,
        "time_series_text": time_series_text,
        "n_channels":       C,
        "_n_channels":      C,
        "_seq_len":         n_ts_patches + text_len_est,
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
                          aligned to patch boundaries; pred_len need not be divisible.
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
        FUTURE_PATCH_TOKEN: str = "<future_patch>",   # unused; kept for API compat
        CH_SEP_TOKEN: str = "<ch_sep>",
        FP_TOKENS: Optional[List[str]] = None,
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
            f"windows_per_col_train must be non-negative, got {windows_per_col_train}"
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

        print(f"\n[Phase2DownstreamDataset] Loading {mode} split "
              f"(patch={patch_size}, horizon_pairs={resolved_horizon_pairs}, "
              f"windows_per_col_train={windows_per_col_train})")
        if forecast_stride is not None:
            print(
                "[Phase2DownstreamDataset] forecast_stride is ignored in Time-MoE mode; "
                "validation/test use stride=pred_len."
            )

        if FP_TOKENS is None:
            from opentslm.model_config import MAX_FUTURE_PATCH_PE_SIZE as _MPS
            FP_TOKENS = ["<future_patch>"] * _MPS

        # ── Forecasting samples ───────────────────────────────────────────
        # Train + windows_per_col_train > 0  →  lazy random-crop (phase-1 style):
        #   store raw tensors; sample a fresh random window at __getitem__ time
        #   so every epoch sees different windows automatically.
        # Val / test, or windows_per_col_train == 0  →  eager pre-enumeration
        #   (deterministic, all non-overlapping windows or all stride-1 windows).
        self._lazy_entries: Optional[List[dict]] = None
        self._chatts_samples: List[dict] = []

        if mode == "train" and windows_per_col_train > 0:
            self._lazy_entries = _build_lazy_entries(
                patch_size=patch_size,
                max_cols_wide=max_cols_wide,
                datasets_dir=datasets_dir,
                eos_token=EOS_TOKEN,
                ch_sep_token=CH_SEP_TOKEN,
                fp_tokens=FP_TOKENS,
                horizon_pairs=resolved_horizon_pairs,
            )
            self._chatts_samples = _load_chatts_samples(
                mode=mode,
                chatts_dir=chatts_dir,
                eos_token=EOS_TOKEN,
                splits=chatts_splits,
                seed=seed,
                patch_size=patch_size,
            )

            # CoT integration in LAZY mode — load HAR / SleepEDF / ECG-QA train
            # samples too. Critical: without this, train never sees CoT data
            # even though val/test do (non-lazy mode).
            _cot_samples_lazy: List[dict] = []
            try:
                from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
                _cot_samples_lazy += _load_cot_samples(
                    HARCoTQADataset, mode, EOS_TOKEN, "har_cot", patch_size,
                )
            except Exception as _e:
                print(f"[Phase2DownstreamDataset] HARCoTQADataset skipped: {_e}")
            try:
                from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
                _cot_samples_lazy += _load_cot_samples(
                    SleepEDFCoTQADataset, mode, EOS_TOKEN, "sleep_cot", patch_size,
                )
            except Exception as _e:
                print(f"[Phase2DownstreamDataset] SleepEDFCoTQADataset skipped: {_e}")
            # ECG-QA disabled — its lazy adapter caused DDP deadlocks via slow
            # per-sample disk reads. Re-enable via separate stage 5 training.
            self._ecg_adapter = None
            _ecg_proxy_rows: List[dict] = []

            # ChatTS-only seq_len cap (original OOM trigger was oversized ChatTS
            # samples). CoT samples (HAR/Sleep/ECG) have intrinsically longer
            # time series and are kept uncapped — the OOM handler in
            # curriculum_learning.py will skip individually too-large batches
            # if any slip through.
            MAX_CHATTS_SEQ_LEN = 600
            n_chatts_before = len(self._chatts_samples)
            self._chatts_samples = [
                s for s in self._chatts_samples
                if int(s.get("_seq_len", 0)) <= MAX_CHATTS_SEQ_LEN
            ]
            self._cot_samples = _cot_samples_lazy
            self._ecg_proxy_rows = _ecg_proxy_rows
            print(
                f"[Phase2DownstreamDataset] {mode} (lazy) seq_cap<={MAX_CHATTS_SEQ_LEN} (chatts only): "
                f"chatts {n_chatts_before}->{len(self._chatts_samples)}, "
                f"cot eager={len(self._cot_samples)}, "
                f"ecg lazy={len(self._ecg_proxy_rows)}"
            )
            self.samples = []  # unused in lazy mode

            # Pre-build proxy rows for TaskHomogeneousDistributedBatchSampler.
            # The sampler calls dataset.rows to group indices by task/channel count;
            # in lazy mode self.samples is empty, so we synthesize the metadata it needs.
            n_lazy = len(self._lazy_entries)
            self._lazy_proxy_rows: List[dict] = [
                {
                    "type": "chronos",
                    "_n_channels": self._lazy_entries[i % n_lazy]["data"].shape[1],
                    "_n_patches": math.ceil(
                        self._lazy_entries[i % n_lazy]["pred_len"]
                        / self._lazy_entries[i % n_lazy]["patch_size"]
                    ),
                    "_seq_len": self._lazy_entries[i % n_lazy]["data"].shape[1] * (
                        self._lazy_entries[i % n_lazy]["history_len"]
                        // self._lazy_entries[i % n_lazy]["patch_size"]
                        + math.ceil(
                            self._lazy_entries[i % n_lazy]["pred_len"]
                            / self._lazy_entries[i % n_lazy]["patch_size"]
                        )
                    ),
                }
                for i in range(n_lazy * windows_per_col_train)
            ] + list(self._chatts_samples) + list(self._cot_samples) + list(self._ecg_proxy_rows)

            n_forecast = len(self._lazy_entries) * windows_per_col_train
            n_chatts   = len(self._chatts_samples)
            n_cot      = len(self._cot_samples)
            n_ecg      = len(self._ecg_proxy_rows)
            print(
                f"[Phase2DownstreamDataset] {mode} (lazy random-crop): "
                f"total_virtual={n_forecast + n_chatts + n_cot + n_ecg} "
                f"({len(self._lazy_entries)} series × {windows_per_col_train} crops "
                f"+ {n_chatts} ChatTS + {n_cot} CoT + {n_ecg} ECG-lazy)"
            )
        else:
            # Env hook: PHASE2_COT_ONLY=1 -> train ONLY on HAR/Sleep CoT (no forecasting,
            # no ChatTS). Backward-compatible: unset -> load forecast + ChatTS as before.
            _cot_only = os.environ.get("PHASE2_COT_ONLY") == "1"
            if _cot_only:
                forecast_samples = []
                self._chatts_samples = []
            else:
                forecast_samples = _load_forecast_samples(
                    mode=mode,
                    patch_size=patch_size,
                    max_cols_wide=max_cols_wide,
                    datasets_dir=datasets_dir,
                    eos_token=EOS_TOKEN,
                    ch_sep_token=CH_SEP_TOKEN,
                    fp_tokens=FP_TOKENS,
                    horizon_pairs=resolved_horizon_pairs,
                    windows_per_col_train=windows_per_col_train,
                    seed=seed,
                )
                self._chatts_samples = _load_chatts_samples(
                    mode=mode,
                    chatts_dir=chatts_dir,
                    eos_token=EOS_TOKEN,
                    splits=chatts_splits,
                    seed=seed,
                    patch_size=patch_size,
                )

            # CoT integration — load HAR / SleepEDF / ECG-QA samples so phase-2
            # training also covers stages 3/4/5 jointly in a single pass.
            # PHASE2_COT_TASKS (comma-sep) restricts WHICH CoT tasks load — e.g.
            # "pamap2" trains on PAMAP2-CoT only. Unset = load all (legacy behaviour).
            _cot_sel = {t.strip() for t in os.environ.get("PHASE2_COT_TASKS", "").split(",") if t.strip()}
            def _want(_t):
                return (not _cot_sel) or (_t in _cot_sel)
            _har_samples: List[dict] = []
            if _want("har"):
                try:
                    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
                    _har_samples = _load_cot_samples(
                        HARCoTQADataset, mode, EOS_TOKEN, "har_cot", patch_size,
                    )
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] HARCoTQADataset skipped: {_e}")
            _sleep_samples: List[dict] = []
            if _want("sleep"):
                try:
                    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
                    _sleep_samples = _load_cot_samples(
                        SleepEDFCoTQADataset, mode, EOS_TOKEN, "sleep_cot", patch_size,
                    )
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] SleepEDFCoTQADataset skipped: {_e}")
            # PAMAP2-CoT (activity recognition, 18 labels) — a HAR-style classification
            # CoT task. Enabled by PHASE2_INCLUDE_PAMAP2=1 or by naming it in PHASE2_COT_TASKS.
            _pamap_samples: List[dict] = []
            if _want("pamap2") and (os.environ.get("PHASE2_INCLUDE_PAMAP2") == "1" or "pamap2" in _cot_sel):
                try:
                    from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
                    _pamap_samples = _load_cot_samples(
                        PAMAP2CoTQADataset, mode, EOS_TOKEN, "pamap2_cot", patch_size,
                    )
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] PAMAP2CoTQADataset skipped: {_e}")
            # TSQA (ChengsenWang) classification QA — enable via PHASE2_COT_TASKS=tsqa
            _tsqa_samples: List[dict] = []
            if _want("tsqa") and "tsqa" in _cot_sel:
                try:
                    from opentslm.time_series_datasets.tsqa.TSQAClsQADataset import TSQAClsQADataset
                    _tsqa_samples = _load_cot_samples(
                        TSQAClsQADataset, mode, EOS_TOKEN, "tsqa_cot", patch_size,
                    )
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] TSQAClsQADataset skipped: {_e}")
            # TimeMQA Open-Ended-QA (multi-domain free-text QA) — via PHASE2_COT_TASKS=timemqa
            _timemqa_samples: List[dict] = []
            if _want("timemqa") and "timemqa" in _cot_sel:
                try:
                    from opentslm.time_series_datasets.timemqa.TimeMQAOpenQADataset import TimeMQAOpenQADataset
                    _timemqa_samples = _load_cot_samples(
                        TimeMQAOpenQADataset, mode, EOS_TOKEN, "timemqa_cot", patch_size,
                    )
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] TimeMQAOpenQADataset skipped: {_e}")

            # ECG-QA (stage 5): enabled via PHASE2_INCLUDE_ECG=1 (default OFF so
            # existing HAR+Sleep-only runs are unchanged). Uses the LAZY adapter
            # (_load_cot_lazy), NOT _load_cot_samples: eager-materializing
            # 159k x 12 x 1000 floats costs ~85 GB/rank, whereas the lazy adapter
            # is ~2 GB and IO-safe — ECGQACoTQADataset._load_splits() preloads all
            # wfdb signals up front, so __getitem__ is pure in-memory numpy (the
            # old "DDP deadlock via lazy IO" no longer applies with
            # dataloader_workers=0). ECG proxies are appended LAST.
            self._ecg_adapter = None
            self._ecg_proxy_rows: List[dict] = []
            if _want("ecg") and os.environ.get("PHASE2_INCLUDE_ECG") == "1":
                try:
                    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
                    _ecg_adapter, _ecg_proxy = _load_cot_lazy(
                        ECGQACoTQADataset, mode, EOS_TOKEN, "ecg_cot", patch_size,
                        preload_processed_data=False,
                    )
                    if _ecg_adapter is not None:
                        self._ecg_adapter = _ecg_adapter
                        # Tag each proxy with its adapter index so __getitem__ can
                        # resolve it after self.samples is shuffled below.
                        self._ecg_proxy_rows = [
                            {**r, "_ecg_idx": i} for i, r in enumerate(_ecg_proxy)
                        ]
                except Exception as _e:
                    print(f"[Phase2DownstreamDataset] ECGQACoTQADataset skipped: {_e}")

            # HAR/Sleep mixing weight relative to ECG. PHASE2_COT_MIX in
            # {natural, sqrt, balanced} (train only, requires ECG enabled):
            #   natural  -> HAR/Sleep at natural size (ECG ~68% of the mix)
            #   sqrt     -> counts proportional to sqrt(size), normalized to ECG
            #   balanced -> HAR = Sleep = ECG  (1:1:1)
            # ECG always stays at its natural count (it is the largest task).
            # When PHASE2_COT_MIX is unset, fall back to the legacy per-task
            # SLEEP_UPSAMPLE / HAR_UPSAMPLE repeat hooks (HAR+Sleep-only compat).
            _mix = os.environ.get("PHASE2_COT_MIX", "").strip().lower()
            if mode == "train" and _mix == "balanced_all":
                # N-way balance: resize EVERY present CoT task (har/sleep/pamap2/tsqa/ecg)
                # to a common target count (PHASE2_BALANCE_TARGET, default 40000). Small
                # tasks are upsampled, ECG is truncated down — so the 5/6-task joint model
                # sees each dataset equally.
                _tgt = int(os.environ.get("PHASE2_BALANCE_TARGET", "40000"))
                if _har_samples:   _har_samples   = _resize_list(_har_samples, _tgt, seed)
                if _sleep_samples: _sleep_samples = _resize_list(_sleep_samples, _tgt, seed + 1)
                if _pamap_samples: _pamap_samples = _resize_list(_pamap_samples, _tgt, seed + 2)
                if _tsqa_samples:  _tsqa_samples  = _resize_list(_tsqa_samples, _tgt, seed + 3)
                if _timemqa_samples: _timemqa_samples = _resize_list(_timemqa_samples, _tgt, seed + 5)
                if self._ecg_proxy_rows:
                    self._ecg_proxy_rows = _resize_list(self._ecg_proxy_rows, _tgt, seed + 4)
                print(f"[Phase2DownstreamDataset] PHASE2_COT_MIX=balanced_all target={_tgt}: "
                      f"har={len(_har_samples)} sleep={len(_sleep_samples)} pamap2={len(_pamap_samples)} "
                      f"tsqa={len(_tsqa_samples)} timemqa={len(_timemqa_samples)} ecg={len(self._ecg_proxy_rows)}")
            elif mode == "train" and _mix in ("sqrt", "balanced") and self._ecg_proxy_rows:
                n_ecg = len(self._ecg_proxy_rows)
                n_har, n_slp = len(_har_samples), len(_sleep_samples)
                if _mix == "balanced":
                    t_har, t_slp = n_ecg, n_ecg
                else:  # sqrt-tempered: counts proportional to sqrt(size), ECG x1
                    t_har = round(n_ecg * (math.sqrt(n_har) / math.sqrt(n_ecg))) if n_har else 0
                    t_slp = round(n_ecg * (math.sqrt(n_slp) / math.sqrt(n_ecg))) if n_slp else 0
                _har_samples = _resize_list(_har_samples, t_har, seed)
                _sleep_samples = _resize_list(_sleep_samples, t_slp, seed + 1)
                print(
                    f"[Phase2DownstreamDataset] PHASE2_COT_MIX={_mix} (train): "
                    f"har {n_har}->{len(_har_samples)}, sleep {n_slp}->{len(_sleep_samples)}, "
                    f"ecg {n_ecg} (natural)"
                )
            else:
                _sup = int(os.environ.get("SLEEP_UPSAMPLE", "1")) if mode == "train" else 1
                _sup = max(1, _sup)
                if _sup > 1:
                    _n0 = len(_sleep_samples)
                    _sleep_samples = _sleep_samples * _sup
                    print(f"[Phase2DownstreamDataset] SLEEP_UPSAMPLE={_sup}: sleep {_n0}->{len(_sleep_samples)}")
                _hup = int(os.environ.get("HAR_UPSAMPLE", "1")) if mode == "train" else 1
                _hup = max(1, _hup)
                if _hup > 1:
                    _n0 = len(_har_samples)
                    _har_samples = _har_samples * _hup
                    print(f"[Phase2DownstreamDataset] HAR_UPSAMPLE={_hup}: har {_n0}->{len(_har_samples)}")

            self._cot_samples = _har_samples + _sleep_samples + _pamap_samples + _tsqa_samples + _timemqa_samples
            self.samples = (
                forecast_samples
                + self._chatts_samples
                + self._cot_samples
                + self._ecg_proxy_rows
            )

            # ChatTS-only seq_len cap (mirrors lazy branch). CoT samples
            # (har_cot/sleep_cot/ecg_cot) intrinsically have longer time series
            # and are kept; the OOM handler in curriculum_learning.py skips
            # individually too-large batches at runtime.
            MAX_SEQ_LEN = 600
            _n_before = len(self.samples)
            _dropped_by_type: Dict[str, int] = {}
            _kept: List[dict] = []
            for s in self.samples:
                t = s.get("type", "unknown")
                # Only drop oversized chatts and chronos; keep all CoT.
                if t in ("chatts", "chronos") and int(s.get("_seq_len", 0)) > MAX_SEQ_LEN:
                    key = t
                    if t == "chronos":
                        key = f"chronos/{s.get('_dataset', 'unknown')}"
                    _dropped_by_type[key] = _dropped_by_type.get(key, 0) + 1
                else:
                    _kept.append(s)
            self.samples = _kept
            n_dropped = _n_before - len(self.samples)
            if n_dropped > 0:
                print(
                    f"[Phase2DownstreamDataset] {mode}: dropped {n_dropped} samples "
                    f"with _seq_len > {MAX_SEQ_LEN} (would OOM as B=1 on V100-32GB):"
                )
                for k, v in sorted(_dropped_by_type.items()):
                    print(f"  - {k}: {v}")

            rng = random.Random(seed)
            rng.shuffle(self.samples)

            n_forecast = sum(1 for s in self.samples if s.get("type") == "chronos")
            n_chatts   = len(self.samples) - n_forecast
            print(
                f"[Phase2DownstreamDataset] {mode}: "
                f"total={len(self.samples)} "
                f"(forecast={n_forecast}, chatts={n_chatts})"
            )

            from collections import Counter
            ds_counts: Counter = Counter()
            for s in self.samples:
                if s.get("type") == "chronos":
                    ds_counts[s.get("_dataset", "unknown")] += 1
                else:
                    subset = s.get("_subset", "unknown")
                    ds_counts[f"ChatTS/{subset}"] += 1
            total = max(len(self.samples), 1)
            print(f"[Phase2DownstreamDataset] {mode} per-dataset breakdown:")
            for ds_key, cnt in sorted(ds_counts.items()):
                print(f"  {ds_key:<22s}: {cnt:7d}  ({100.0 * cnt / total:5.1f}%)")

    @property
    def rows(self):
        if self._lazy_entries is not None:
            return self._lazy_proxy_rows
        return self.samples

    def __len__(self) -> int:
        if self._lazy_entries is not None:
            return (
                len(self._lazy_entries) * self.windows_per_col_train
                + len(self._chatts_samples)
                + len(getattr(self, "_cot_samples", []))
                + len(getattr(self, "_ecg_proxy_rows", []))
            )
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._lazy_entries is not None:
            n_forecast = len(self._lazy_entries) * self.windows_per_col_train
            n_chatts   = len(self._chatts_samples)
            n_cot      = len(getattr(self, "_cot_samples", []))
            if idx < n_forecast:
                entry     = self._lazy_entries[idx % len(self._lazy_entries)]
                start     = random.randint(0, entry["max_start"])
                window_len = entry["history_len"] + entry["pred_len"]
                df_window  = entry["data"][start : start + window_len]
                return _make_sample(
                    df_window,
                    entry["col_names"],
                    entry["history_len"],
                    entry["pred_len"],
                    entry["ds_name"],
                    entry["eos_token"],
                    entry["ch_sep_token"],
                    entry["fp_tokens"],
                    entry["patch_size"],
                    entry["global_mean"],
                    entry["global_std"],
                )
            if idx < n_forecast + n_chatts:
                return self._chatts_samples[idx - n_forecast]
            if idx < n_forecast + n_chatts + n_cot:
                return self._cot_samples[idx - n_forecast - n_chatts]
            # ECG-QA via lazy adapter: format on demand from the raw HF Dataset.
            ecg_idx = idx - n_forecast - n_chatts - n_cot
            return self._ecg_adapter[ecg_idx]
        # Eager branch. ECG-QA proxies (type "ecg_cot") carry only sampler
        # metadata + _ecg_idx; fetch the real sample on demand from the lazy
        # adapter (IO-safe: signals preloaded at init). Resolves correctly even
        # after self.samples was shuffled, because the index is carried in-row.
        sample = self.samples[idx]
        if sample.get("type") == "ecg_cot" and self._ecg_adapter is not None:
            return self._ecg_adapter[sample["_ecg_idx"]]
        return sample


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
                _nm, _ns = s['time_series_norm_stats'][0]
                print(f"    norm_mean={_nm:.4f}  norm_std={_ns:.4f}")
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
