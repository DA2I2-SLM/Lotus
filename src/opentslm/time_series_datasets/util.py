# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from typing import List
from opentslm.model_config import PATCH_SIZE

import torch.nn.functional as F

import torch

MAX_VALUE = 50_000
MIN_VALUE = -MAX_VALUE


def _sample_has_time_series(element) -> bool:
    ts_list = element.get("time_series", [])
    if isinstance(ts_list, torch.Tensor):
        ts_iter = [ts_list]
    else:
        ts_iter = list(ts_list)

    for ts in ts_iter:
        try:
            ts_tensor = torch.as_tensor(ts, dtype=torch.float32)
        except Exception:
            continue
        if ts_tensor.numel() > 0 and ts_tensor.shape[0] > 0:
            return True
    return False


def _summarize_missing_ts_example(element, max_chars: int = 2000) -> str:
    summary = {
        "type": element.get("type", "unknown"),
        "keys": sorted(element.keys()),
        "pre_prompt": element.get("pre_prompt", ""),
        "post_prompt": element.get("post_prompt", ""),
        "answer": element.get("answer", ""),
        "time_series_text": element.get("time_series_text", []),
        "time_series": element.get("time_series", []),
        "_raw_idx": element.get("_raw_idx"),
        "_chronos_split": element.get("_chronos_split"),
    }
    text = repr(summary)
    if len(text) > max_chars:
        text = text[:max_chars] + "...<truncated>"
    return text


def extend_time_series_to_match_patch_size_and_aggregate(
    batch, *, patch_size: int = PATCH_SIZE, normalize: bool = False
):
    """
    Pad variable-length series so each sample length is a multiple of *patch_size*.
    Optionally normalize each time series to have zero mean and unit variance.
    """

    filtered_batch = []
    for idx, element in enumerate(batch):
        if not _sample_has_time_series(element):
            print(
                f"[WARN][collate] Skipping sample with no time_series at batch index {idx}: "
                f"{_summarize_missing_ts_example(element)}",
                flush=True,
            )
            continue
        filtered_batch.append(element)

    batch = filtered_batch
    if not batch:
        return []

    for element in batch:
        # 1) pull out the list of (1D) time‑series
        ts_list = element["time_series"]

        # 2) convert each to a torch.Tensor (float)
        ts_tensors = [torch.as_tensor(ts, dtype=torch.float32) for ts in ts_list]
        orig_lengths = [int(ts.size(0)) for ts in ts_tensors]

        # 3) normalize each time series if requested
        if normalize:
            normalized_tensors = []
            for ts in ts_tensors:
                mean = ts.mean()
                std = ts.std()
                if std > 1e-8:  # Avoid division by zero
                    ts_normalized = (ts - mean) / std
                else:
                    ts_normalized = ts - mean
                normalized_tensors.append(ts_normalized)
            ts_tensors = normalized_tensors

        # 4) find the longest series length
        max_len = max([ts.size(0) for ts in ts_tensors])

        # 5) round up to nearest multiple of patch_size
        padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size

        # 6) pad (or trim) each series to padded_len
        padded = []
        for ts in ts_tensors:
            L = ts.size(0)
            if L < padded_len:
                pad_amt = padded_len - L
                ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
            else:
                ts = ts[:padded_len]
            padded.append(ts)

        # 7) stack into a single 2D tensor: (num_series, padded_len)
        element["time_series"] = torch.stack(padded, dim=0)
        element["time_series_lengths"] = torch.tensor(orig_lengths, dtype=torch.long)

    return batch
