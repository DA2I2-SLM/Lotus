# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Worker-side preprocessor for OpenTSLM batches.

Modeled after HuggingFace `Processor` classes: bundles tokenization +
per-sample time-series prep (channel split, normalization, patch-aligned
padding, target shift, token-injection flags) into a single collate that
runs in DataLoader workers. The model's ``pad_and_apply_batch`` detects the
preprocessed batch and skips its CPU-bound steps, so the GPU rank only does
embedding lookup, encoder/projector forward, and the per-sample reassembly.

Outputs are wrapped in :class:`ProcessedBatch`, a ``list`` subclass that
keeps existing call sites (``for s in batch``, ``batch[i]["answer"]``,
``len(batch)``) working unchanged. The precomputed tensors live on
``batch.processed`` and survive ``pin_memory``/``DataLoader`` transport.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from opentslm.time_series_datasets.util import _sample_has_time_series


class ProcessedBatch(list):
    """List of original sample dicts plus a ``processed`` tensor bundle.

    Iterating / indexing behaves like the legacy ``list[dict]`` collate
    output, so loss code that reads ``s["answer"]`` or ``batch[i]["type"]``
    keeps working. ``pad_and_apply_batch`` checks ``isinstance(batch,
    ProcessedBatch)`` to take the fast path.
    """

    __slots__ = ("processed",)

    def __init__(self, samples: List[Dict[str, Any]], processed: Dict[str, Any]):
        super().__init__(samples)
        self.processed = processed

    def pin_memory(self) -> "ProcessedBatch":
        for s in self:
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    s[k] = v.pin_memory()
        new_processed = {}
        for k, v in self.processed.items():
            if torch.is_tensor(v):
                new_processed[k] = v.pin_memory()
            else:
                new_processed[k] = v
        self.processed = new_processed
        return self


class OpenTSLMProcessor:
    """Bundles tokenizer + per-sample TS prep into a single collate.

    Construct with the model's tokenizer (which already has the ``<ts>`` /
    ``</ts>`` special tokens) and patch size. The returned callable is the
    DataLoader's ``collate_fn``.

    The output mirrors the contract that ``pad_and_apply_batch`` needs
    internally; see :meth:`__call__` for the dict layout.
    """

    def __init__(
        self,
        tokenizer,
        patch_size: int,
        max_length: int = 4096,
        z_normalize_when_unnormed: bool = True,
    ):
        self.tokenizer = tokenizer
        self.patch_size = int(patch_size)
        self.max_length = int(max_length)
        self.z_normalize_when_unnormed = z_normalize_when_unnormed

    def __call__(self, batch: List[Dict[str, Any]]) -> ProcessedBatch:
        filtered: List[Dict[str, Any]] = [
            s for s in batch if _sample_has_time_series(s)
        ]
        if not filtered:
            return ProcessedBatch([], {"ts_present": False})

        # 1) Pre-tokenize all texts (pre_prompt + each ts_text + post_prompt).
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in filtered:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            text_ptrs.append((start, len(all_texts)))
            ts_counts.append(len(sample["time_series_text"]))

        tok = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # 2) Per-sample TS prep: split MV into channels, pad each channel to a
        #    multiple of patch_size, record real lengths and patch counts.
        ts_per_channel: List[torch.Tensor] = []  # each [T_pad, 1]
        ts_chan_id: List[int] = []
        ts_real_lens: List[int] = []
        ts_num_patches: List[int] = []
        ts_n_ch_per_outer: List[int] = []
        ts_norm_stats: List[Tuple[float, float]] = []
        needs_open_ts: List[bool] = []

        for sample in filtered:
            ts_lengths = sample.get("time_series_lengths")
            ts_normed = bool(sample.get("ts_already_normalized", False))
            sample_norm_stats = sample.get("time_series_norm_stats")
            ts_text_list = sample.get("time_series_text", [])

            for c, ts in enumerate(sample["time_series"]):
                ts = torch.as_tensor(ts, dtype=torch.float32)
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)  # [T, 1]
                T_in = ts.size(0)
                n_ch_inner = ts.size(-1)

                # Pad time dim to a multiple of patch_size (per-outer entry).
                pad_target = (
                    (T_in + self.patch_size - 1) // self.patch_size
                ) * self.patch_size
                if pad_target > T_in:
                    ts = F.pad(ts, (0, 0, 0, pad_target - T_in))

                ts_text_i = ts_text_list[c] if c < len(ts_text_list) else ""
                needs_open_ts.append("<ts>" not in str(ts_text_i))
                ts_n_ch_per_outer.append(n_ch_inner)

                if n_ch_inner == 1:
                    flat = ts.squeeze(-1)  # [T_pad]
                    if ts_lengths is not None and len(ts_lengths) > c:
                        lv = ts_lengths[c]
                        real_len = int(lv.item() if torch.is_tensor(lv) else lv)
                        real_len = max(0, min(real_len, flat.size(0)))
                    else:
                        real_len = T_in
                    ts_real_lens.append(real_len)
                    ts_num_patches.append(
                        (real_len + self.patch_size - 1) // self.patch_size
                    )

                    if ts_normed or not self.z_normalize_when_unnormed:
                        ts_proc = flat
                    else:
                        flat_real = flat[:real_len] if real_len > 0 else flat
                        m = flat_real.mean()
                        s = flat_real.std().clamp(min=1e-8)
                        ts_proc = (flat - m) / s

                    if sample_norm_stats is not None and len(sample_norm_stats) > c:
                        rm, rs = sample_norm_stats[c]
                        ts_norm_stats.append((float(rm), float(rs)))
                    elif (
                        ts_normed
                        and len(sample["time_series"]) == 1
                        and "norm_mean" in sample
                        and "norm_std" in sample
                    ):
                        ts_norm_stats.append(
                            (float(sample["norm_mean"]), float(sample["norm_std"]))
                        )
                    else:
                        flat_real = flat[:real_len] if real_len > 0 else flat
                        ts_norm_stats.append(
                            (
                                float(flat_real.mean().item()),
                                float(flat_real.std().clamp(min=1e-8).item()),
                            )
                        )

                    ts_per_channel.append(ts_proc.unsqueeze(-1))
                    ts_chan_id.append(c)
                else:
                    real_len_mv = T_in
                    for ch in range(n_ch_inner):
                        flat_ch = ts[:, ch]
                        ts_real_lens.append(real_len_mv)
                        ts_num_patches.append(
                            (real_len_mv + self.patch_size - 1) // self.patch_size
                        )
                        if ts_normed or not self.z_normalize_when_unnormed:
                            ts_proc_ch = flat_ch
                        else:
                            flat_real = (
                                flat_ch[:real_len_mv] if real_len_mv > 0 else flat_ch
                            )
                            m = flat_real.mean()
                            s = flat_real.std().clamp(min=1e-8)
                            ts_proc_ch = (flat_ch - m) / s

                        if sample_norm_stats is not None and len(sample_norm_stats) > ch:
                            rm, rs = sample_norm_stats[ch]
                            ts_norm_stats.append((float(rm), float(rs)))
                        else:
                            flat_real = (
                                flat_ch[:real_len_mv] if real_len_mv > 0 else flat_ch
                            )
                            ts_norm_stats.append(
                                (
                                    float(flat_real.mean().item()),
                                    float(flat_real.std().clamp(min=1e-8).item()),
                                )
                            )
                        ts_per_channel.append(ts_proc_ch.unsqueeze(-1))
                        ts_chan_id.append(ch)

        # 3) Pad TS time-dim to common T_max (already a multiple of patch_size
        #    because every entry was rounded up individually).
        ts_padded = pad_sequence(ts_per_channel, batch_first=True)
        T_max = ts_padded.size(1)
        rem = T_max % self.patch_size
        if rem:
            pad_len = self.patch_size - rem
            pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
            ts_padded = torch.cat([ts_padded, pad], dim=1)

        n_ch_batch = (max(ts_chan_id) + 1) if ts_chan_id else 1

        processed: Dict[str, Any] = {
            "ts_present": True,
            "input_ids": tok.input_ids,
            "text_attn_mask": tok.attention_mask,
            "text_ptrs": text_ptrs,
            "ts_counts": ts_counts,
            "ts_padded": ts_padded,
            "ts_chan_id": torch.tensor(ts_chan_id, dtype=torch.long),
            "ts_real_lens": torch.tensor(ts_real_lens, dtype=torch.long),
            "ts_num_patches": torch.tensor(ts_num_patches, dtype=torch.long),
            "ts_n_ch_per_outer": ts_n_ch_per_outer,
            "needs_open_ts": needs_open_ts,
            "ts_norm_stats": ts_norm_stats,
            "n_ch_batch": int(n_ch_batch),
        }

        return ProcessedBatch(filtered, processed)

