#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Autoregressive forecasting evaluation for OpenTSLM.

Two modes:
  --mode autoregressive  (default)
      Runs model.generate() with n_forecast_patches. Each predicted patch is
      fed back as input to produce the next patch — identical to inference.
      Metrics: MSE / MAE on the full forecast horizon.

  --mode teacher_forced
      Feeds all ground-truth history to the LLM in one forward pass, then
      reads hidden states at each TS position and decodes via the MLP head.
      Measures decoder quality; not a realistic forecasting benchmark.

Usage:
  python fixed_inference.py --ckpt /path/best_model.pt --llm_id meta-llama/Llama-3.2-1B
  python fixed_inference.py --ckpt /path/best_model.pt --llm_id ... --mode teacher_forced

Fix (vs inference.py):
  _generate_patches_direct for the "llm" backbone now correctly traverses all
  three model components — encoder, LLM, decoder — for every generated patch.
  Previously the LLM backbone path raised NotImplementedError after building the
  initial KV-cache; now it runs a proper autoregressive loop:
    predicted patch → encoder (incremental KV-cache) → projector
    → LLM (single-token KV-cache step) → decoder_projector
    → decoder history buffer → decoder → next patch
"""

import argparse
import json
import os
import re
import random
from types import SimpleNamespace
from typing import List

import pandas as pd

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.ChronosDataset import (
    BASE_DATA_PATH,
    PREPROCESSED_SPLITS_DIR,
    ChronosDataset,
    _last_value_impute,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


# ── Dataset helpers (unchanged from original) ─────────────────────────────────

_SEEN_PLOT_WARNINGS = set()


def _warn_plot_failure(plot_kind: str, out_dir: str, exc: Exception):
    """Warn once per unique plotting failure instead of failing silently."""
    key = (plot_kind, type(exc).__name__, str(exc), out_dir)
    if key in _SEEN_PLOT_WARNINGS:
        return
    _SEEN_PLOT_WARNINGS.add(key)
    print(
        f"[plot warning] Could not save {plot_kind} plot to {out_dir}: "
        f"{type(exc).__name__}: {exc}"
    )


def _decode_ts_patch_predictions(
    model,
    last_hidden,
    ts_mask,
    ts_patch_targets_shifted,
    ts_patch_valid,
    ts_chan_id=None,
):
    """
    Decode Chronos patch predictions exactly like the joint phase-1 MSE branch.

    This mirrors the Chronos section of ``compute_joint_phase1_loss``:
      last_hidden at TS positions
      -> decoder_projector
      -> layer_norm
      -> pad_sequence(split(..., patch_counts))
      -> decoder

    That is the path trained by ``joint_phase1_extend`` checkpoints.
    """
    _, _, _ = ts_patch_targets_shifted.shape
    ts_hidden = last_hidden[ts_mask]

    patch_counts = (ts_patch_valid.sum(dim=1) + 1).tolist()
    expected_flat = int(sum(patch_counts))
    if ts_hidden.size(0) != expected_flat:
        raise RuntimeError(
            f"ts_mask extracted {ts_hidden.size(0)} hidden vectors, "
            f"expected {expected_flat} real patch vectors."
        )

    # For multivariate samples the LLM sequence carries TS tokens in time-major
    # order ([t0c0, t0c1, …, t0c_{n-1}, t1c0, …]) while the decoder / targets
    # expect channel-first order ([c0_t0, c0_t1, …, c1_t0, …]).  Mirror the
    # reorder done in compute_joint_phase1_loss so the split-by-patch_counts
    # below yields one contiguous per-channel sequence per chunk.
    n_ch = getattr(model, "_last_n_ch_chronos", 1)
    if n_ch > 1:
        B = len(patch_counts) // n_ch
        assert B * n_ch == len(patch_counts), (
            f"patch_counts length {len(patch_counts)} not divisible by n_ch={n_ch}"
        )
        T_per_sample = [patch_counts[j * n_ch] for j in range(B)]
        reordered_parts = []
        offset = 0
        for T_j in T_per_sample:
            chunk = ts_hidden[offset: offset + T_j * n_ch]  # [T*C, H] time-major
            reordered_parts.append(
                chunk.reshape(T_j, n_ch, -1)
                     .permute(1, 0, 2)
                     .reshape(T_j * n_ch, -1)
            )
            offset += T_j * n_ch
        ts_hidden = torch.cat(reordered_parts, dim=0)

    h_enc = model.decoder_projector(ts_hidden.float())
    h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
    h_enc_3d = pad_sequence(
        list(torch.split(h_enc, patch_counts)),
        batch_first=True,
    )

    return model.decoder(h_enc_3d, ts_chan_id)


def _decode_ts_patch_predictions_foundational(model):
    """
    Decode next-patch predictions from the direct foundational encoder/decoder path.

    Assumes pad_and_apply_batch(..., return_targets=True) has already populated
    the model's cached normalized history series.
    """
    return model.decoder(model.build_foundational_history_hidden())


class ChronosOnlySubset(Dataset):
    """Wraps Phase1JointDataset / Phase1JointExtendDataset test split,
    exposing only the Chronos samples for forecasting evaluation."""

    def __init__(self, joint_ds):
        self._indices = [
            i for i, row in enumerate(joint_ds.rows)
            if isinstance(row, dict) and row.get("type") == "chronos"
        ]
        if not self._indices:
            raise RuntimeError("No chronos stubs found in the joint dataset test split.")
        self._joint_ds = joint_ds
        print(f"[ChronosOnlySubset] {len(self._indices)} chronos samples "
              f"(out of {len(joint_ds.rows)} total).")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._joint_ds[self._indices[idx]]


TEST_FRAC = 0.1
VAL_FRAC  = 0.1
TRAIN_ALIGNED_HISTORY_LENS = [96, 192, 384, 416]
TRAIN_ALIGNED_PRED_LEN     = 96

PHASE1_EVAL_HISTORY_LENS = [96, 192, 336, 512]
PHASE1_EVAL_PRED_LEN     = 96


class Phase1PretrainEvalDataset(ChronosDataset):
    """Chronos eval dataset whose prompt format exactly matches Phase1JointDataset.

    Differences from TrainingAlignedChronosEvalDataset:
      - Pre/post-prompt are empty / EOS — identical to training chronos rows.
      - TS text includes original mean & std:
            "This is the time series, it has mean X and std Y. <ts>"
      - Answer is EOS only — no future values in the LLM context, which
        would be a data leak during autoregressive eval.
      - ts_future is attached as a tensor field for run_autoregressive.
      - Loads the shared training Chronos split (PREPROCESSED_SPLITS_DIR/all)
        so the test pool is the held-out 10 % of the training corpus.
    """

    CACHE_SUFFIX = "_phase1_pretrain_eval"

    def __init__(
        self,
        mode: str,
        EOS_TOKEN: str,
        patch_size: int = 16,
        pred_len: int = PHASE1_EVAL_PRED_LEN,
        history_lens: List[int] | None = None,
        source_filter: str | None = None,
    ):
        self.patch_size  = int(patch_size)
        self.pred_len    = int(pred_len)
        self.history_lens = [int(h) for h in (history_lens or PHASE1_EVAL_HISTORY_LENS)]
        self.valid_history_lens = [
            h for h in self.history_lens if h > 0 and h % self.patch_size == 0
        ]
        if not self.valid_history_lens:
            raise ValueError(
                f"No valid history lengths for patch_size={self.patch_size}: {self.history_lens}"
            )
        if pred_len % self.patch_size != 0:
            raise ValueError(
                f"pred_len={pred_len} must be divisible by patch_size={self.patch_size}"
            )

        # Optional per-source filter — "chronos_uv" or "synthetic" reads only
        # rows tagged with that source in the merged "all" split (set by
        # Phase1JointDataset._load_or_build_chronos_raw at train time).
        # None / "all" / "both" keep the legacy mixed behaviour.
        sf = (source_filter or "").strip().lower() or None
        if sf in ("", "all", "both", "none"):
            sf = None
        if sf is not None and sf not in ("chronos_uv", "synthetic"):
            raise ValueError(
                f"source_filter must be one of None/'all'/'chronos_uv'/'synthetic', "
                f"got {source_filter!r}"
            )
        self.source_filter = sf

        cache_key = (
            self.patch_size, self.pred_len, tuple(self.valid_history_lens),
            self.source_filter,
        )
        if getattr(self.__class__, "_cache_key", None) != cache_key:
            for attr in ("loaded", "_train_rows", "_val_rows", "_test_rows"):
                if hasattr(self.__class__, attr):
                    delattr(self.__class__, attr)
            self.__class__._cache_key = cache_key

        super().__init__(
            mode=mode,
            EOS_TOKEN=EOS_TOKEN,
            dataset_name="all",
            max_seq_len=max(self.valid_history_lens) + self.pred_len,
        )

    def _load_splits(self):
        """Load the same held-out split built during training, filtering short series."""
        split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all")
        if not os.path.exists(os.path.join(split_dir, "_DONE")):
            raise RuntimeError(
                f"Training Chronos splits not found at {split_dir}. "
                "Run Phase1JointDataset first to build them."
            )

        # source_filter fast path: prefer the per-source caches written by
        # Phase1JointDataset._load_or_build_chronos_raw (gated by
        # _DONE_PER_SOURCE). Skips the ~80-min row-level Python filter.
        if self.source_filter is not None:
            want = self.source_filter
            per_src_done = os.path.join(split_dir, "_DONE_PER_SOURCE")
            per_src_paths = {
                name: os.path.join(split_dir, f"{name}_{want}")
                for name in ("train", "val", "test")
            }
            if os.path.exists(per_src_done) and all(
                os.path.isdir(p) for p in per_src_paths.values()
            ):
                train = load_from_disk(per_src_paths["train"], keep_in_memory=False)
                val   = load_from_disk(per_src_paths["val"],   keep_in_memory=False)
                test  = load_from_disk(per_src_paths["test"],  keep_in_memory=False)
                print(
                    f"[Phase1PretrainEvalDataset] source_filter={want!r}: "
                    f"loaded per-source caches "
                    f"(train={len(train)}, val={len(val)}, test={len(test)})"
                )
                return train, val, test

            # Fallback: combined splits + vectorised numpy mask + ds.select().
            # Used when the build hasn't backfilled per-source dirs yet
            # (older cache, or first eval after the fix landed). ~seconds for
            # 600k rows — Arrow native, no per-row Python callback.
            # Also writes the per-source subdirs so the next eval skips this.
            print(
                f"[Phase1PretrainEvalDataset] source_filter={want!r}: per-source "
                f"caches not found under {split_dir}; building inline "
                f"(numpy mask + select — fast)."
            )
            import numpy as _np
            train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=False)
            val   = load_from_disk(os.path.join(split_dir, "val"),   keep_in_memory=False)
            test  = load_from_disk(os.path.join(split_dir, "test"),  keep_in_memory=False)
            kept = []
            for name, ds in (("train", train), ("val", val), ("test", test)):
                if "source" not in ds.column_names:
                    raise RuntimeError(
                        f"Split '{name}' at {split_dir} has no 'source' column; "
                        "rebuild the chronos split cache with the updated "
                        "Phase1JointDataset to tag chronos_uv vs synthetic."
                    )
                _sources = _np.asarray(ds["source"], dtype=object)
                _idx = _np.nonzero(_sources == want)[0]
                ds_f = ds.select(_idx.tolist(), keep_in_memory=True) if _idx.size else ds.select([])
                kept.append(ds_f)
                # Best-effort write-through so subsequent evals load directly.
                _out = os.path.join(split_dir, f"{name}_{want}")
                if not os.path.isdir(_out) and len(ds_f) > 0:
                    try:
                        ds_f.save_to_disk(_out)
                        print(
                            f"[Phase1PretrainEvalDataset]   wrote per-source "
                            f"cache: {_out} ({len(ds_f)} rows)"
                        )
                    except Exception as _exc:
                        print(
                            f"[Phase1PretrainEvalDataset]   could not save "
                            f"{_out}: {_exc!r} (skipping cache write, eval "
                            "continues normally)"
                        )
            print(
                f"[Phase1PretrainEvalDataset] source_filter={want!r}: "
                f"train={len(train)}->{len(kept[0])}, "
                f"val={len(val)}->{len(kept[1])}, "
                f"test={len(test)}->{len(kept[2])}"
            )
            return tuple(kept)

        # No filter: load combined splits as before.
        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=False)
        val   = load_from_disk(os.path.join(split_dir, "val"),   keep_in_memory=False)
        test  = load_from_disk(os.path.join(split_dir, "test"),  keep_in_memory=False)

        return train, val, test

    def _process_row(self, row):
        """Return (history_norm, future_norm, n_hist_patches, n_fut_patches, mean, std).

        Mirrors Phase1JointDataset.Phase1ChronosFuturePatchDataset
        ``_normalize_history_and_future`` (per-channel z-norm + std_floor 1e-3
        + post-norm clip ±20). The clamp/clip MUST match training; otherwise
        synthetic samples with near-constant channels get normalised to huge
        values at eval that the model never saw during training, and per-channel
        info is destroyed by the OOD inputs.

        Always uses the last valid window (deterministic) regardless of split —
        this is an eval-only class; randomness is wrong here even for train split.
        """
        # Match Phase1ChronosFuturePatchDataset's defaults so eval input
        # distribution = training input distribution. These are the values
        # written by every existing train script (no CLI override sets them).
        _STD_FLOOR = 1e-3
        _CLIP      = 20.0
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = series.shape[0]  # works for both [T] and [T, C]

        candidates = [h for h in self.valid_history_lens if T >= h + self.pred_len]
        if not candidates:
            raise ValueError(
                f"Series too short (len={T}); need >= "
                f"{min(self.valid_history_lens) + self.pred_len}"
            )

        history_len   = max(candidates)
        window_needed = history_len + self.pred_len
        start         = T - window_needed   # last window — deterministic for all splits

        window  = _last_value_impute(series[start : start + window_needed])
        history = window[:history_len]
        future  = window[history_len:]

        if history.dim() == 1:
            mean = history.mean()
            std  = history.std().clamp(min=_STD_FLOOR)
            history_norm = (history - mean) / std
            future_norm  = (future  - mean) / std
            mean_scalar  = mean.item()
            std_scalar   = std.item()
        else:
            mean = history.mean(dim=0, keepdim=True)            # [1, C]
            std  = history.std(dim=0, keepdim=True).clamp(min=_STD_FLOOR)
            history_norm = (history - mean) / std
            future_norm  = (future  - mean) / std
            mean_scalar  = mean[0, 0].item()
            std_scalar   = std[0, 0].item()

        # Post-norm clip to ±20, matching training's clamp at
        # Phase1JointDataset._normalize_history_and_future:1836-1838. Without
        # this, near-constant synthetic channels yield normalised values in the
        # hundreds at eval, far outside the model's training distribution.
        history_norm = history_norm.clamp(-_CLIP, _CLIP)
        future_norm  = future_norm.clamp(-_CLIP, _CLIP)

        return (
            history_norm,
            future_norm,
            history_len // self.patch_size,
            self.pred_len // self.patch_size,
            mean_scalar,
            std_scalar,
        )

    # Prompt format matches Phase1JointDataset chronos rows exactly.

    def _get_pre_prompt(self, row) -> str:
        return ""

    def _get_post_prompt(self, row) -> str:
        return self.EOS_TOKEN

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        history_norm, _, _, _, mean, std = self._process_row(row)
        n_channels = history_norm.shape[1] if history_norm.dim() == 2 else 1
        # Match training prompt semantics: n_timesteps = total window being
        # modeled (history + forecast horizon). Training emits the full window
        # length (e.g. 2048); eval must do the same so the model sees a
        # length-token from the same distribution.
        n_timesteps = history_norm.shape[0] + self.pred_len
        modality = "multivariate" if n_channels > 1 else "univariate"
        ch_word = "channels" if n_channels > 1 else "channel"
        prompt_text = (
            f"This is a {modality} time series with {n_channels} {ch_word} "
            f"and {n_timesteps} time steps. "
            f"It has mean {mean:.4f} and std {std:.4f}. <ts>"
        )
        return [TextTimeSeriesPrompt(prompt_text, history_norm.numpy())]

    def _get_answer(self, row) -> str:
        # EOS only — future values must NOT be in the LLM context (data leak).
        # ts_future is attached separately in __getitem__.
        return self.EOS_TOKEN

    def __getitem__(self, idx):
        try:
            sample = super().__getitem__(idx)
            row = self._get_split()[idx]
            history_norm, future_norm, _, _, mean, std = self._process_row(row)
            sample["ts_already_normalized"] = True
            # ts_future canonicalisation for downstream eval MSE compare:
            #   UV (chronos_univariate=True path produces [P, 1]) → squeeze to [P]
            #     so the processor/model's UV splice check (`dim()==1`) passes.
            #   MV → keep [P, C] (time-first); run_autoregressive does `gt_flat.t()`
            #     to get [C, P] for MSE comparison. DO NOT transpose here.
            #   The training-side splice expects [C, P] for MV, but at eval the
            #   splice's output is unused (no backward pass) — the harmless
            #   mismatch keeps the eval MSE computation correct.
            if future_norm.dim() == 2 and future_norm.size(1) == 1:
                sample["ts_future"] = future_norm.squeeze(-1)            # [P, 1] → [P]
            else:
                sample["ts_future"] = future_norm                          # [P] or [P, C]
            # For multivariate [L, C] series, channel-0 stats are tracked here;
            # downstream pad_and_apply_batch derives per-channel stats as needed
            # (matches Phase1JointDataset._format_sample contract).
            sample["time_series_norm_stats"] = [(mean, std)]
            return sample
        except ValueError:
            return None

    def _get_split(self):
        if self.mode == "train":
            return self._train_rows
        elif self.mode == "val":
            return self._val_rows
        return self._test_rows


class TrainingAlignedChronosEvalDataset(ChronosDataset):
    """Chronos eval dataset that mirrors the joint-training forecasting prompt path.

    Each sample exposes:
      time_series  — normalized history patches (fed to encoder)
      ts_future    — normalized future values  (ground truth for MSE)
      answer       — 'Answer: <ts> v1 v2 ... </ts>' (for teacher-forced mode)
    """

    CACHE_SUFFIX = "_train_aligned_eval"

    def __init__(
        self,
        mode: str,
        EOS_TOKEN: str,
        dataset_name: str = "all",
        base_path: str = BASE_DATA_PATH,
        patch_size: int = 4,
        pred_len: int = TRAIN_ALIGNED_PRED_LEN,
        history_lens: List[int] | None = None,
    ):
        self.patch_size   = int(patch_size)
        self.pred_len     = int(pred_len)
        self.history_lens = [int(h) for h in (history_lens or TRAIN_ALIGNED_HISTORY_LENS)]
        self.valid_history_lens = [
            h for h in self.history_lens if h > 0 and h % self.patch_size == 0
        ]
        if not self.valid_history_lens:
            raise ValueError(
                f"No valid history lengths for patch_size={self.patch_size}: {self.history_lens}"
            )

        cache_key = (
            dataset_name, base_path,
            self.patch_size, self.pred_len,
            tuple(self.valid_history_lens),
        )
        if getattr(self.__class__, "_cache_key", None) != cache_key:
            for attr in ("loaded", "_train_rows", "_val_rows", "_test_rows"):
                if hasattr(self.__class__, attr):
                    delattr(self.__class__, attr)
            self.__class__._cache_key = cache_key

        super().__init__(
            mode=mode,
            EOS_TOKEN=EOS_TOKEN,
            dataset_name=dataset_name,
            base_path=base_path,
            max_seq_len=max(self.valid_history_lens) + self.pred_len,
        )

    def _load_splits(self):
        training_split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all")
        done_flag = os.path.join(training_split_dir, "_DONE")
        if not os.path.exists(done_flag):
            raise RuntimeError(
                f"Training Chronos splits not found at {training_split_dir}. "
                "Run ChronosDataset with dataset_name='all' first."
            )
        train = load_from_disk(os.path.join(training_split_dir, "train"), keep_in_memory=False)
        val   = load_from_disk(os.path.join(training_split_dir, "val"),   keep_in_memory=False)
        test  = load_from_disk(os.path.join(training_split_dir, "test"),  keep_in_memory=False)
        return train, val, test

    def _process_row(self, row):
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = len(series)

        candidates = [
            h for h in self.valid_history_lens if T >= (h + self.pred_len)
        ]
        if not candidates:
            raise ValueError(
                f"Series too short: len={T}, "
                f"need >= {min(self.valid_history_lens) + self.pred_len}"
            )

        history_len   = max(candidates)
        window_needed = history_len + self.pred_len
        start         = T - window_needed   # last window — deterministic for all splits

        window      = _last_value_impute(series[start : start + window_needed])
        history     = window[:history_len]
        future      = window[history_len:]

        mean = history.mean()
        std  = history.std().clamp(min=1e-8)
        history_norm = (history - mean) / std
        future_norm  = (future  - mean) / std

        return (
            history_norm, future_norm,
            history_len // self.patch_size, self.pred_len // self.patch_size,
            mean, std,
        )

    def _get_answer(self, row) -> str:
        _, future_norm, _, _, _, _ = self._process_row(row)
        future_str = " ".join(f"{v:.4f}" for v in future_norm.tolist())
        return f"Answer: <ts> {future_str} </ts>"

    def _get_pre_prompt(self, row) -> str:
        _, _, n_hist_patches, _, _, _ = self._process_row(row)
        return f"Given {n_hist_patches} time series patches"

    def _get_post_prompt(self, row) -> str:
        _, _, _, n_fut_patches, _, _ = self._process_row(row)
        return f"</ts>, please predict the next {n_fut_patches} patches"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        history_norm, _, _, _, mean, std = self._process_row(row)
        # This dataset is always univariate (single column from a Monash-style series).
        # Prompt mirrors Phase1JointDataset chronos branch (mean+std included) so
        # train/eval distribution of prompt text matches.
        n_timesteps = history_norm.shape[0] + self.pred_len
        prompt_text = (
            f"This is a univariate time series with 1 channel "
            f"and {n_timesteps} time steps. "
            f"It has mean {float(mean):.4f} and std {float(std):.4f}. <ts>"
        )
        return [TextTimeSeriesPrompt(prompt_text, history_norm.tolist())]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Attach ts_future so both evaluation modes can access it without
        # parsing the answer string.
        row = self._get_split()[idx]
        _, future_norm, _, _, _, _ = self._process_row(row)
        sample["ts_already_normalized"] = True
        # ts_future canonicalisation (see Phase1PretrainEvalDataset): only the
        # UV [P, 1] case is reshaped to [P]; MV [P, C] passes through unchanged
        # because run_autoregressive expects time-first for `gt_flat.t()`.
        if future_norm.dim() == 2 and future_norm.size(1) == 1:
            sample["ts_future"] = future_norm.squeeze(-1)
        else:
            sample["ts_future"] = future_norm
        return sample

    def _get_split(self):
        if self.mode == "train":
            return self._train_rows
        elif self.mode == "val":
            return self._val_rows
        return self._test_rows


class ETTh1EvalDataset(Dataset):
    """
    ETTh1 test-set evaluation dataset for standard forecasting benchmarking.

    Loads ETTh1.csv, applies a 60/20/20 train/val/test split, and yields
    sliding windows of (history_len + pred_len) from each of the 7 target
    columns.  Each window uses history-only z-normalization — the standard
    in TS papers (analogous to ReVIN: normalize with history stats, apply the
    same shift/scale to the future, compute MSE in normalized space).

    The sample dict is compatible with extend_time_series_to_match_patch_size_and_aggregate
    and run_autoregressive.
    """

    TARGET_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    def __init__(
        self,
        csv_path: str,
        mode: str = "test",
        patch_size: int = 16,
        pred_len: int = 96,
        history_len: int = 480,
        stride: int = 1,
        split_ratios: tuple = (0.6, 0.2, 0.2),
        EOS_TOKEN: str = "",
    ):
        self.patch_size  = patch_size
        self.pred_len    = pred_len
        self.history_len = history_len
        self.EOS_TOKEN   = EOS_TOKEN

        if history_len % patch_size != 0:
            raise ValueError(
                f"history_len={history_len} must be divisible by patch_size={patch_size}"
            )
        if pred_len % patch_size != 0:
            raise ValueError(
                f"pred_len={pred_len} must be divisible by patch_size={patch_size}"
            )
        self.scaler = StandardScaler()

        df = pd.read_csv(csv_path)
        N  = len(df)
        train_end = int(N * split_ratios[0])
        val_end   = int(N * (split_ratios[0] + split_ratios[1]))
        # self.scaler.fit(df.iloc[:train_end].values)
        if mode == "train":
            df_split = df.iloc[:train_end]
            
        elif mode == "val":
            df_split = df.iloc[train_end:val_end]
        else:
            df_split = df.iloc[val_end:]
        
        # df_split = self.scaler.transform(df_split.values)
        

        window_len = history_len + pred_len

        # Keep only columns that exist in the CSV
        avail_cols = [c for c in self.TARGET_COLS if c in df_split.columns]

        # Build a [T, C] tensor covering all available channels
        all_series = torch.stack(
            [torch.tensor(df_split[c].values, dtype=torch.float32) for c in avail_cols],
            dim=1,
        )   # [T, C]
        T = all_series.size(0)

        # One sample per sliding window — all channels together
        self.samples: List[int] = []
        for start in range(0, T - window_len + 1, stride):
            self.samples.append(start)

        self._all_series  = all_series
        self._avail_cols  = avail_cols

        print(
            f"[ETTh1EvalDataset] mode={mode} split_rows={len(df_split)} | "
            f"{len(self.samples)} windows "
            f"(history={history_len} pred={pred_len} stride={stride}) "
            f"channels={avail_cols}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start   = self.samples[idx]
        window  = self._all_series[start : start + self.history_len + self.pred_len]  # [T+P, C]
        history = window[: self.history_len]   # [T, C]
        future  = window[self.history_len :]   # [P, C]

        # Per-channel history-only z-normalization (standard TS practice)
        mean = history.mean(dim=0)             # [C]
        std  = history.std(dim=0).clamp(min=1e-8)
        history_norm = (history - mean) / std  # [T, C]
        future_norm  = (future  - mean) / std  # [P, C]

        # Pass as a single [L, C] multivariate tensor so pad_and_apply_batch
        # uses the MV path (cross-channel attention + time-major interleave),
        # matching how Chronos MV data is laid out during training.
        n_ch = history_norm.size(1)
        n_timesteps = self.history_len + self.pred_len
        modality = "multivariate" if n_ch > 1 else "univariate"
        ch_word = "channels" if n_ch > 1 else "channel"
        prompt_text = (
            f"This is a {modality} time series with {n_ch} {ch_word} "
            f"and {n_timesteps} time steps. <ts>"
        )

        return {
            "pre_prompt":            "ETTh1",
            "time_series":           [history_norm],       # single [L, C] tensor
            "time_series_text":      [prompt_text],        # single prompt
            "post_prompt":           self.EOS_TOKEN,
            "answer":               self.EOS_TOKEN,
            "ts_future":             future_norm,          # [P, C]
            "ts_already_normalized": True,
        }


# ── Autoregressive evaluation ─────────────────────────────────────────────────

def _parse_future_from_answer(answer: str) -> torch.Tensor | None:
    """Extract float values from 'Answer: <ts> v1 v2 ... </ts>'."""
    m = re.search(r'<ts>(.*?)</ts>', answer, re.DOTALL)
    if not m:
        return None
    try:
        vals = [float(v) for v in m.group(1).strip().split()]
        return torch.tensor(vals, dtype=torch.float32)
    except ValueError:
        return None


@torch.no_grad()
def _generate_patches_direct(model, batch, n_forecast_patches, device, forecast_backbone="llm"):
    """Autoregressive patch-by-patch forecasting via the full encoder → LLM → decoder pipeline.

    Foundational backbone: re-encodes the growing raw sequence from scratch each step.

    LLM backbone: same strategy — at every step we append the predicted patch to the
    raw history and re-run the full encoder + LLM + decoder. This mirrors the
    training data flow exactly (cross-channel block-causal encoder attention and
    time-major LLM interleaving across channels), which the previous incremental
    KV-cache path broke for multivariate samples.
    """
    if forecast_backbone in {"timesfm", "foundational"}:
        model.pad_and_apply_batch(batch, return_targets=True)
        if model._last_ts_raw is None or model._last_ts_chan_id is None:
            raise RuntimeError(
                "Foundational rollout requires cached raw history tensors. "
                "pad_and_apply_batch(..., return_targets=True) did not populate them."
            )

        cur_raw = model._last_ts_raw
        chan_id = model._last_ts_chan_id
        preds = []
        for _ in range(n_forecast_patches):
            history_hidden = model._build_foundational_history_hidden_from_raw(
                cur_raw,
                chan_id,
            )
            pred_step = model.decoder(history_hidden)[:, -1, :]
            preds.append(pred_step)
            cur_raw = torch.cat([cur_raw, pred_step.to(dtype=cur_raw.dtype)], dim=1)

        return torch.stack(preds, dim=1)

    # ── LLM backbone: re-run full encoder + LLM + decoder each step ─────────────
    B = len(batch)

    # Shallow-copy each sample dict so we can extend its time_series in place
    # without mutating the caller's data. TS entries are materialised as cloned
    # tensors on the target device so we can torch.cat predicted patches onto them.
    work_batch = []
    for s in batch:
        s2 = dict(s)
        ts_entries = []
        for t in s2["time_series"]:
            if isinstance(t, torch.Tensor):
                ts_entries.append(t.detach().clone().to(device))
            else:
                ts_entries.append(torch.as_tensor(t, dtype=torch.float32, device=device))
        s2["time_series"] = ts_entries
        lens = s2.get("time_series_lengths")
        if lens is not None:
            s2["time_series_lengths"] = [
                int(l.item() if torch.is_tensor(l) else l) for l in lens
            ]
        work_batch.append(s2)

    # use_cache isn't needed (we never reuse past_kv) but gradient checkpointing
    # still interferes with output_hidden_states, so disable it for this block.
    gc_was_enabled = getattr(model.llm, "is_gradient_checkpointing", False)
    if gc_was_enabled:
        model.llm.gradient_checkpointing_disable()

    preds = []
    n_ch = None
    try:
        for step in range(n_forecast_patches):
            # 1) Full cross-channel encoder + sequence assembly.
            inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
                work_batch, return_targets=True
            )
            ts_raw = model._last_ts_raw
            N_ts = ts_raw.size(0)
            if N_ts % B != 0:
                raise ValueError(
                    f"N_ts={N_ts} is not divisible by B={B}. "
                    "All samples in a batch must have the same number of TS segments."
                )
            n_ch = N_ts // B

            # 2) Full LLM forward over the extended prompt.
            # position_ids: built by pad_and_apply_batch so MV time-major tokens
            # at one timestep share a single RoPE position (the MV position-id
            # fix). Falls back to None for safety so HF defaults to sequential
            # if upstream forgot to populate it.
            llm_kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # RE-ENABLED (required for MV): the time-major MV interleave puts
            # all C channels at the same time-step at the SAME RoPE position.
            # HF's default arange(L) would assign distinct positions and the
            # MV predictions collapse during autoregressive eval.
            _pos = getattr(model, "_last_position_ids", None)
            if _pos is not None:
                llm_kwargs["position_ids"] = _pos
            prompt_out = model.llm(**llm_kwargs)
            prompt_hidden = prompt_out.hidden_states[-1]   # [B, L, H]
            del prompt_out

            # 3) Per-channel decoder history from LLM TS-token hidden states.
            # TS tokens follow time-major layout [t0c0, t0c1, ..., t1c0, ...],
            # so channel c of sample b uses patch_positions[c::n_ch].
            hist_buf_list = []
            for i in range(N_ts):
                b_idx = i // n_ch
                c     = i % n_ch
                patch_positions = ts_mask[b_idx].nonzero(as_tuple=True)[0]
                if len(patch_positions) == 0:
                    h_hist = prompt_hidden[b_idx, -1:, :]
                elif n_ch == 1:
                    h_hist = prompt_hidden[b_idx, patch_positions, :]
                else:
                    ch_pos = patch_positions[c::n_ch]
                    h_hist = (
                        prompt_hidden[b_idx, ch_pos, :]
                        if len(ch_pos) > 0
                        else prompt_hidden[b_idx, -1:, :]
                    )
                h_enc = model.decoder_projector(h_hist.float())
                h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
                hist_buf_list.append(h_enc)
            del prompt_hidden

            # 4) Decoder predicts the next patch at the last position.
            dec_input  = pad_sequence(hist_buf_list, batch_first=True)   # [N_ts, N, enc_dim]
            dec_out    = model.decoder(dec_input)                        # [N_ts, N, patch_size]
            pred_patch = dec_out[:, -1, :].float()                       # [N_ts, patch_size]
            preds.append(pred_patch)

            if step == n_forecast_patches - 1:
                break

            # 5) Append the predicted patch to each sample's time_series.
            # Predictions live in normalized space; ts_already_normalized=True on
            # the sample keeps pad_and_apply_batch from re-normalizing on the next step.
            P = pred_patch.size(-1)
            for b_idx in range(B):
                ts_list = work_batch[b_idx]["time_series"]
                ts0 = ts_list[0]                                          # [T] or [T, C]
                pred_rows = pred_patch[b_idx * n_ch:(b_idx + 1) * n_ch]   # [n_ch, P]
                pred_rows = pred_rows.to(dtype=ts0.dtype, device=ts0.device)
                if ts0.dim() == 1:
                    ts_list[0] = torch.cat([ts0, pred_rows[0]], dim=0)
                else:
                    # [n_ch, P] → [P, n_ch] for [T, C] layout
                    ts_list[0] = torch.cat([ts0, pred_rows.t()], dim=0)
                lens = work_batch[b_idx].get("time_series_lengths")
                if lens is not None:
                    lens[0] = int(lens[0]) + P
    finally:
        if gc_was_enabled:
            model.llm.gradient_checkpointing_enable()

    out = torch.stack(preds, dim=1)              # [N_ts, n_forecast_patches, patch_size]
    n_fp = out.size(1)
    out = out.reshape(B, n_ch, n_fp, -1)
    return out                                   # [B, n_ch, n_forecast_patches, patch_size]


def _plot_sample(bi: int, b: int, sample: dict, pred_v, gt_v, out_dir: str, n_sample: int):
    """Plot history + predicted + ground truth for one sample, one image per channel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)

        # ── Extract history  [T] or [T, C] ────────────────────────────────
        ts_segs = sample.get("time_series", [])
        if isinstance(ts_segs, torch.Tensor) and ts_segs.numel() > 0:
            hist = ts_segs[0].float().cpu().numpy()
        elif isinstance(ts_segs, (list, tuple)) and len(ts_segs) > 0:
            h = ts_segs[0]
            hist = (h.float().cpu().numpy()
                    if isinstance(h, torch.Tensor)
                    else np.array(h, dtype=np.float32))
        else:
            hist = np.array([], dtype=np.float32)

        pred = pred_v.numpy() if isinstance(pred_v, torch.Tensor) else np.array(pred_v, dtype=np.float32)
        gt   = gt_v.numpy()   if isinstance(gt_v,   torch.Tensor) else np.array(gt_v,   dtype=np.float32)
        # Normalise to [n_ch, L]
        if pred.ndim == 1:
            pred = pred[np.newaxis, :]
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        n_ch = pred.shape[0]
        L = min(pred.shape[1], gt.shape[1])
        pred, gt = pred[:, :L], gt[:, :L]

        # hist: [T] → [T, 1] so indexing is uniform
        if hist.ndim == 1:
            hist = hist[:, np.newaxis]
        n_ch_hist = hist.shape[1]

        # ── Reconstruct prompt text for title ──────────────────────────────
        pre      = sample.get("pre_prompt", "")
        ts_texts = sample.get("time_series_text", [])
        post     = sample.get("post_prompt", "")
        ts_label = ts_texts[0] if ts_texts else ""
        prompt_str = " | ".join(filter(None, [pre, ts_label, post]))
        if len(prompt_str) > 120:
            prompt_str = prompt_str[:117] + "..."

        H      = hist.shape[0]
        hist_x = np.arange(H)
        fore_x = np.arange(H, H + L)

        # ── One image per channel ──────────────────────────────────────────
        for c in range(n_ch):
            hist_c = hist[:, c] if c < n_ch_hist else hist[:, -1]
            pred_c = pred[c]
            gt_c   = gt[c]
            ch_mse = float(np.mean((pred_c - gt_c) ** 2))
            ch_mae = float(np.mean(np.abs(pred_c - gt_c)))

            fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                                     gridspec_kw={"height_ratios": [3, 1]})
            fig.suptitle(
                f"Sample {n_sample}  ch{c}  (batch {bi}, item {b})  |"
                f"  MSE={ch_mse:.4f}  MAE={ch_mae:.4f}\n"
                f"Prompt: {prompt_str}",
                fontsize=9, wrap=True,
            )

            ax = axes[0]
            if H > 0:
                ax.plot(hist_x, hist_c, color="steelblue", lw=1.2,
                        label=f"history ({H} steps)")
                ax.axvline(H, color="grey", ls="--", lw=0.8, label="forecast start")
            ax.plot(fore_x, gt_c,   color="seagreen",  lw=1.5, label="ground truth")
            ax.plot(fore_x, pred_c, color="darkorange", lw=1.5, label="predicted", alpha=0.85)
            ax.set_ylabel("value (normalised)")
            ax.set_title(f"Channel {c} — Context + Forecast", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.bar(np.arange(L), np.abs(pred_c - gt_c),
                    color="indianred", alpha=0.7, width=1.0)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_xlabel("forecast timestep")
            ax2.set_ylabel("|error|")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fname = os.path.join(out_dir, f"sample_{n_sample:06d}_b{bi}_i{b}_ch{c}.png")
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        _warn_plot_failure("sample_plot", out_dir, exc)


@torch.no_grad()
def run_autoregressive(
    model,
    loader,
    n_forecast_patches,
    out_dir,
    save_plots_every,
    patch_size: int = 1,
    plot_every: int = 0,
    forecast_backbone: str = "llm",
    max_plot_samples: int = 100,
):
    """
    Autoregressive forecasting via direct encoder→LLM→decoder pipeline.
    Only n_forecast_patches LLM forward steps per sample — no text generation.

    Reports mean AND median MSE/MAE.  Mean is sensitive to catastrophic
    failures on a few series; median reflects typical model behaviour.
    """
    device = model.device
    all_mse, all_mae = [], []
    all_results = []
    # Per-(univariate/multivariate) plot counters — caps each subfolder at
    # max_plot_samples. Combined with the sampler's batch shuffle, this picks
    # ~uniformly random samples across the test pool rather than the first
    # save_plots_every-strided batches (which were biased toward low-C content
    # when the group-by-C sampler ordered batches ascending).
    n_plotted_uv = 0
    n_plotted_mv = 0

    for bi, batch in enumerate(tqdm(loader, desc="Autoregressive")):
        if not batch:
            continue
        ts0 = batch[0]['time_series']
        if isinstance(ts0, (list, tuple)):
            print('batch[time_series]: list of', len(ts0), 'tensors,',
                  'first shape:', ts0[0].shape if hasattr(ts0[0], 'shape') else '?')
        else:
            print('batch[time_series]: ', ts0.shape)
        pred_patches = _generate_patches_direct(
            model,
            batch,
            n_forecast_patches,
            device,
            forecast_backbone=forecast_backbone,
        )
        # [B, n_ch, n_forecast_patches, patch_size]

        for b, sample in enumerate(batch):
            # pred_patches[b]: [n_ch, n_fp, patch_size] → [n_ch, T_pred]
            pred_per_ch = pred_patches[b].reshape(pred_patches[b].size(0), -1).float()

            gt_flat = sample.get("ts_future", None)
            if gt_flat is None:
                gt_flat = _parse_future_from_answer(sample.get("answer", ""))
            if gt_flat is None:
                continue
            gt_flat = torch.as_tensor(gt_flat).float()
            # Ensure gt is [T, C]; if 1-D treat as single channel
            if gt_flat.dim() == 1:
                gt_flat = gt_flat.unsqueeze(-1)   # [T, 1]
            # gt_per_ch: [C, T]
            gt_per_ch = gt_flat.t()

            n_ch   = pred_per_ch.size(0)
            # Broadcast: if gt has fewer channels than pred, repeat last channel
            if gt_per_ch.size(0) < n_ch:
                gt_per_ch = gt_per_ch.expand(n_ch, -1)

            L = min(pred_per_ch.size(1), gt_per_ch.size(1))
            pred_v = pred_per_ch[:, :L]                   # [n_ch, L]
            gt_v   = gt_per_ch[:, :L].to(pred_v.device)  # [n_ch, L]

            # Extract history for plotting — stored as [T, C] nested list
            ts_segs = sample.get("time_series", None)
            if isinstance(ts_segs, torch.Tensor) and ts_segs.numel() > 0:
                hist_v = ts_segs[0].float().cpu().tolist()   # [T] or [T, C]
            elif isinstance(ts_segs, (list, tuple)) and len(ts_segs) > 0:
                h = ts_segs[0]
                if isinstance(h, torch.Tensor):
                    hist_v = h.float().cpu().tolist()
                else:
                    import numpy as np
                    hist_v = np.asarray(h, dtype=np.float32).tolist()
            else:
                hist_v = []

            # MSE/MAE averaged across channels for summary metrics
            mse = F.mse_loss(pred_v, gt_v).item()
            mae = F.l1_loss(pred_v, gt_v).item()
            all_mse.append(mse)
            all_mae.append(mae)

            # print('hist_v: ', len(hist_v))
            # print('pred_v: ', pred_v.shape)
            # print('gt: ', gt_v.shape)
            all_results.append({
                "mse": mse, "mae": mae,
                "pred": pred_v.cpu().tolist(),   # [n_ch, L]
                "gt":   gt_v.cpu().tolist(),     # [n_ch, L]
                "history": hist_v,               # [T, C] or [T]
            })

            if plot_every > 0 and len(all_results) % plot_every == 1:
                _plot_sample(bi, b, sample, pred_v.cpu(), gt_v.cpu(),
                             out_dir=os.path.join(out_dir, "sample_plots"),
                             n_sample=len(all_results))

        if all_results and (n_plotted_uv < max_plot_samples or n_plotted_mv < max_plot_samples):
            _last = all_results[-1]
            _pred = _last.get("pred", [])
            try:
                _n_ch = len(_pred) if isinstance(_pred, list) else int(_pred.shape[0])
            except Exception:
                _n_ch = 1
            _sub = "univariate" if _n_ch <= 1 else "multivariate"
            _at_cap = (
                n_plotted_uv >= max_plot_samples if _sub == "univariate"
                else n_plotted_mv >= max_plot_samples
            )
            if not _at_cap:
                _plot_dir = os.path.join(out_dir, _sub)
                _save_plot(_last, _plot_dir, step=bi, tag="autoregressive",
                           patch_size=patch_size)
                if _sub == "univariate":
                    n_plotted_uv += 1
                else:
                    n_plotted_mv += 1

        if bi % 20 == 0 and all_mse:
            t = torch.tensor(all_mse)
            print(f"[batch {bi:05d} | n={len(all_mse):6d}]  "
                  f"median_mse={t.median():.4f}  mean_mse={t.mean():.4f}  "
                  f"max_mse={t.max():.2e}")

    t_mse = torch.tensor(all_mse) if all_mse else torch.zeros(1)
    t_mae = torch.tensor(all_mae) if all_mae else torch.zeros(1)
    return {
        "mean_mse":   t_mse.mean().item(),
        "median_mse": t_mse.median().item(),
        "mean_mae":   t_mae.mean().item(),
        "median_mae": t_mae.median().item(),
        "n_exploded": int((t_mse > 100).sum().item()),   # series where model completely failed
        "n":          len(all_mse),
    }, all_results


# ── Teacher-forced evaluation (kept for comparison) ───────────────────────────

@torch.no_grad()
def run_teacher_forced(
    model,
    loader,
    out_dir,
    save_plots_every,
    patch_size: int = 1,
    forecast_backbone: str = "llm",
    max_batches: int | None = None,
):
    """
    Teacher-forced next-patch prediction.

    For ``forecast_backbone="llm"``, this mirrors the Chronos MSE path used by
    ``joint_phase1_extend``-style checkpoints:
      history prompt -> LLM hidden states at TS positions
      -> decoder_projector -> layer_norm -> causal decoder

    This measures the trained hidden-state-to-patch decoder path in isolation;
    it is useful for joint phase-1 checkpoints but is still not a realistic
    multi-step forecasting benchmark.
    """
    total_mse = total_mae = total_n = 0
    all_results = []
    plot_dir = os.path.join(out_dir, "teacher_forced_plots")
    plotted_files = 0
    skipped_outlier_batches = 0
    skipped_empty_batches = 0

    for bi, batch in enumerate(tqdm(loader, desc="Teacher-forced")):
        if max_batches is not None and bi >= max_batches:
            break

        (
            inputs_embeds,
            attention_mask,
            _,
            ts_mask,
            ts_patch_targets_shifted,
            ts_patch_valid,
            _,
        ) = model.pad_and_apply_batch(batch, return_targets=True)

        if ts_patch_targets_shifted is None or ts_patch_targets_shifted.numel() == 0:
            skipped_empty_batches += 1
            continue

        # Skip batches where any ground-truth value exceeds 1e3 (outlier/unnormalised data)
        if ts_patch_targets_shifted.abs().max().item() > 1e3:
            skipped_outlier_batches += 1
            continue

        _, _, P = ts_patch_targets_shifted.shape
        if forecast_backbone in {"timesfm", "foundational"}:
            pred = _decode_ts_patch_predictions_foundational(model)
        else:
            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pred = _decode_ts_patch_predictions(
                model=model,
                last_hidden=outputs.hidden_states[-1],
                ts_mask=ts_mask,
                ts_patch_targets_shifted=ts_patch_targets_shifted,
                ts_patch_valid=ts_patch_valid,
            )
        tgt       = ts_patch_targets_shifted
        valid     = ts_patch_valid

        pred_v = pred.reshape(-1, P)[valid.reshape(-1)]
        tgt_v  = tgt.reshape(-1, P)[valid.reshape(-1)]
        n      = tgt_v.size(0)

        mse = F.mse_loss(pred_v, tgt_v).item()
        mae = F.l1_loss(pred_v, tgt_v).item()
        total_mse += mse * n
        total_mae += mae * n
        total_n   += n

        all_results.append({
            "mse": mse, "mae": mae,
            "pred": pred_v.cpu().tolist(),
            "gt":   tgt_v.cpu().tolist(),
        })

        if save_plots_every > 0 and bi % save_plots_every == 0:
            plotted_now = _save_plot_patches(
                pred,
                tgt,
                valid,
                plot_dir,
                step=bi,
                patch_size=patch_size,
            )
            plotted_files += plotted_now
            if plotted_now == 0:
                print(
                    f"[teacher_forced] attempted plot at batch {bi} but wrote 0 files. "
                    f"pred_shape={tuple(pred.shape)} tgt_shape={tuple(tgt.shape)} "
                    f"valid_counts={[int(v.sum().item()) for v in valid[:min(3, valid.size(0))]]}"
                )
            if plotted_now > 0 and plotted_files == plotted_now:
                print(f"[teacher_forced] saving plots to: {plot_dir}")

        if bi % 10 == 0 and total_n > 0:
            print(f"[{bi:04d}] batch_mse={mse:.6f}  "
                  f"running avg mse={total_mse/total_n:.6f}  "
                  f"mae={total_mae/total_n:.6f}")

    if save_plots_every > 0 and plotted_files == 0:
        if skipped_outlier_batches > 0:
            print(
                f"[teacher_forced] no plots saved to {plot_dir}. "
                f"Skipped {skipped_outlier_batches} outlier batches before plotting."
            )
        elif skipped_empty_batches > 0:
            print(
                f"[teacher_forced] no plots saved to {plot_dir}. "
                f"Skipped {skipped_empty_batches} empty TS batches before plotting."
            )
        else:
            print(f"[teacher_forced] no plots saved to {plot_dir}.")

    return total_mse / max(1, total_n), total_mae / max(1, total_n), total_n, all_results


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _save_plot(result: dict, out_dir: str, step: int, tag: str, patch_size: int = 1):
    """
    Three-panel plot for autoregressive forecasting results.

    Panel 1 — Full view (history + forecast overlap):
      · History shown in blue; GT future in green; predicted future in orange,
        overlaid so the overlap is immediately visible.
      · Vertical dashed line marks the history / forecast boundary.

    Panel 2 — Ground truth patches (forecast region):
      · Each patch drawn as a solid block with shaded background alternating
        per patch, so individual patch identity is clear.

    Panel 3 — Predicted patches (same layout as Panel 2):
      · GT shown faded for reference; predicted on top in orange.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)

        history = np.array(result.get("history", []), dtype=np.float32)
        pred    = np.array(result["pred"], dtype=np.float32)   # [n_ch, L] or [L]
        gt      = np.array(result["gt"],   dtype=np.float32)   # [n_ch, L] or [L]
        # Normalise to [n_ch, L]
        if pred.ndim == 1:
            pred = pred[np.newaxis, :]
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]
        n_ch = pred.shape[0]
        L = min(pred.shape[1], gt.shape[1])
        pred, gt = pred[:, :L], gt[:, :L]

        # history: [T] → [T, 1] for uniform indexing
        if history.ndim == 1:
            history = history[:, np.newaxis]
        n_ch_hist = history.shape[1]
        hist_len  = history.shape[0]
        hist_x    = np.arange(hist_len)
        fore_x    = np.arange(hist_len, hist_len + L)
        fore_rel  = np.arange(L)

        def _draw_patch_bg(ax, n_patches):
            for p in range(n_patches):
                s, e = p * patch_size, min((p + 1) * patch_size, L)
                if p % 2 == 0:
                    ax.axvspan(s, e, alpha=0.06, color="grey")
            if patch_size > 1:
                for t in range(0, L, patch_size):
                    ax.axvline(t, color="lightgrey", ls=":", lw=0.7)

        n_patches = (L + patch_size - 1) // patch_size

        # ── One image per channel ──────────────────────────────────────────
        for c in range(n_ch):
            hist_c = history[:, c] if c < n_ch_hist else history[:, -1]
            pred_c = pred[c]
            gt_c   = gt[c]
            ch_mse = float(np.mean((pred_c - gt_c) ** 2))
            ch_mae = float(np.mean(np.abs(pred_c - gt_c)))

            fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                                     gridspec_kw={"height_ratios": [2.5, 1, 1]})
            fig.suptitle(
                f"{tag}  |  ch{c}  |  step={step}"
                f"  |  MSE={ch_mse:.4f}  MAE={ch_mae:.4f}",
                fontsize=11,
            )

            # Panel 1 — context + forecast
            ax = axes[0]
            if hist_len > 0:
                ax.plot(hist_x, hist_c, color="steelblue", lw=1.2, label="history")
                ax.axvline(hist_len, color="grey", ls="--", lw=0.8, label="forecast start")
            ax.plot(fore_x, gt_c,   color="seagreen",  lw=1.5, label="ground truth")
            ax.plot(fore_x, pred_c, color="darkorange", lw=1.5, label="predicted", alpha=0.85)
            if patch_size > 1:
                for t in range(hist_len, hist_len + L, patch_size):
                    ax.axvline(t, color="lightgrey", ls=":", lw=0.5)
            ax.set_ylabel("value (normalised)")
            ax.set_title(f"Channel {c} — Context + Forecast (overlaid)", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)

            # Panel 2 — ground truth patches
            ax2 = axes[1]
            _draw_patch_bg(ax2, n_patches)
            ax2.plot(fore_rel, gt_c, color="seagreen", lw=1.5, label="ground truth")
            ax2.set_ylabel("value (normalised)")
            ax2.set_title("Ground Truth Patches", fontsize=9)
            ax2.legend(fontsize=8, loc="upper left")

            # Panel 3 — predicted vs gt patches
            ax3 = axes[2]
            _draw_patch_bg(ax3, n_patches)
            ax3.plot(fore_rel, gt_c,   color="seagreen",  lw=1.5, label="ground truth", alpha=0.35)
            ax3.plot(fore_rel, pred_c, color="darkorange", lw=1.5, label="predicted")
            ax3.set_xlabel("forecast timestep")
            ax3.set_ylabel("value (normalised)")
            ax3.set_title("Predicted Patches (GT faded for reference)", fontsize=9)
            ax3.legend(fontsize=8, loc="upper left")

            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"{tag}_step{step:06d}_ch{c}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
    except Exception as exc:
        _warn_plot_failure("autoregressive", out_dir, exc)


def _save_plot_patches(pred, tgt, valid, out_dir, step, patch_size: int = 1, max_series: int = 2):
    """
    Teacher-forced plot: for each series, show gt-next-patch vs predicted-next-patch
    with patch boundaries marked.
    """
    saved = 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        n = min(max_series, pred.size(0))
        for s in range(n):
            idx = valid[s].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            gt_1d = tgt[s][idx].reshape(-1).float().cpu().numpy()
            pr_1d = pred[s][idx].reshape(-1).float().cpu().numpy()
            L     = len(gt_1d)

            fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [2, 1]})
            fig.suptitle(f"Teacher-forced  step={step}  series={s}", fontsize=11)

            # full overlap plot
            ax = axes[0]
            ax.plot(gt_1d, color="seagreen",   lw=1.4, label="ground truth (next patch)")
            ax.plot(pr_1d, color="darkorange",  lw=1.4, label="predicted",  alpha=0.85)
            if patch_size > 1:
                for b in range(0, L, patch_size):
                    ax.axvline(b, color="lightgrey", ls=":", lw=0.7)
            ax.set_ylabel("value (normalised)")
            ax.legend(fontsize=8)
            ax.set_title("Ground Truth vs Predicted (all positions)", fontsize=9)

            # residual plot
            ax2 = axes[1]
            ax2.bar(range(L), gt_1d - pr_1d, color="indianred", alpha=0.7, width=1.0)
            ax2.axhline(0, color="black", lw=0.8)
            if patch_size > 1:
                for b in range(0, L, patch_size):
                    ax2.axvline(b, color="lightgrey", ls=":", lw=0.7)
            ax2.set_xlabel("timestep")
            ax2.set_ylabel("residual (gt − pred)")
            ax2.set_title("Residual", fontsize=9)

            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"tf_step{step:06d}_s{s}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            saved += 1
    except Exception as exc:
        _warn_plot_failure("teacher_forced", out_dir, exc)
    return saved


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(args) -> OpenTSLMSPFixPos2D:
    cfg_kwargs = dict(
        is_pretraining=True,
        use_only_time_pe=args.use_only_time_pe,
        use_only_channel_pe=args.use_only_channel_pe,
        use_freq_embed=args.use_freq_embed,
        enc_dec_type=getattr(args, "enc_dec_type", "transformer"),
        mlp_enc_hidden_dim=getattr(args, "mlp_enc_hidden_dim", None),
        mlp_enc_ff_dim=getattr(args, "mlp_enc_ff_dim", None),
        mlp_dec_hidden_dim=getattr(args, "mlp_dec_hidden_dim", None),
        mlp_dec_ff_dim=getattr(args, "mlp_dec_ff_dim", None),
    )
    if args.patch_size is not None:
        cfg_kwargs["patch_size"] = args.patch_size
    cfg = SimpleNamespace(**cfg_kwargs)
    model_cls = OpenTSLMSPFixPos2D
    model = model_cls(cfg, llm_id=args.llm_id, device=args.device)
    if args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model.load_from_file(args.ckpt)
    model.eval()
    return model


_STAGE_TO_EVAL_FORMAT = {
    # Joint Phase-1 stages — all use Phase1PretrainEvalDataset (phase1_pretrain).
    # Phase1JointDataset chronos rows are next-patch pretraining rows with no
    # ts_future, so ChronosOnlySubset would silently skip every sample.
    # Phase1PretrainEvalDataset uses the same Chronos split + prompt format and
    # correctly attaches ts_future for autoregressive evaluation.
    "joint_phase1":                              "phase1_pretrain",
    "joint_phase1_extend":                       "phase1_pretrain",
    "joint_phase1_extend_lorallm":               "phase1_pretrain",
    "joint_phase1_extend_balanced":              "phase1_pretrain",
    "joint_phase1_extend_balanced_m4_by_size":   "phase1_pretrain",
    "joint_phase1_extend_m4_w05":                "phase1_pretrain",
    "joint_phase1_extend_m4_w15":                "phase1_pretrain",
    "joint_phase1_tsqa_m4":                      "phase1_pretrain",
    "joint_phase1_tsqa":                         "phase1_pretrain",
    # Pre-joint stages
    "phase1_chronos_mse":                        "phase1_pretrain",
    "phase1_tsqa_m4_ce":                         "phase1_pretrain",
    "phase1_mse_then_ce":                        "phase1_pretrain",
    "stage2_all":                                "train_aligned",
    "foundation_stage1_chatts":                  "train_aligned",
}


def _resolve_eval_format(eval_format: str, ckpt_path: str, stage_name: str = "") -> str:
    if eval_format != "auto":
        return eval_format
    # Stage name takes priority — unambiguous and independent of path layout.
    if stage_name and stage_name in _STAGE_TO_EVAL_FORMAT:
        return _STAGE_TO_EVAL_FORMAT[stage_name]
    # Fallback: infer from checkpoint path for backwards compatibility.
    if "joint_phase1" in ckpt_path:
        return "phase1_pretrain"
    if "stage2_all" in ckpt_path:
        return "train_aligned"
    if any(k in ckpt_path for k in ("phase1_chronos_mse", "phase1_tsqa_m4_ce", "phase1_mse_then_ce")):
        return "phase1_pretrain"
    return "chronos"


def _resolve_forecast_backbone(
    forecast_backbone: str,
    ckpt_path: str,
    stage_name: str = "",
) -> str:
    if forecast_backbone == "timesfm":
        return "foundational"
    if forecast_backbone != "auto":
        return forecast_backbone
    return "llm"


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="OpenTSLM forecasting evaluation")
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--llm_id",       default="google/gemma-3-270m")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--mode",         default="autoregressive",
                    choices=["autoregressive", "teacher_forced"],
                    help="autoregressive: model.generate() patch-by-patch; "
                         "teacher_forced: one-pass hidden-state decoding")
    ap.add_argument("--dataset_name", default="all")
    ap.add_argument("--eval_format",  default="auto",
                    choices=["auto", "chronos", "train_aligned", "joint", "joint_extend", "phase1_pretrain", "etth1"])
    ap.add_argument("--batch_size",   type=int,   default=32,
                    help="Batch size for inference. Larger = faster; limited by GPU memory.")
    ap.add_argument("--patch_size",   type=int,   default=None)
    ap.add_argument("--history_len",  type=int,   default=96,
                    help="History context length in time steps fed to the encoder. "
                         "Default 96 — symmetric with the default pred_len.")
    ap.add_argument("--pred_len",     type=int,   default=96,
                    help="Forecast horizon in time steps.")
    ap.add_argument("--max_batches",  type=int,   default=10_000_000)
    ap.add_argument("--out_dir",      default="./phase1_ts_eval")
    ap.add_argument("--save_plots_every", type=int, default=10)
    ap.add_argument("--plot_every",       type=int, default=0,
                    help="Save a forecast plot every N samples. 0 = disabled. Saved to <out_dir>/sample_plots/.")
    ap.add_argument("--enc_dec_type", type=str, default="transformer",
                    choices=["transformer", "mlp"],
                    help="Must match the backbone the checkpoint was trained with.")
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=None,
                    help="Must match training if --enc_dec_type=mlp.")
    ap.add_argument("--mlp_enc_ff_dim",     type=int, default=None,
                    help="Must match training if --enc_dec_type=mlp.")
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=None,
                    help="Must match training if --enc_dec_type=mlp.")
    ap.add_argument("--mlp_dec_ff_dim",     type=int, default=None,
                    help="Must match training if --enc_dec_type=mlp.")
    ap.add_argument("--use_only_time_pe",    action="store_true")
    ap.add_argument("--use_only_channel_pe", action="store_true")
    ap.add_argument("--use_freq_embed",      action="store_true")
    ap.add_argument("--stage_name",    default="",
                    help="Training stage name (e.g. joint_phase1_extend_balanced_m4_by_size). "
                         "Used to resolve --eval_format automatically when set to 'auto'.")
    ap.add_argument("--lora_r",       type=int, default=0)
    ap.add_argument("--lora_alpha",   type=int, default=32)
    ap.add_argument("--etth1_csv",    default=os.path.join(os.environ.get("FTS_ETT_DIR", "data/ETT"), "ETTh1.csv"),
                    help="Path to ETTh1.csv (used when --eval_format etth1).")
    ap.add_argument("--etth1_stride", type=int, default=1,
                    help="Sliding window stride for ETTh1 evaluation (default 1 = dense).")
    ap.add_argument("--eval_split",   default="train", choices=["train", "val", "test"],
                    help="Which data split to evaluate on. Use 'train' to check whether "
                         "the model fits its own training distribution (overfitting sanity check). "
                         "Default 'test' for held-out evaluation.")
    ap.add_argument("--forecast_backbone", default="auto", choices=["auto", "llm", "foundational", "timesfm"],
                    help="Forecasting backbone to evaluate. 'auto' uses the "
                         "direct foundational encoder/decoder path for phase1_chronos_mse "
                         "checkpoints and the LLM-routed path otherwise.")
    ap.add_argument("--max_plot_samples", type=int, default=100,
                    help="Cap the autoregressive plots per univariate / "
                         "multivariate subfolder. Plots land under "
                         "<out_dir>/{univariate,multivariate}/. Combined with "
                         "the group-by-C sampler's batch shuffle, this picks "
                         "~uniformly random samples instead of the first "
                         "save_plots_every-strided ones. 0 disables.")
    ap.add_argument("--source_filter", default="all",
                    choices=["all", "chronos_uv", "synthetic"],
                    help="phase1_pretrain only: restrict the test pool to one source. "
                         "'chronos_uv' = univariate registry rows, 'synthetic' = MV "
                         "rows from synthetic_data/unified/data. 'all' (default) keeps "
                         "the merged behaviour. Requires the chronos split cache to "
                         "include the 'source' column (training builds it).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Resolved output directory: {args.out_dir}")

    model = load_model(args)

    eval_patch_size = args.patch_size or getattr(model, "patch_size", None)
    if eval_patch_size is None:
        raise ValueError("Cannot infer patch_size. Pass --patch_size explicitly.")
    eval_patch_size = int(eval_patch_size)


    eos_str      = model.get_eos_token()
    eval_format  = _resolve_eval_format(args.eval_format, args.ckpt, args.stage_name)
    forecast_backbone = _resolve_forecast_backbone(
        args.forecast_backbone, args.ckpt, args.stage_name
    )
    n_forecast_patches = args.pred_len // eval_patch_size

    split = args.eval_split   # "train" | "val" | "test"

    if eval_format == "etth1":
        test_ds = ETTh1EvalDataset(
            csv_path=args.etth1_csv,
            mode=split,
            patch_size=eval_patch_size,
            pred_len=args.pred_len,
            history_len=args.history_len,
            stride=args.etth1_stride,
            EOS_TOKEN=eos_str,
        )
    elif eval_format == "phase1_pretrain":
        test_ds = Phase1PretrainEvalDataset(
            mode=split, EOS_TOKEN=eos_str,
            patch_size=eval_patch_size,
            pred_len=args.pred_len,
            history_lens=[args.history_len],
            source_filter=args.source_filter,
        )
    elif eval_format == "train_aligned":
        test_ds = TrainingAlignedChronosEvalDataset(
            mode=split, EOS_TOKEN=eos_str,
            dataset_name=args.dataset_name,
            patch_size=eval_patch_size,
            pred_len=args.pred_len,
            history_lens=[args.history_len],
        )
    else:
        test_ds = ChronosDataset(
            mode=split, EOS_TOKEN=eos_str, dataset_name=args.dataset_name,
        )

    print(f"Dataset: {len(test_ds)} series | split={split} | mode={args.mode} | "
          f"patch_size={eval_patch_size} | pred_len={args.pred_len} "
          f"({n_forecast_patches} patches) | eval_format={eval_format} | "
          f"forecast_backbone={forecast_backbone}")

    def _collate(batch):
        batch = [s for s in batch if s is not None]  # drop short series
        if not batch:
            return []
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=eval_patch_size
        )

    # `_generate_patches_direct` asserts `N_ts % B == 0` (uniform channel count
    # per batch). Synthetic MV rows have C ∈ {3..10}, so a uniform DataLoader
    # produces mixed-C batches that crash. Build per-row C from Arrow offsets
    # (no float materialisation) and yield uniform-C batches.
    batch_sampler = None
    if eval_format == "phase1_pretrain" and hasattr(test_ds, "_get_split"):
        try:
            _split_ds = test_ds._get_split()
            import numpy as _np_b
            _tgt_col = _split_ds.data.column("target")
            _n_rows = len(_split_ds)
            _n_ch = _np_b.empty(_n_rows, dtype=_np_b.int32)
            _cursor = 0
            for _chunk in _tgt_col.chunks:
                _outer = _chunk.offsets.to_numpy(zero_copy_only=False)
                _inner = _chunk.values.offsets.to_numpy(zero_copy_only=False)
                _nc = len(_chunk)
                for _i in range(_nc):
                    _start_t = int(_outer[_i])
                    _end_t   = int(_outer[_i + 1])
                    if _end_t > _start_t:
                        _n_ch[_cursor + _i] = int(_inner[_start_t + 1] - _inner[_start_t])
                    else:
                        _n_ch[_cursor + _i] = 0
                _cursor += _nc

            _groups: dict = {}
            for _i, _c in enumerate(_n_ch.tolist()):
                _groups.setdefault(int(_c), []).append(_i)
            # Shuffle WITHIN each C bucket so the within-bucket sample order
            # is randomised (fixed seed for reproducibility across runs).
            import random as _random_b
            _shuffle_rng = _random_b.Random(42)
            _batches: list = []
            for _c in sorted(_groups):
                _idxs = _groups[_c]
                _shuffle_rng.shuffle(_idxs)
                for _s in range(0, len(_idxs), args.batch_size):
                    _batches.append(_idxs[_s : _s + args.batch_size])
            # Shuffle batch ORDER across C buckets so the run interleaves
            # univariate and multivariate batches — gives the per-subfolder
            # 100-plot cap a representative sample instead of "first 100 of
            # whichever bucket comes first".
            _shuffle_rng.shuffle(_batches)
            _summary = ", ".join(f"C={_c}:{len(_groups[_c])}" for _c in sorted(_groups))
            print(
                f"[fixed_inference] group-by-C batch sampler: {len(_batches)} "
                f"uniform-C batches over {_n_rows} samples ({_summary}) "
                f"[order shuffled with seed=42]"
            )

            class _PrebuiltBatchSampler:
                def __init__(self, batches):
                    self._batches = batches
                def __iter__(self):
                    return iter(self._batches)
                def __len__(self):
                    return len(self._batches)

            batch_sampler = _PrebuiltBatchSampler(_batches)
        except Exception as _exc:
            print(
                f"[fixed_inference] group-by-C sampler build failed ({_exc!r}); "
                "falling back to uniform batch_size — may crash on mixed-C batches."
            )
            batch_sampler = None

    if batch_sampler is not None:
        loader = DataLoader(
            test_ds,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
        )
    else:
        loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

    if args.mode == "autoregressive":
        metrics, _ = run_autoregressive(
            model, loader, n_forecast_patches, args.out_dir, args.save_plots_every,
            patch_size=eval_patch_size,
            plot_every=args.plot_every,
            forecast_backbone=forecast_backbone,
            max_plot_samples=args.max_plot_samples,
        )
    else:
        avg_mse, avg_mae, n, _ = run_teacher_forced(
            model, loader, args.out_dir, args.save_plots_every,
            patch_size=eval_patch_size,
            forecast_backbone=forecast_backbone,
            max_batches=args.max_batches,
        )
        metrics = {"mean_mse": avg_mse, "mean_mae": avg_mae, "n": n}

    metrics["mode"] = args.mode

    print(f"\n{'='*60}")
    print(f"Mode       : {args.mode}")
    print(f"Checkpoint : {args.ckpt}")
    for k, v in metrics.items():
        if k == "mode":
            continue
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print(f"{'='*60}")

    out_json = os.path.join(args.out_dir, f"metrics_{args.mode}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
