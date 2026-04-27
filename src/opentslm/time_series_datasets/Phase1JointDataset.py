# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Phase1JointDataset.py
---------------------
Joint training dataset for Phase 1, combining three tasks:
  1. Chronos next-patch pretraining on time-series windows
  2. TSQA multiple-choice QA  (stage1_mcq)
  3. M4 caption generation    (stage2_captioning)

All three splits are balanced to the size of the smallest split, then
shuffled together, giving equal task exposure per training epoch.
"""

import json
import os
import random
from typing import List, Tuple, Dict

import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    DownloadMode,
)
import time


from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.ChronosDataset import (
    BASE_DATA_PATH,
    PREPROCESSED_SPLITS_DIR,
    _last_value_impute,
)
from opentslm.time_series_datasets.PretrainDataset import PretrainDataset
from opentslm.model_config import PATCH_SIZE, WINDOW_LENS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_FRAC = 0.1
VAL_FRAC = 0.1

CHRONOS_DATASET_REGISTRY: Dict[str, List[str]] = {
    "dominick": ["target"],
    "monash_fred_md": ["target"],
    # "electricity_15min": ["consumption_kW"],
    "ercot": ["target"],
    "exchange_rate": ["target"],
    # "ushcn_daily": ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
    "weatherbench_daily": ['target'],
    "weatherbench_weekly": ['target'],
    "wiki_daily_100k": ['target'],
    "monash_australian_electricity": ['target'],

}

# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Phase1JointDataset(PretrainDataset):
    """Joint Phase-1 dataset: Chronos next-patch pretraining + TSQA MCQ + M4 captioning.

    Inherits from PretrainDataset so that rows are formatted lazily in
    __getitem__, not all at once at init time.  This allows the Chronos
    corpus (180k+ series) to be used in full without hanging at startup.

    _load_splits() builds a lightweight index:
      - TSQA rows are stored eagerly (small: ~5k rows).
      - Chronos rows are stored as stubs {"type": "chronos", "_raw_idx": i, ...};
        the actual window is cropped and normalised on-demand inside _format_sample.

    Rows are tagged with a ``"type"`` key (``"chronos"``, ``"tsqa"``, or
    ``"m4"``) so the training loop can route CE vs MSE losses correctly.

    Args:
        max_samples_per_source: If set, cap each data source (chronos, tsqa, m4)
            to this many samples per split.  Useful for quick smoke-tests.
        patch_size: Chronos patch size used to compute patch-count prompts.
        pred_len: Chronos forecast horizon in timesteps.
    """

    def __init__(
        self,
        *args,
        max_samples_per_source: int | None = None,
        data_seed: int = 42,
        patch_size: int = PATCH_SIZE,
        val_fraction: float = 1.0,
        chronos_window_stride: int | None = None,
        **kwargs,
    ):
        self.max_samples_per_source = max_samples_per_source
        self.val_fraction = val_fraction
        self.data_seed = data_seed
        self.patch_size = int(patch_size)
        self.window_lens = [int(w) for w in WINDOW_LENS]
        # Stride between enumerated training windows (in timesteps). When set,
        # each HF row yields multiple training stubs — one per window start —
        # so len(dataset) scales with the number of windows rather than the
        # number of rows. Val/test still use one deterministic last-window stub.
        self.chronos_window_stride = (
            int(chronos_window_stride) if chronos_window_stride is not None else None
        )

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")

        self.valid_window_lens = [
            w for w in self.window_lens if w > 0 and w % self.patch_size == 0
        ]
        if not self.valid_window_lens:
            raise ValueError(
                f"No valid window lengths for patch_size={self.patch_size}. "
                f"Configured WINDOW_LENS={self.window_lens}"
            )

        # Per-dataset deterministic RNG so all DDP ranks build identical splits.
        self.rng = random.Random(data_seed)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Chronos helpers  (mirrors Stage2AllDataset / ChronosDataset)
    # ------------------------------------------------------------------

    def _normalize_to_target(self, ds: Dataset, target_keys: List[str]) -> Dataset:
        # All branches must emit the same schema so parts can be concatenated:
        # "target" is always Sequence(Sequence(float32)) shaped [T, C], with
        # C=1 for truly univariate sources.
        if len(target_keys) == 1:
            key = target_keys[0]
            if key != "target":
                ds = ds.rename_column(key, "target")

            # If every row shares the same timestamp index, treat rows as channels
            # of one multivariate series (e.g. monash_australian_electricity has 5
            # rows = 5 Australian states on the same half-hourly timeline).
            # Falls through to univariate for datasets where rows are independent
            # series with different timelines (e.g. wiki_daily_100k).
            stacked = self._maybe_stack_rows_as_channels(ds)
            if stacked is not None:
                return stacked

            ds = ds.select_columns(["target"])
            ds = ds.map(
                lambda batch: {
                    "target": [[[float(v)] for v in seq] for seq in batch["target"]]
                },
                batched=True,
                remove_columns=ds.column_names,
            )
            ds = ds.cast_column("target", Sequence(Sequence(Value(dtype="float32"))))
        else:
            # Multi-channel: keep one row per entity; stack all channel columns
            # into a single "target" column stored as [T, C] (nested list).
            # This preserves the full multivariate structure for 2-D positional
            # encoding in the encoder.
            keys = target_keys
            def _stack_channels(batch):
                B = len(batch[keys[0]])
                out = []
                for i in range(B):
                    channels = [batch[k][i] for k in keys]  # C lists of length T_i
                    T = len(channels[0])
                    # Transpose to [T, C]
                    row_tc = [
                        [float(channels[c][t]) for c in range(len(keys))]
                        for t in range(T)
                    ]
                    out.append(row_tc)
                return {"target": out}
            ds = ds.map(_stack_channels, batched=True, remove_columns=ds.column_names)
            ds = ds.cast_column("target", Sequence(Sequence(Value(dtype="float32"))))
        return ds

    # Row-count ceiling for the "rows-as-channels" heuristic. Datasets with many
    # rows on the same timeline (e.g. wiki_daily_100k: 100k Wikipedia articles
    # from 2015-2022) are almost always independent entities that happen to
    # share a timeline, NOT channels of one giant multivariate series. Anything
    # over this bound stays univariate.
    MAX_ROWS_AS_CHANNELS = 64

    @staticmethod
    def _temporal_split_rows(
        ds: Dataset, val_frac: float, test_frac: float
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split each row's ``target`` along time into train/val/test slices.

        Every row appears in all three splits but with non-overlapping time
        spans: train = first (1 - val_frac - test_frac), val = middle val_frac,
        test = last test_frac. Used when the row count is too small for a
        row-level split (e.g. one long multichannel series).
        """
        train_rows, val_rows, test_rows = [], [], []
        for i in range(len(ds)):
            row = ds[i]
            target = row["target"]
            T = len(target)
            T_test = max(1, int(T * test_frac))
            T_val = max(1, int(T * val_frac))
            T_train = T - T_val - T_test
            if T_train <= 0:
                # Series too short — keep it whole in train only.
                train_rows.append({"target": target})
                continue
            train_rows.append({"target": target[:T_train]})
            val_rows.append({"target": target[T_train:T_train + T_val]})
            test_rows.append({"target": target[T_train + T_val:]})

        def _from_rows(rows: List[dict]) -> Dataset:
            if not rows:
                return Dataset.from_dict({"target": []}, features=ds.features)
            return Dataset.from_dict(
                {"target": [r["target"] for r in rows]}, features=ds.features
            )

        return _from_rows(train_rows), _from_rows(val_rows), _from_rows(test_rows)

    @staticmethod
    def _maybe_stack_rows_as_channels(ds: Dataset) -> Dataset | None:
        """Collapse rows with a shared start timestamp into one multivariate series.

        Stacking applies when:
          - row count is small enough (see MAX_ROWS_AS_CHANNELS), AND
          - every row's timestamp column starts at the same point AND advances
            identically over the first `min_T` steps (the shortest row's length).

        All rows are truncated to min_T. This handles Chronos datasets like
        monash_australian_electricity where five states share a 2002-01-01 start
        but terminate on slightly different dates.

        Returns None when rows are semantically independent (different start
        dates, or too many rows) — the caller keeps the per-row univariate layout.
        """
        if "timestamp" not in ds.column_names or len(ds) <= 1:
            return None
        if len(ds) > Phase1JointDataset.MAX_ROWS_AS_CHANNELS:
            return None

        timestamps = [ds[i]["timestamp"] for i in range(len(ds))]
        min_T = min(len(ts) for ts in timestamps)
        if min_T <= 0:
            return None

        ref = timestamps[0]
        ref_start = ref[0]
        ref_end_trunc = ref[min_T - 1]
        for ts_i in timestamps[1:]:
            if ts_i[0] != ref_start or ts_i[min_T - 1] != ref_end_trunc:
                return None

        targets = [ds[i]["target"][:min_T] for i in range(len(ds))]   # C lists of length min_T
        C = len(targets)
        row_tc = [[float(targets[c][t]) for c in range(C)] for t in range(min_T)]
        out = Dataset.from_dict({"target": [row_tc]})
        return out.cast_column("target", Sequence(Sequence(Value(dtype="float32"))))

    def _load_or_build_chronos_raw(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Return (train, val, test) Chronos HF Datasets with a ``target`` column.

        Reuses ChronosDataset's cached splits when available; builds once on
        rank 0 otherwise.
        """
        rank = int(os.environ.get("RANK", 0))
        split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all")
        done_flag = os.path.join(split_dir, "_DONE")

        if rank == 0 and not os.path.exists(done_flag):
            print("[Phase1Joint] Building Chronos splits (runs once)…")
            parts_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "_parts")
            os.makedirs(parts_dir, exist_ok=True)

            part_paths = []
            for name, keys in CHRONOS_DATASET_REGISTRY.items():
                part_path = os.path.join(parts_dir, name)
                if not os.path.exists(part_path):
                    try:
                        ds = load_dataset(
                            f"{BASE_DATA_PATH}/{name}",
                            split="train",
                            trust_remote_code=True,
                            keep_in_memory=True,
                        )
                        ds = self._normalize_to_target(ds, keys)
                        print(f"  -> '{name}': {len(ds)} series — saving…")
                        ds.save_to_disk(part_path)
                        del ds
                    except Exception as exc:
                        print(f"  -> WARNING: failed to load '{name}': {exc}")
                        continue
                else:
                    print(f"  -> cached '{name}' found, skipping.")
                part_paths.append(part_path)

            if not part_paths:
                raise RuntimeError("No Chronos datasets could be loaded!")

            parts = [load_from_disk(p, keep_in_memory=False) for p in part_paths]
            ds_full = concatenate_datasets(parts)

            # When the Chronos corpus is tiny (e.g. a single long multichannel
            # series from row-stacking), HF's row-level train_test_split can't
            # produce non-empty splits. Fall back to a *temporal* split on each
            # row's target: train gets the first (1 - VAL - TEST) fraction of
            # every series, val the middle fraction, test the last fraction.
            # This matches standard time-series practice (no future-looking
            # leakage) and works for arbitrary row counts.
            MIN_ROWS_FOR_ROW_SPLIT = 10
            if len(ds_full) < MIN_ROWS_FOR_ROW_SPLIT:
                print(
                    f"[Phase1Joint] {len(ds_full)} row(s) in Chronos corpus — "
                    "splitting each row temporally (train=first 80%, val=next "
                    "10%, test=last 10%) instead of by row."
                )
                train, val, test = self._temporal_split_rows(
                    ds_full, val_frac=VAL_FRAC, test_frac=TEST_FRAC
                )
            else:
                train_val, test = ds_full.train_test_split(test_size=TEST_FRAC, seed=42).values()
                train, val = train_val.train_test_split(
                    test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
                ).values()

            os.makedirs(split_dir, exist_ok=True)
            train.save_to_disk(os.path.join(split_dir, "train"))
            val.save_to_disk(os.path.join(split_dir, "val"))
            test.save_to_disk(os.path.join(split_dir, "test"))
            with open(done_flag, "w") as fh:
                fh.write("done\n")
            del ds_full, train, val, test
            print(f"[Phase1Joint] Chronos splits saved to {split_dir}")

        if dist.is_initialized():
            dist.barrier()

        # keep_in_memory=True: load Arrow data into RAM instead of using memory-mapped
        # files. Arrow mmaps are NOT fork-safe — PyTorch DataLoader workers (forked
        # from the main process) inherit the mmap handles but cannot safely access
        # them, causing the first Chronos batch fetch to hang indefinitely.
        # The Chronos splits are small (~MB) so in-memory loading is fine.
        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=True)
        val = load_from_disk(os.path.join(split_dir, "val"), keep_in_memory=True)
        test = load_from_disk(os.path.join(split_dir, "test"), keep_in_memory=True)
        return train, val, test

    # @staticmethod
    # def _znorm(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    #     """Z-normalize a 1-D tensor. Returns (normalized, mean, std).
    #     Uses a consistent epsilon=1e-8 across all tasks."""
    #     if x.dim() > 1:
    #         x = x.reshape(-1)          # flatten multivariate to 1-D for scalar stats
    #     m = x.mean()
    #     s = x.std().clamp(min=1e-8)
    #     return (x - m) / s, m.item(), s.item()
    @staticmethod
    def _znorm(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Z-normalize a tensor. Returns (normalized, mean_ch0, std_ch0).

        For univariate [T]: global z-norm.
        For multivariate [T, C]: per-channel z-norm (each channel independently).
        The scalar mean/std returned correspond to channel 0 and are used only
        for logging / de-normalization metadata.
        """
        if x.dim() == 1:
            m = x.mean()
            s = x.std().clamp(min=1e-8)
            return (x - m) / s, m.item(), s.item()
        else:  # [T, C]
            m = x.mean(dim=0, keepdim=True)          # [1, C]
            s = x.std(dim=0, keepdim=True).clamp(min=1e-8)  # [1, C]
            normed = (x - m) / s
            return normed, m[0, 0].item(), s[0, 0].item()
    

    def _sample_phase1_window(
        self,
        series: torch.Tensor,
        mode: str = "train",
        window_start: int | None = None,
    ) -> dict | None:
        """Prepare one full-window Chronos sample for Phase 1 next-patch training.

        If ``window_start`` is provided (via a stub from ``_make_chronos_stubs``),
        it overrides the usual random/last-window selection — each enumerated
        window becomes a deterministic sample.

        Also grabs the first patch *beyond* the window as ``ts_future`` whenever
        the series is long enough.  This closes the training/inference gap:
        ``compute_pretrain_loss`` uses it to supervise decoder position N_p-1
        (the last history position) — the exact position read at inference time.
        Without this, position N_p-1 receives zero gradient during Phase 1 and
        collapses to the normalised mean (flat-line forecast).
        """
        T = series.shape[0]  # works for both [T] and [T, C]
        candidates = [w for w in self.valid_window_lens if T >= w]
        if candidates:
            window_len = (
                self.rng.choice(candidates)
                if mode == "train"
                else max(candidates)
            )
            if window_start is not None:
                start = max(0, min(int(window_start), T - window_len))
            else:
                start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
            window = series[start: start + window_len]
            window = _last_value_impute(window)
        else:
            is_mv = series.dim() == 2
            min_window_len = 2 * self.patch_size
            if T <= 0:
                # Empty series: synthesise a tiny constant window.
                if is_mv:
                    C = series.shape[1]
                    window = torch.zeros(min_window_len, C, dtype=torch.float32)
                else:
                    window = torch.zeros(min_window_len, dtype=torch.float32)
                window_len = min_window_len
                start = 0
            elif T >= min_window_len:
                window_len = max(min_window_len, (T // self.patch_size) * self.patch_size)
                start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
                window = series[start: start + window_len]
                window = _last_value_impute(window)
            else:
                # Very short series: pad by repeating the last observed value.
                window = _last_value_impute(series)
                if is_mv:
                    pad_len = min_window_len - window.shape[0]
                    pad_val = window[-1:].expand(pad_len, -1)
                    window = torch.cat([window, pad_val], dim=0)
                else:
                    pad_len = min_window_len - window.numel()
                    pad_val = window[-1]
                    window = torch.cat([window, window.new_full((pad_len,), pad_val)])
                window_len = min_window_len
                start = 0

        window_norm, norm_mean, norm_std = self._znorm(window)

        # Grab the first patch beyond the window end as the forecasting target.
        # Used by compute_pretrain_loss to supervise decoder position N_p-1.
        ts_future = None
        future_start = start + window_len
        if future_start + self.patch_size <= T:
            future_raw = _last_value_impute(series[future_start: future_start + self.patch_size])
            if series.dim() == 1:
                ts_future = (
                    (future_raw - norm_mean) / max(norm_std, 1e-8)
                ).to(dtype=torch.float32).clone()
            else:
                # Multivariate: per-channel normalise using the window's per-channel stats.
                ch_mean = window.mean(dim=0, keepdim=True)                     # [1, C]
                ch_std  = window.std(dim=0, keepdim=True).clamp(min=1e-8)      # [1, C]
                future_norm = (future_raw - ch_mean) / ch_std                   # [P, C]
                ts_future = future_norm.transpose(0, 1).to(dtype=torch.float32).contiguous()  # [C, P]

        return {
            "type": "chronos",
            "series": window_norm.numpy(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_patches": window_len // self.patch_size,
            "ts_future": ts_future,   # UV: [patch_size]; MV: [C, patch_size]; or None
        }

    def _process_chronos_row(
        self,
        row: dict,
        mode: str = "train",
        window_start: int | None = None,
    ) -> dict | None:
        # "target" stores [T] for univariate or [T, C] (nested list) for multivariate.
        series = torch.tensor(row["target"], dtype=torch.float32)
        return self._sample_phase1_window(series, mode, window_start=window_start)

    def _get_raw_ts(self, _row):
        # Required by PretrainDataset ABC; the actual TS data is returned via
        # _get_text_time_series_prompt_list, so this method is unused.
        return None

    # ------------------------------------------------------------------
    # TSQA loader (small — load eagerly as dicts)
    # ------------------------------------------------------------------

    def _load_tsqa_lists(self, cap) -> Tuple[list, list, list]:
        """Return (train, val, test) TSQA lists, formatted as raw dicts."""
        def _safe_load(name, split, cache_dir=None, retries=5, wait=8):
            last_err = None
            for attempt in range(retries):
                try:
                    return load_dataset(
                        name, split=split, cache_dir=cache_dir,
                        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    )
                except OSError as e:
                    last_err = e
                    if (getattr(e, "errno", None) == 116 or "Stale file handle" in str(e)) and attempt < retries - 1:
                        time.sleep(wait)
                        continue
                    raise
            raise last_err

        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

        if rank == 0:
            tsqa_full = _safe_load("ChengsenWang/TSQA", "train", cache_dir)
        if is_distributed:
            dist.barrier()
        if rank != 0:
            tsqa_full = _safe_load("ChengsenWang/TSQA", "train", cache_dir)
        if is_distributed:
            dist.barrier()

        tsqa_tv, tsqa_test_hf = tsqa_full.train_test_split(test_size=TEST_FRAC, seed=42).values()
        tsqa_train_hf, tsqa_val_hf = tsqa_tv.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        def _cap(lst):
            return self.rng.sample(lst, min(cap, len(lst))) if cap else lst

        return (
            _cap([{"type": "tsqa", **r} for r in tsqa_train_hf]),
            _cap([{"type": "tsqa", **r} for r in tsqa_val_hf]),
            _cap([{"type": "tsqa", **r} for r in tsqa_test_hf]),
        )

    # ------------------------------------------------------------------
    # Chronos lazy-stub builder (no processing at index-build time)
    # ------------------------------------------------------------------

    def _make_chronos_stubs(self, ds: Dataset, split_name: str, cap) -> List[dict]:
        """Build lightweight Chronos index stubs — NO row processing, NO disk access.

        Each stub is ``{"type": "chronos", "_raw_idx": i, "_chronos_split": split_name}``.
        The actual window crop + normalisation happens on-demand in ``_format_sample``.

        When ``self.chronos_window_stride`` is set *and* this is the train split,
        each row is additionally expanded into one stub per window start so that
        ``len(dataset)`` scales with the total number of windows. Val/test keep
        the original one-stub-per-row semantics; their deterministic last-window
        logic lives in ``_sample_phase1_window``.
        """
        n = cap if cap else len(ds)
        n = min(n, len(ds))
        indices = self.rng.sample(range(len(ds)), n)

        stride = self.chronos_window_stride
        if stride is None or split_name != "train":
            return [
                {"type": "chronos", "_raw_idx": i, "_chronos_split": split_name}
                for i in indices
            ]

        history_len = max(self.valid_window_lens)
        min_total = history_len + self.patch_size
        stride = max(1, int(stride))
        stubs: List[dict] = []
        for i in indices:
            try:
                T = len(ds[i]["target"])
            except Exception:
                T = 0
            if T < min_total:
                # Series too short for any full window — emit a single stub so
                # _sample_phase1_window's short-series fallback can still run.
                stubs.append({
                    "type": "chronos",
                    "_raw_idx": i,
                    "_chronos_split": split_name,
                })
                continue
            max_start = T - min_total
            starts = list(range(0, max_start + 1, stride))
            if not starts or starts[-1] != max_start:
                starts.append(max_start)   # always include the final window
            for s in starts:
                stubs.append({
                    "type": "chronos",
                    "_raw_idx": i,
                    "_chronos_split": split_name,
                    "_window_start": int(s),
                })

        print(
            f"[{self.__class__.__name__}] train stride={stride}: "
            f"{len(indices)} rows -> {len(stubs)} window stubs "
            f"(history_len={history_len}, patch_size={self.patch_size})"
        )
        return stubs

    # ------------------------------------------------------------------
    # _load_splits: builds the full lazy index
    # ------------------------------------------------------------------

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = self.max_samples_per_source

        # ── Load and cache raw Chronos HF Datasets (memory-mapped, read-only) ──
        # Use the class-level attribute so all split instances share one copy.
        if not hasattr(Phase1JointDataset, "_chronos_raw_cache"):
            tr, va, te = self._load_or_build_chronos_raw()
            Phase1JointDataset._chronos_raw_cache = {"train": tr, "validation": va, "test": te}
        cache = Phase1JointDataset._chronos_raw_cache

        # ── TSQA (small, eager) ──
        tsqa_t, tsqa_v, tsqa_te = self._load_tsqa_lists(cap)

        # ── Chronos stubs (lazy) ──
        c_t  = self._make_chronos_stubs(cache["train"],      "train",      cap)
        c_v  = self._make_chronos_stubs(cache["validation"], "validation", cap)
        c_te = self._make_chronos_stubs(cache["test"],       "test",       cap)

        # Subsample val per source to preserve source ratio
        if self.val_fraction < 1.0:
            tsqa_v = tsqa_v[:max(1, int(len(tsqa_v) * self.val_fraction))]
            c_v    = c_v[:max(1, int(len(c_v)    * self.val_fraction))]

        def _merge(a, b):
            combined = a + b
            self.rng.shuffle(combined)
            return combined

        train = _merge(tsqa_t, c_t)
        val   = _merge(tsqa_v, c_v)
        test  = _merge(tsqa_te, c_te)

        print(
            f"[{self.__class__.__name__}] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  tsqa    — train={len(tsqa_t)}, val={len(tsqa_v)}, test={len(tsqa_te)}\n"
            f"  chronos — train={len(c_t)} (lazy), val={len(c_v)} (lazy), test={len(c_te)} (lazy)"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Format dispatch — Chronos stubs are resolved here, on-demand
    # ------------------------------------------------------------------

    def _format_sample(self, row):
        if row.get("type") == "chronos" and "_raw_idx" in row:
            # Lazy stub: fetch the raw series and process it now
            split = row["_chronos_split"]
            raw_row = Phase1JointDataset._chronos_raw_cache[split][row["_raw_idx"]]
            window_start = row.get("_window_start")
            processed = self._process_chronos_row(
                raw_row, split, window_start=window_start
            )
            if processed is None:
                # Defensive fallback: synthesize the smallest valid window rather
                # than returning an empty sample that would later break collation.
                series = torch.tensor(raw_row["target"], dtype=torch.float32)
                processed = self._sample_phase1_window(series, "validation")
                if processed is None:
                    series = torch.zeros(2 * self.patch_size, dtype=torch.float32)
                    processed = self._sample_phase1_window(series, "validation")
            row = processed

        # super() calls _get_text_time_series_prompt_list, which sets
        # row["_norm_mean"] and row["_norm_std"] as a side-effect.
        d = super()._format_sample(row)
        d["type"]      = row.get("type", "unknown")
        d["norm_mean"] = row.get("_norm_mean", row.get("norm_mean", 0.0))
        d["norm_std"]  = row.get("_norm_std",  row.get("norm_std",  1.0))
        if row.get("ts_future") is not None:
            d["ts_future"] = row["ts_future"]
        if d.get("time_series"):
            d["ts_already_normalized"] = True
            # For multivariate [L, C] series the norm stats track channel-0 only;
            # pad_and_apply_batch will compute per-channel stats for channels 1..C-1.
            d["time_series_norm_stats"] = [(d["norm_mean"], d["norm_std"])]
        return d

    def _get_answer(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Answer"]
        # Phase 1 Chronos uses causal next-patch pretraining on the input
        # sequence itself, so there is no explicit future-answer text.
        return self.EOS_TOKEN

    def _get_pre_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Question"]
        return ""

    def _get_post_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return "Predict the " + row["Task"] + " Answer:"
        return self.EOS_TOKEN

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if row["type"] == "chronos":
            # Already z-normalized per channel in _process_chronos_row.
            # series shape: [L] for univariate, [L, C] for multivariate.
            series = torch.as_tensor(row["series"], dtype=torch.float32)
            row["_norm_mean"] = row.get("norm_mean", 0.0)
            row["_norm_std"]  = row.get("norm_std",  1.0)
            n_channels = series.shape[1] if series.dim() == 2 else 1
            if n_channels > 1:
                prompt_text = (
                    f"This is a {n_channels}-channel multivariate time series. <ts>"
                )
            else:
                prompt_text = (
                    f"This is the time series, it has mean {row['_norm_mean']:.4f}"
                    f" and std {row['_norm_std']:.4f}. <ts>"
                )
            return [TextTimeSeriesPrompt(prompt_text, series.numpy())]

        # tsqa: keep preprocessing aligned with standalone TSQADataset.
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)
        means = series.mean(dim=0, keepdim=True)
        stds = series.std(dim=0, keepdim=True)
        series = (series - means) / (stds + 1e-8)

        mean_val = means.flatten()[0].item()
        std_val = stds.flatten()[0].item()
        row["_norm_mean"] = mean_val
        row["_norm_std"] = std_val
        return [TextTimeSeriesPrompt(
            f"This is the time series, it has mean {mean_val:.4f} and std {std_val:.4f}.",
            series.numpy(),
        )]


class Phase1JointExtendDataset(Phase1JointDataset):
    """Extended Phase-1 dataset: Chronos next-patch pretraining + TSQA MCQ + M4 captioning.

    Same as Phase1JointDataset but adds M4 captioning as a third task.
    All three splits are balanced to the smallest sub-split.
    CE loss is applied to TSQA and M4; MSE to Chronos.
    """

    def _load_splits(self) -> Tuple[list, list, list]:
        # Get TSQA (eager) + Chronos (lazy stubs) from parent
        train, val, test = super()._load_splits()

        # ---- M4 (stage2_captioning) — eager, manageable size ----
        cap = getattr(self, "max_samples_per_source", None)

        def _cap(lst):
            return self.rng.sample(lst, min(cap, len(lst))) if cap else lst

        from opentslm.time_series_datasets.m4.m4_loader import load_all_m4_data, create_combined_dataset
        data_dict = load_all_m4_data()
        m4_train_raw, m4_val_raw, m4_test_raw = create_combined_dataset(data_dict, seed=42)
        m4_t  = _cap([{"type": "m4", **r} for r in m4_train_raw])
        m4_v  = _cap([{"type": "m4", **r} for r in m4_val_raw])
        m4_te = _cap([{"type": "m4", **r} for r in m4_test_raw])

        # Subsample M4 val to match val_fraction applied to other sources
        frac = getattr(self, "val_fraction", 1.0)
        if frac < 1.0:
            m4_v = m4_v[:max(1, int(len(m4_v) * frac))]

        def _merge(base, m4):
            combined = base + m4
            self.rng.shuffle(combined)
            return combined

        train = _merge(train, m4_t)
        val   = _merge(val,   m4_v)
        test  = _merge(test,  m4_te)

        print(
            f"[Phase1JointExtendDataset] after M4: train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  m4 — train={len(m4_t)}, val={len(m4_v)}, test={len(m4_te)}"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Override prompt/answer helpers to handle the added "m4" task type
    # ------------------------------------------------------------------

    def _get_answer(self, row) -> str:
        if row["type"] == "m4":
            return row["caption"]
        return super()._get_answer(row)

    def _get_pre_prompt(self, row) -> str:
        if row["type"] == "m4":
            return "You are an expert in time series analysis."
        return super()._get_pre_prompt(row)

    def _get_post_prompt(self, row) -> str:
        if row["type"] == "m4":
            return "Please generate a detailed caption for this time-series, describing it as accurately as possible."
        return super()._get_post_prompt(row)

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if row["type"] == "m4":
            series = row["series"]
            if isinstance(series, list):
                series_tensor = torch.tensor(series, dtype=torch.float32)
            else:
                series_tensor = series
            mean = series_tensor.mean().item()
            std = series_tensor.std().item()
            if std > 0:
                normalized = ((series_tensor - mean) / std).numpy()
            else:
                normalized = (series_tensor - mean).numpy()
            row["_norm_mean"] = mean
            row["_norm_std"] = std if std > 0 else 1.0
            text_prompt = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
            return [TextTimeSeriesPrompt(text_prompt, normalized)]
        return super()._get_text_time_series_prompt_list(row)


class Phase1JointTsqaM4Dataset(Phase1JointExtendDataset):
    """Joint Phase-1 dataset with only TSQA and M4 captioning.

    Reuses the TSQA formatting from Phase1JointDataset and the M4 prompt /
    answer handling from Phase1JointExtendDataset, but intentionally skips
    building any Chronos samples so the stage stays CE-only.
    """

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = getattr(self, "max_samples_per_source", None)

        def _cap(lst):
            return self.rng.sample(lst, min(cap, len(lst))) if cap else lst

        tsqa_t, tsqa_v, tsqa_te = self._load_tsqa_lists(cap)

        from opentslm.time_series_datasets.m4.m4_loader import (
            create_combined_dataset,
            load_all_m4_data,
        )

        data_dict = load_all_m4_data()
        m4_train_raw, m4_val_raw, m4_test_raw = create_combined_dataset(
            data_dict, seed=42
        )
        m4_t = _cap([{"type": "m4", **r} for r in m4_train_raw])
        m4_v = _cap([{"type": "m4", **r} for r in m4_val_raw])
        m4_te = _cap([{"type": "m4", **r} for r in m4_test_raw])

        frac = getattr(self, "val_fraction", 1.0)
        if frac < 1.0:
            tsqa_v = tsqa_v[: max(1, int(len(tsqa_v) * frac))]
            m4_v = m4_v[: max(1, int(len(m4_v) * frac))]

        def _merge(*parts):
            combined = []
            for part in parts:
                combined.extend(part)
            self.rng.shuffle(combined)
            return combined

        train = _merge(tsqa_t, m4_t)
        val = _merge(tsqa_v, m4_v)
        test = _merge(tsqa_te, m4_te)

        print(
            f"[{self.__class__.__name__}] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  tsqa — train={len(tsqa_t)}, val={len(tsqa_v)}, test={len(tsqa_te)}\n"
            f"  m4   — train={len(m4_t)}, val={len(m4_v)}, test={len(m4_te)}"
        )
        return train, val, test


class Phase1JointTsqaOnlyDataset(Phase1JointDataset):
    """Joint Phase-1 dataset with only TSQA (no M4, no Chronos).

    CE-only training on TSQA multiple-choice QA.
    """

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = getattr(self, "max_samples_per_source", None)

        tsqa_t, tsqa_v, tsqa_te = self._load_tsqa_lists(cap)

        frac = getattr(self, "val_fraction", 1.0)
        if frac < 1.0:
            tsqa_v = tsqa_v[:max(1, int(len(tsqa_v) * frac))]

        self.rng.shuffle(tsqa_t)
        self.rng.shuffle(tsqa_v)
        self.rng.shuffle(tsqa_te)

        print(
            f"[{self.__class__.__name__}] train={len(tsqa_t)}, val={len(tsqa_v)}, test={len(tsqa_te)}"
        )
        return tsqa_t, tsqa_v, tsqa_te


class Phase1ChronosOnlyDataset(Phase1JointDataset):
    """Phase-1 Chronos-only dataset for the MSE pretraining sub-stage.

    Identical to Phase1JointDataset but loads only Chronos stubs — TSQA and M4
    are omitted.  All samples have type ``"chronos"`` so the training loop routes
    them to ``compute_pretrain_loss`` (NTP + MSE).
    """

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = self.max_samples_per_source

        if not hasattr(Phase1JointDataset, "_chronos_raw_cache"):
            tr, va, te = self._load_or_build_chronos_raw()
            Phase1JointDataset._chronos_raw_cache = {"train": tr, "validation": va, "test": te}
        cache = Phase1JointDataset._chronos_raw_cache

        c_t  = self._make_chronos_stubs(cache["train"],      "train",      cap)
        c_v  = self._make_chronos_stubs(cache["validation"], "validation", cap)
        c_te = self._make_chronos_stubs(cache["test"],       "test",       cap)

        if self.val_fraction < 1.0:
            c_v = c_v[:max(1, int(len(c_v) * self.val_fraction))]

        self.rng.shuffle(c_t)
        self.rng.shuffle(c_v)
        self.rng.shuffle(c_te)

        print(
            f"[Phase1ChronosOnlyDataset] train={len(c_t)}, val={len(c_v)}, test={len(c_te)} (all lazy Chronos)"
        )
        return c_t, c_v, c_te


class Phase1ChronosFuturePatchDataset(Phase1ChronosOnlyDataset):
    """Chronos-only dataset that always reserves one real future patch target.

    This is the cleaner next-patch objective: the model sees only the history
    window, and supervision is applied only to the first unseen patch beyond
    that history. When a series is too short to supply a real future patch, we
    pad it to two patches so the sample still has one history patch and one
    target patch.
    """

    def __init__(
        self,
        *args,
        min_history_patches: int = 8,
        norm_std_floor: float = 1e-3,
        norm_clip: float | None = 20.0,
        **kwargs,
    ):
        self.min_history_patches = max(1, int(min_history_patches))
        self.norm_std_floor = max(float(norm_std_floor), 1e-8)
        self.norm_clip = None if norm_clip is None else float(norm_clip)
        super().__init__(*args, **kwargs)

    def _normalize_history_and_future(
        self,
        history_raw: torch.Tensor,
        future_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Numerically stable z-normalisation for next-patch regression.

        Handles both univariate ``[T]`` and multivariate ``[T, C]`` history
        windows. For MV, stats are per-channel (matching the parent class and
        the eval dataset), so the channel structure is preserved.
        """
        history_raw = history_raw.float()
        future_raw = future_raw.float()

        if history_raw.dim() == 1:
            norm_mean_t = history_raw.mean()
            norm_std_t = history_raw.std().clamp(min=self.norm_std_floor)
            history_norm = (history_raw - norm_mean_t) / norm_std_t
            future_norm = (future_raw - norm_mean_t) / norm_std_t
            norm_mean = norm_mean_t.item()
            norm_std = norm_std_t.item()
        else:
            # [T, C] — per-channel normalisation. Scalar mean/std returned for
            # logging/plot metadata mirror channel-0 stats, as elsewhere in the
            # repo.
            ch_mean = history_raw.mean(dim=0, keepdim=True)                        # [1, C]
            ch_std = history_raw.std(dim=0, keepdim=True).clamp(min=self.norm_std_floor)
            history_norm = (history_raw - ch_mean) / ch_std                        # [T, C]
            future_norm = (future_raw - ch_mean) / ch_std                          # [P, C]
            norm_mean = ch_mean[0, 0].item()
            norm_std = ch_std[0, 0].item()

        if self.norm_clip is not None:
            history_norm = history_norm.clamp(-self.norm_clip, self.norm_clip)
            future_norm = future_norm.clamp(-self.norm_clip, self.norm_clip)

        return history_norm, future_norm, norm_mean, norm_std

    def _sample_phase1_window(
        self,
        series: torch.Tensor,
        mode: str = "train",
        window_start: int | None = None,
    ) -> dict | None:
        is_mv = series.dim() == 2
        T = series.shape[0]
        min_history_len = self.min_history_patches * self.patch_size
        min_total_len = min_history_len + self.patch_size

        if T <= 0:
            if is_mv:
                C = series.shape[1] if series.dim() == 2 else 1
                series = torch.zeros(min_total_len, C, dtype=torch.float32)
            else:
                series = torch.zeros(min_total_len, dtype=torch.float32)
            T = series.shape[0]

        if T < 2 * self.patch_size:
            window = _last_value_impute(series)
            cur_len = window.shape[0]
            pad_len = max(0, min_total_len - cur_len)
            if cur_len == 0:
                window = (torch.zeros(1, series.shape[1], dtype=torch.float32)
                          if is_mv else torch.zeros(1, dtype=torch.float32))
                cur_len = 1
                pad_len = min_total_len - 1
            if pad_len > 0:
                if is_mv:
                    pad_block = window[-1:].expand(pad_len, -1)           # [pad_len, C]
                else:
                    pad_val = float(window[-1])
                    pad_block = window.new_full((pad_len,), pad_val)      # [pad_len]
                window = torch.cat([window, pad_block], dim=0)
            history_raw = window[:min_history_len]
            future_raw = window[min_history_len:min_total_len]
        else:
            max_history_len = ((T - self.patch_size) // self.patch_size) * self.patch_size
            max_history_len = max(self.patch_size, max_history_len)
            candidates = [
                w for w in self.valid_window_lens
                if min_history_len <= w <= max_history_len
            ]
            if candidates:
                history_len = self.rng.choice(candidates) if mode == "train" else max(candidates)
            else:
                history_len = max_history_len

            max_start = max(0, T - history_len - self.patch_size)
            if window_start is not None:
                start = max(0, min(int(window_start), max_start))
            else:
                start = self.rng.randint(0, max_start) if mode == "train" and max_start > 0 else max_start

            history_raw = _last_value_impute(series[start : start + history_len])
            future_raw = _last_value_impute(
                series[start + history_len : start + history_len + self.patch_size]
            )

        history_norm, future_norm, norm_mean, norm_std = self._normalize_history_and_future(
            history_raw,
            future_raw,
        )

        # ts_future shape must match what compute_pretrain_loss expects:
        # univariate -> [patch_size], multivariate -> [C, patch_size].
        if history_norm.dim() == 2:
            ts_future = future_norm.transpose(0, 1).to(dtype=torch.float32).contiguous()
        else:
            ts_future = future_norm.to(dtype=torch.float32)

        # n_patches counts time-steps only, regardless of channels.
        n_patches = history_norm.shape[0] // self.patch_size

        return {
            "type": "chronos",
            "series": history_norm.numpy(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_patches": n_patches,
            "ts_future": ts_future,
        }


class Phase1ChronosSeq2SeqDataset(Phase1JointDataset):
    """Seq2seq Phase-1 dataset for Chronos forecasting.

    Each sample is structured as:
      Input  : "You are given a time series of {T_in} steps ..." + input patches
      Output : "Predict the next {T_out} steps:"               + target patches

    This enables a Transfusion-inspired autoregressive setup: the LLM processes
    the full combined sequence causally, so hidden states at target patch positions
    carry context from the text prompt AND all input patches. Forecast training
    is driven by next-patch MSE on the target horizon; this stage does not use
    a diffusion objective.

    Args:
        pred_len: int or list of ints (steps).  Each value generates one sample
            per Chronos series at that forecasting horizon.  Defaults to
            ``PRED_LEN`` from ``model_config``.  Non-divisible values are
            floor-divided to the nearest patch boundary automatically.
    """

    def __init__(self, *args, pred_len=None, **kwargs):
        super().__init__(*args, **kwargs)
        from opentslm.model_config import PRED_LEN as _DEFAULT
        if pred_len is None:
            pred_len = _DEFAULT
        self.pred_lens: List[int] = (
            [int(pred_len)] if isinstance(pred_len, (int, float))
            else [int(p) for p in pred_len]
        )
        # Floor-divide so pred_len never needs to be exactly divisible by patch_size.
        self.n_pred_patches_list: List[int] = [
            max(1, p // self.patch_size) for p in self.pred_lens
        ]

    # ------------------------------------------------------------------
    # Stub builder — one stub per (series, pred_len) combination
    # ------------------------------------------------------------------

    def _make_seq2seq_stubs(self, ds: Dataset, split_name: str, cap) -> List[dict]:
        n = min(cap, len(ds)) if cap else len(ds)
        indices = self.rng.sample(range(len(ds)), n)
        stubs = []
        for i in indices:
            for pred_len, n_pred in zip(self.pred_lens, self.n_pred_patches_list):
                stubs.append({
                    "type": "chronos",
                    "_raw_idx": i,
                    "_chronos_split": split_name,
                    "_pred_len": pred_len,
                    "_n_pred_patches": n_pred,
                })
        return stubs

    # ------------------------------------------------------------------
    # _load_splits — Chronos-only, expanded across pred_lens
    # ------------------------------------------------------------------

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = self.max_samples_per_source

        if not hasattr(Phase1JointDataset, "_chronos_raw_cache"):
            tr, va, te = self._load_or_build_chronos_raw()
            Phase1JointDataset._chronos_raw_cache = {"train": tr, "validation": va, "test": te}
        cache = Phase1JointDataset._chronos_raw_cache

        c_t  = self._make_seq2seq_stubs(cache["train"],      "train",      cap)
        c_v  = self._make_seq2seq_stubs(cache["validation"], "validation", cap)
        c_te = self._make_seq2seq_stubs(cache["test"],       "test",       cap)

        if self.val_fraction < 1.0:
            c_v = c_v[:max(1, int(len(c_v) * self.val_fraction))]

        self.rng.shuffle(c_t)
        self.rng.shuffle(c_v)
        self.rng.shuffle(c_te)

        print(
            f"[Phase1ChronosSeq2SeqDataset] pred_lens={self.pred_lens} "
            f"train={len(c_t)}, val={len(c_v)}, test={len(c_te)}"
        )
        return c_t, c_v, c_te

    # ------------------------------------------------------------------
    # Window sampling — splits into input + target
    # ------------------------------------------------------------------

    def _sample_seq2seq_window(
        self, series: torch.Tensor, n_pred_patches: int, mode: str = "train"
    ) -> dict | None:
        """Crop a window and split into input (history) + target (forecast)."""
        pred_steps = n_pred_patches * self.patch_size
        # Need at least 1 input patch + target patches
        min_window = pred_steps + self.patch_size

        candidates = [w for w in self.valid_window_lens if w >= min_window and len(series) >= w]

        if candidates:
            window_len = self.rng.choice(candidates) if mode == "train" else max(candidates)
            start = self.rng.randint(0, len(series) - window_len) if mode == "train" else len(series) - window_len
            window = series[start: start + window_len]
            window = _last_value_impute(window)
        elif len(series) >= min_window:
            # Series shorter than any configured window — use what we have
            window_len = (len(series) // self.patch_size) * self.patch_size
            window_len = max(window_len, min_window)
            window_len = min(window_len, len(series))
            window_len = (window_len // self.patch_size) * self.patch_size
            if window_len < min_window:
                window_len = min_window
            start = 0 if len(series) <= window_len else (
                self.rng.randint(0, len(series) - window_len) if mode == "train" else len(series) - window_len
            )
            window = series[start: start + window_len]
            window = _last_value_impute(window)
        else:
            # Very short — pad to minimum
            window = _last_value_impute(series)
            pad_len = min_window - window.numel()
            pad_val = window[-1] if window.numel() > 0 else 0.0
            window = torch.cat([window, window.new_full((pad_len,), float(pad_val))])

        window_norm, norm_mean, norm_std = self._znorm(window)

        # Split: last pred_steps = target, everything before = input
        # Trim input so it is a multiple of patch_size
        in_steps = (len(window_norm) - pred_steps) // self.patch_size * self.patch_size
        in_steps = max(in_steps, self.patch_size)  # at least 1 input patch
        input_series  = window_norm[:in_steps]
        target_series = window_norm[in_steps: in_steps + pred_steps]

        return {
            "type": "chronos",
            "input_series":    input_series.numpy(),
            "target_series":   target_series.numpy(),
            "norm_mean":       norm_mean,
            "norm_std":        norm_std,
            "n_input_patches": in_steps // self.patch_size,
            "n_target_patches": n_pred_patches,
        }

    # ------------------------------------------------------------------
    # _format_sample — resolve stub → seq2seq sample
    # ------------------------------------------------------------------

    def _format_sample(self, row):
        if row.get("type") == "chronos" and "_raw_idx" in row:
            split     = row["_chronos_split"]
            n_pred    = row["_n_pred_patches"]
            raw_row   = Phase1JointDataset._chronos_raw_cache[split][row["_raw_idx"]]
            series    = torch.tensor(raw_row["target"], dtype=torch.float32)
            processed = self._sample_seq2seq_window(series, n_pred, split)
            if processed is None:
                processed = self._sample_seq2seq_window(
                    torch.zeros(n_pred * self.patch_size + self.patch_size, dtype=torch.float32),
                    n_pred, "train",
                )
            row = processed

        d = super(Phase1JointDataset, self)._format_sample(row)
        d["type"]      = row.get("type", "unknown")
        d["norm_mean"] = row.get("norm_mean", 0.0)
        d["norm_std"]  = row.get("norm_std",  1.0)
        if d.get("time_series"):
            d["ts_already_normalized"]    = True
            d["time_series_norm_stats"]   = [(d["norm_mean"], d["norm_std"])]
        return d

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _get_pre_prompt(self, row) -> str:
        return ""

    def _get_post_prompt(self, row) -> str:
        return self.EOS_TOKEN

    def _get_answer(self, row) -> str:
        return self.EOS_TOKEN

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        input_series  = torch.as_tensor(row["input_series"],  dtype=torch.float32)
        target_series = torch.as_tensor(row["target_series"], dtype=torch.float32)
        row["_norm_mean"] = row.get("norm_mean", 0.0)
        row["_norm_std"]  = row.get("norm_std",  1.0)

        n_in  = row["n_input_patches"]
        n_out = row["n_target_patches"]
        mean  = row["_norm_mean"]
        std   = row["_norm_std"]

        return [
            TextTimeSeriesPrompt(
                f"You are given a time series of {n_in * self.patch_size} steps "
                f"(mean={mean:.4f}, std={std:.4f}):",
                input_series.numpy(),
            ),
            TextTimeSeriesPrompt(
                f"Predict the next {n_out * self.patch_size} steps:",
                target_series.numpy(),
            ),
        ]


if __name__ == "__main__":
    import statistics

    PATCH_SIZE = 16

    # ── TSQA ──────────────────────────────────────────────────────────────────
    print("Loading TSQA …")
    from datasets import load_dataset as _load_dataset
    tsqa_ds = _load_dataset("ChengsenWang/TSQA", split="train")
    # row["Series"] is a JSON string; its first dimension is always the time axis
    # whether the array is 1-D (T,) or 2-D (T, D).
    tsqa_lengths = [len(json.loads(row["Series"])) for row in tsqa_ds]
    tsqa_patches = [t / PATCH_SIZE for t in tsqa_lengths]
    print(f"  TSQA  — {len(tsqa_patches)} series")
    print(f"    series length : mean={statistics.mean(tsqa_lengths):.1f}, "
          f"min={min(tsqa_lengths)}, max={max(tsqa_lengths)}")
    print(f"    patches (÷{PATCH_SIZE}): mean={statistics.mean(tsqa_patches):.2f}, "
          f"min={min(tsqa_patches):.2f}, max={max(tsqa_patches):.2f}")

    # ── M4 ────────────────────────────────────────────────────────────────────
    print("\nLoading M4 …")
    from opentslm.time_series_datasets.m4.m4_loader import load_all_m4_data, create_combined_dataset
    data_dict = load_all_m4_data()
    m4_train, m4_val, m4_test = create_combined_dataset(data_dict, seed=42)
    from datasets import concatenate_datasets as _concat
    all_m4 = _concat([m4_train, m4_val, m4_test])
    m4_lengths = [len(row["series"]) for row in all_m4]
    m4_patches = [t / PATCH_SIZE for t in m4_lengths]
    print(f"  M4    — {len(m4_patches)} series (train+val+test)")
    print(f"    series length : mean={statistics.mean(m4_lengths):.1f}, "
          f"min={min(m4_lengths)}, max={max(m4_lengths)}")
    print(f"    patches (÷{PATCH_SIZE}): mean={statistics.mean(m4_patches):.2f}, "
          f"min={min(m4_patches):.2f}, max={max(m4_patches):.2f}")
