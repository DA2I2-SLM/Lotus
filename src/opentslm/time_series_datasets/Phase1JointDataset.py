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

import numpy as np
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
    # "dominick": ["target"],
    # "monash_fred_md": ["target"],
    # "electricity_15min": ["consumption_kW"],
    # "ercot": ["target"],
    # "exchange_rate": ["target"],
    "ushcn_daily": ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
    # "weatherbench_daily": ['target'],
    # "weatherbench_weekly": ['target'],
    # "wiki_daily_100k": ['target'],
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
        **kwargs,
    ):
        self.max_samples_per_source = max_samples_per_source
        self.val_fraction = val_fraction
        self.data_seed = data_seed
        self.patch_size = int(patch_size)
        self.window_lens = [int(w) for w in WINDOW_LENS]

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
        if len(target_keys) == 1:
            key = target_keys[0]
            if key != "target":
                ds = ds.rename_column(key, "target")
            ds = ds.select_columns(["target"])
            ds = ds.cast_column("target", Sequence(feature=Value(dtype="float32")))
        else:
            # Multivariate: keep all channels together per row.
            # Each output row has a "targets" field = list of C sequences so that
            # _sample_phase1_window_mv can window and normalise all channels jointly.
            # The old flattening approach ([seq for k in keys for seq in batch[k]])
            # incorrectly promoted each channel to a separate dataset row, causing
            # every sample in pad_and_apply_batch to appear univariate (channel_id=0).
            keys = target_keys
            ds = ds.map(
                lambda batch: {
                    "targets": [
                        [batch[k][i] for k in keys]
                        for i in range(len(batch[keys[0]]))
                    ]
                },
                batched=True,
                remove_columns=ds.column_names,
            )
            # "targets" is a list-of-lists (C channels per row); cast each channel
            # sequence as float32.  HF Datasets represents this as Sequence(Sequence(float32)).
            ds = ds.cast_column(
                "targets",
                Sequence(feature=Sequence(feature=Value(dtype="float32"))),
            )
        return ds

    def _load_or_build_chronos_raw(
        self,
    ) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset]]:
        """Return three {dataset_name: HF Dataset} dicts for train / val / test.

        Each dataset is preprocessed and split **independently** — no concatenation,
        no shared schema constraint.  This fixes two problems with the old design:

        1. Schema conflict: univariate datasets use a "target" column while
           multivariate ones (e.g. ushcn_daily) use "targets" (list-of-lists).
           concatenate_datasets() fails when schemas differ.

        2. Slow startup: the old approach loaded ALL datasets into one Arrow table
           and then did a global train_test_split, which dominated startup time for
           large sources (wiki_daily_100k, ushcn_daily).  Per-dataset splits are
           built and loaded independently, so a new source only costs its own build.

        Storage layout (under PREPROCESSED_SPLITS_DIR/_splits/):
            {name}/_DONE        — sentinel written after a successful build
            {name}/train/       — Arrow dataset
            {name}/val/         — Arrow dataset
            {name}/test/        — Arrow dataset
        """
        rank = int(os.environ.get("RANK", 0))
        splits_root = os.path.join(PREPROCESSED_SPLITS_DIR, "_splits")

        # ── Rank-0 builds any missing per-dataset splits ──────────────────
        if rank == 0:
            os.makedirs(splits_root, exist_ok=True)
            for name, keys in CHRONOS_DATASET_REGISTRY.items():
                done_flag = os.path.join(splits_root, name, "_DONE")
                if os.path.exists(done_flag):
                    print(f"[Phase1Joint] '{name}': cached splits found, skipping.")
                    continue
                print(f"[Phase1Joint] Building splits for '{name}'…")
                try:
                    ds = load_dataset(
                        f"{BASE_DATA_PATH}/{name}",
                        split="train",
                        trust_remote_code=True,
                        keep_in_memory=True,
                    )
                    ds = self._normalize_to_target(ds, keys)
                    train_val, test = ds.train_test_split(test_size=TEST_FRAC, seed=42).values()
                    train, val = train_val.train_test_split(
                        test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
                    ).values()
                    part_dir = os.path.join(splits_root, name)
                    os.makedirs(part_dir, exist_ok=True)
                    train.save_to_disk(os.path.join(part_dir, "train"))
                    val.save_to_disk(os.path.join(part_dir, "val"))
                    test.save_to_disk(os.path.join(part_dir, "test"))
                    open(done_flag, "w").write("done\n")
                    print(
                        f"[Phase1Joint]   -> '{name}': "
                        f"{len(train)}/{len(val)}/{len(test)} train/val/test rows."
                    )
                    del ds, train_val, train, val, test
                except Exception as exc:
                    print(f"[Phase1Joint] WARNING: failed to build splits for '{name}': {exc}")

        if dist.is_initialized():
            dist.barrier()

        # ── All ranks load per-dataset splits ────────────────────────────
        # keep_in_memory=True: Arrow data copied into RAM so forked DataLoader
        # workers never mmap NFS files (NFS + mmap in forked processes deadlocks).
        train_dsets: Dict[str, Dataset] = {}
        val_dsets:   Dict[str, Dataset] = {}
        test_dsets:  Dict[str, Dataset] = {}

        for name in CHRONOS_DATASET_REGISTRY:
            done_flag = os.path.join(splits_root, name, "_DONE")
            if not os.path.exists(done_flag):
                if rank == 0:
                    print(f"[Phase1Joint] WARNING: splits for '{name}' missing — skipping.")
                continue
            try:
                part_dir = os.path.join(splits_root, name)
                train_dsets[name] = load_from_disk(os.path.join(part_dir, "train"), keep_in_memory=True)
                val_dsets[name]   = load_from_disk(os.path.join(part_dir, "val"),   keep_in_memory=True)
                test_dsets[name]  = load_from_disk(os.path.join(part_dir, "test"),  keep_in_memory=True)
                if rank == 0:
                    print(f"[Phase1Joint]   -> '{name}': {len(train_dsets[name])} train rows loaded.")
            except Exception as exc:
                if rank == 0:
                    print(f"[Phase1Joint] WARNING: failed to load splits for '{name}': {exc}")

        if not train_dsets:
            raise RuntimeError("[Phase1Joint] No Chronos datasets could be loaded!")

        return train_dsets, val_dsets, test_dsets

    @staticmethod
    def _znorm(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Z-normalize a 1-D tensor. Returns (normalized, mean, std).
        Uses a consistent epsilon=1e-8 across all tasks."""
        if x.dim() > 1:
            x = x.reshape(-1)          # flatten multivariate to 1-D for scalar stats
        m = x.mean()
        s = x.std().clamp(min=1e-8)
        return (x - m) / s, m.item(), s.item()

    def _sample_phase1_window(self, series: torch.Tensor, mode: str = "train") -> dict | None:
        """Prepare one full-window Chronos sample for Phase 1 next-patch training.

        Also grabs the first patch *beyond* the window as ``ts_future`` whenever
        the series is long enough.  This closes the training/inference gap:
        ``compute_pretrain_loss`` uses it to supervise decoder position N_p-1
        (the last history position) — the exact position read at inference time.
        Without this, position N_p-1 receives zero gradient during Phase 1 and
        collapses to the normalised mean (flat-line forecast).
        """
        T = len(series)
        candidates = [w for w in self.valid_window_lens if T >= w]

        if candidates:
            window_len = (
                self.rng.choice(candidates)
                if mode == "train"
                else max(candidates)
            )
            start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
            window = series[start: start + window_len]
            window = _last_value_impute(window)
        else:
            min_window_len = 2 * self.patch_size
            if T <= 0:
                # Empty series are rare, but returning an invalid sample breaks
                # DDP collation. Use a tiny constant window so the batch keeps
                # moving and contributes a near-zero training signal.
                window = torch.zeros(min_window_len, dtype=torch.float32)
                window_len = min_window_len
                start = 0
            elif T >= min_window_len:
                window_len = max(min_window_len, (T // self.patch_size) * self.patch_size)
                start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
                window = series[start: start + window_len]
                window = _last_value_impute(window)
            else:
                # Very short series: pad by repeating the last observed value so the
                # sample still provides at least one valid next-patch target.
                window = _last_value_impute(series)
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
            ts_future = torch.tensor(
                ((future_raw - norm_mean) / max(norm_std, 1e-8)).tolist(),
                dtype=torch.float32,
            )

        return {
            "type": "chronos",
            "series": window_norm.numpy(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_patches": window_len // self.patch_size,
            "ts_future": ts_future,   # [patch_size] tensor or None
        }

    def _process_chronos_row(self, row: dict, mode: str = "train") -> dict | None:
        import os as _os
        _debug = _os.environ.get("_DEBUG_CHRONOS_MV", "0") == "1"
        if "targets" in row:
            # Multivariate row — list of C channel sequences (from _normalize_to_target).
            series_list = [torch.tensor(ch, dtype=torch.float32) for ch in row["targets"]]
            if _debug:
                print(
                    f"[DEBUG mv] n_channels={len(series_list)}, "
                    f"lengths={[len(s) for s in series_list]}", flush=True,
                )
            return self._sample_phase1_window_mv(series_list, mode)
        if _debug:
            print(f"[DEBUG uv] row keys={list(row.keys())}", flush=True)
        series = torch.tensor(row["target"], dtype=torch.float32)
        return self._sample_phase1_window(series, mode)

    def _sample_phase1_window_mv(
        self, series_list: list, mode: str = "train"
    ) -> dict | None:
        """Multivariate analogue of _sample_phase1_window.

        Picks a window range valid for *all* channels, then normalises each
        channel independently (channel-independent z-norm, same as PatchTST /
        iTransformer convention).

        Returns a dict with:
            "series"     : list of C np.ndarray (normalised windows)
            "norm_means" : list of C floats
            "norm_stds"  : list of C floats
            "n_patches"  : int
            "ts_future"  : torch.Tensor [C, patch_size] or None
        """
        if not series_list:
            return None

        min_window_len = 2 * self.patch_size

        # Keep the channel count fixed even when some variates are entirely
        # missing for a station (common in ushcn_daily, e.g. SNOW/SNWD). Empty
        # channels are replaced with a constant zero track so the sample remains
        # multivariate instead of collapsing to the univariate fallback.
        non_empty = [torch.as_tensor(s, dtype=torch.float32).reshape(-1) for s in series_list if len(s) > 0]
        ref_len = min((len(s) for s in non_empty), default=0)
        fill_len = max(ref_len, min_window_len)

        repaired_series = []
        for s in series_list:
            s = torch.as_tensor(s, dtype=torch.float32).reshape(-1)
            if s.numel() == 0:
                repaired_series.append(torch.zeros(fill_len, dtype=torch.float32))
            else:
                repaired_series.append(s)

        # Use the shortest available channel length to define the shared crop.
        T = min(len(s) for s in repaired_series)
        C = len(series_list)

        candidates = [w for w in self.valid_window_lens if T >= w]
        if candidates:
            window_len = (
                self.rng.choice(candidates) if mode == "train" else max(candidates)
            )
            start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
        else:
            if T <= 0:
                start = 0
                window_len = min_window_len
            elif T >= min_window_len:
                window_len = max(min_window_len, (T // self.patch_size) * self.patch_size)
                start = self.rng.randint(0, T - window_len) if mode == "train" else T - window_len
            else:
                start = 0
                window_len = min_window_len

        windows_norm, norm_means, norm_stds = [], [], []
        for s in repaired_series:
            w = s[start: start + window_len]
            w = _last_value_impute(w)
            if w.numel() == 0:
                w = torch.zeros(1, dtype=torch.float32)
            if w.numel() < window_len:
                pad_val = float(w[-1].item()) if w.numel() > 0 else 0.0
                pad = w.new_full((window_len - w.numel(),), pad_val)
                w = torch.cat([w, pad], dim=0)
            wn, m, std = self._znorm(w)
            windows_norm.append(wn.numpy())
            norm_means.append(m)
            norm_stds.append(std)

        # Future patch: first patch beyond the window, per channel → [C, patch_size]
        ts_future = None
        future_start = start + window_len
        if all(future_start + self.patch_size <= len(s) for s in repaired_series):
            future_patches = []
            for s, m, std in zip(repaired_series, norm_means, norm_stds):
                f_raw = _last_value_impute(s[future_start: future_start + self.patch_size])
                f_norm = (f_raw - m) / max(std, 1e-8)
                future_patches.append(f_norm)
            ts_future = torch.stack(future_patches, dim=0)  # [C, patch_size]

        return {
            "type":       "chronos",
            "series":     windows_norm,       # list of C np.ndarray
            "norm_means": norm_means,         # list of C floats
            "norm_stds":  norm_stds,          # list of C floats
            "n_patches":  window_len // self.patch_size,
            "ts_future":  ts_future,          # [C, patch_size] or None
        }

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

    def _make_chronos_stubs(
        self, datasets_by_name: Dict[str, Dataset], split_name: str, cap
    ) -> List[dict]:
        """Build lightweight Chronos index stubs — NO row processing, NO disk access.

        Each stub carries _chronos_name so _format_sample can look up the correct
        per-dataset cache entry.  The actual window crop + normalisation happens
        on-demand in _format_sample; too-short series are handled there gracefully.
        """
        stubs = []
        for name, ds in datasets_by_name.items():
            n = min(cap, len(ds)) if cap else len(ds)
            indices = self.rng.sample(range(len(ds)), n)
            stubs.extend(
                {"type": "chronos", "_chronos_name": name, "_raw_idx": i, "_chronos_split": split_name}
                for i in indices
            )
        return stubs

    # ------------------------------------------------------------------
    # _load_splits: builds the full lazy index
    # ------------------------------------------------------------------

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = self.max_samples_per_source

        # ── Load and cache raw Chronos HF Datasets (per-dataset, memory-mapped) ──
        # Class-level so all split instances share one copy (train / val / test
        # each a {dataset_name: HF Dataset} dict).
        if not hasattr(Phase1JointDataset, "_chronos_raw_cache"):
            tr, va, te = self._load_or_build_chronos_raw()
            Phase1JointDataset._chronos_raw_cache = {"train": tr, "validation": va, "test": te}
        cache = Phase1JointDataset._chronos_raw_cache

        # ── TSQA (small, eager) ──
        tsqa_t, tsqa_v, tsqa_te = self._load_tsqa_lists(cap)

        # ── Chronos stubs (lazy, one per source row) ──
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
            # Lazy stub: fetch the raw series and process it now.
            # Cache is now {split: {dataset_name: Dataset}}; stubs carry _chronos_name.
            split = row["_chronos_split"]
            name  = row["_chronos_name"]
            raw_row = Phase1JointDataset._chronos_raw_cache[split][name][row["_raw_idx"]]
            processed = self._process_chronos_row(raw_row, split)
            if processed is None:
                # Defensive fallback: synthesize the smallest valid window rather
                # than returning an empty sample that would later break collation.
                if "targets" in raw_row:
                    series_list = [torch.tensor(t, dtype=torch.float32) for t in raw_row["targets"]]
                    processed = self._sample_phase1_window_mv(series_list, "validation")
                else:
                    series = torch.tensor(raw_row["target"], dtype=torch.float32)
                    processed = self._sample_phase1_window(series, "validation")
                if processed is None:
                    if "targets" in raw_row:
                        n_channels = max(1, len(raw_row["targets"]))
                        zero_series = [
                            torch.zeros(2 * self.patch_size, dtype=torch.float32)
                            for _ in range(n_channels)
                        ]
                        processed = self._sample_phase1_window_mv(zero_series, "validation")
                    else:
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
            # Multivariate: one norm-stat pair per variate; univariate: single pair.
            if "norm_means" in row:
                d["time_series_norm_stats"] = list(zip(row["norm_means"], row["norm_stds"]))
                d["n_channels"] = len(row["norm_means"])
            else:
                d["time_series_norm_stats"] = [(d["norm_mean"], d["norm_std"])]
                d["n_channels"] = 1
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
            # Already z-normalized in _process_chronos_row using full-window stats.
            if isinstance(row.get("series"), list) and isinstance(row["series"][0], (list, np.ndarray)):
                # Multivariate: one TextTimeSeriesPrompt per variate.
                norm_means = row.get("norm_means", [0.0] * len(row["series"]))
                norm_stds  = row.get("norm_stds",  [1.0] * len(row["series"]))
                row["_norm_mean"] = norm_means[0]
                row["_norm_std"]  = norm_stds[0]
                return [
                    TextTimeSeriesPrompt(
                        f"Channel {i} (mean={m:.4f}, std={s:.4f}): <ts>",
                        np.asarray(ch, dtype=np.float32),
                    )
                    for i, (ch, m, s) in enumerate(zip(row["series"], norm_means, norm_stds))
                ]
            # Univariate
            series = torch.as_tensor(row["series"], dtype=torch.float32)
            row["_norm_mean"] = row.get("norm_mean", 0.0)
            row["_norm_std"]  = row.get("norm_std",  1.0)
            return [TextTimeSeriesPrompt(
                f"This is the time series, it has mean {row['_norm_mean']:.4f} and std {row['_norm_std']:.4f}. <ts>",
                series.numpy(),
            )]

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

        Phase-1 next-patch MSE is unusually sensitive to near-constant history
        windows: a tiny history std can explode both the history tokens and the
        future target.  We therefore use a stronger std floor here than the
        generic dataset path and optionally clip the normalised values.
        """
        history_raw = history_raw.reshape(-1).float()
        future_raw = future_raw.reshape(-1).float()

        norm_mean = history_raw.mean()
        norm_std = history_raw.std().clamp(min=self.norm_std_floor)

        history_norm = (history_raw - norm_mean) / norm_std
        future_norm = (future_raw - norm_mean) / norm_std

        if self.norm_clip is not None:
            history_norm = history_norm.clamp(-self.norm_clip, self.norm_clip)
            future_norm = future_norm.clamp(-self.norm_clip, self.norm_clip)

        return history_norm, future_norm, norm_mean.item(), norm_std.item()

    def _sample_phase1_window(self, series: torch.Tensor, mode: str = "train") -> dict | None:
        T = len(series)
        min_history_len = self.min_history_patches * self.patch_size
        min_total_len = min_history_len + self.patch_size

        if T <= 0:
            series = torch.zeros(min_total_len, dtype=torch.float32)
            T = len(series)

        if T < 2 * self.patch_size:
            window = _last_value_impute(series)
            pad_len = min_total_len - window.numel()
            pad_val = window[-1] if window.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
            if window.numel() == 0:
                window = torch.zeros(1, dtype=torch.float32)
            if pad_len > 0:
                window = torch.cat([window, window.new_full((pad_len,), float(pad_val))])
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
            start = self.rng.randint(0, max_start) if mode == "train" and max_start > 0 else max_start

            history_raw = _last_value_impute(series[start : start + history_len])
            future_raw = _last_value_impute(
                series[start + history_len : start + history_len + self.patch_size]
            )

        history_norm, future_norm, norm_mean, norm_std = self._normalize_history_and_future(
            history_raw,
            future_raw,
        )
        ts_future = future_norm.to(dtype=torch.float32)

        return {
            "type": "chronos",
            "series": history_norm.numpy(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_patches": len(history_norm) // self.patch_size,
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

    def _make_seq2seq_stubs(self, datasets_by_name: Dict[str, Dataset], split_name: str, cap) -> List[dict]:
        stubs = []
        for name, ds in datasets_by_name.items():
            n = min(cap, len(ds)) if cap else len(ds)
            indices = self.rng.sample(range(len(ds)), n)
            for i in indices:
                for pred_len, n_pred in zip(self.pred_lens, self.n_pred_patches_list):
                    stubs.append({
                        "type": "chronos",
                        "_chronos_name": name,
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
            split   = row["_chronos_split"]
            name    = row["_chronos_name"]
            n_pred  = row["_n_pred_patches"]
            raw_row = Phase1JointDataset._chronos_raw_cache[split][name][row["_raw_idx"]]
            # Handle univariate ("target") or multivariate ("targets") — take first channel
            if "target" in raw_row:
                series = torch.tensor(raw_row["target"], dtype=torch.float32)
            else:
                series = torch.tensor(raw_row["targets"][0], dtype=torch.float32)
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
