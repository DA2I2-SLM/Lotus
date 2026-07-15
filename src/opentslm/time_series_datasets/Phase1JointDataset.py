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

import glob
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

from tqdm.auto import tqdm

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker for file-level parallelism in _load_or_build_chronos_raw.
# Must live at module scope (not as a method) so ProcessPoolExecutor can pickle
# it. Processes a single synthetic .arrow file end-to-end (read → cap →
# transpose → save_to_disk) in a dedicated worker process — N workers running
# concurrently on N files lets a CPU-only cache-build job actually scale past
# a single core. Each worker uses num_proc=1 internally to keep the per-file
# work single-threaded (otherwise we'd nest multiprocessing).
# ─────────────────────────────────────────────────────────────────────────────
def _build_one_synthetic_part(arrow_path: str, part_path: str,
                              max_rows_per_file: "int | None") -> Tuple[str, str, int]:
    """Read + transpose + save a single synthetic arrow file. Returns
    ``(part_path, status, n_rows)``. Status is one of:
        "cached"   — part directory already existed; no work done.
        "built"    — newly built and saved.
        "empty"    — 0 rows after read; no part written.
        "failed:…" — raised an exception (message included).
    """
    import pyarrow as pa
    import numpy as np_local
    from datasets import Dataset as _HFDataset
    from datasets import Sequence as _Sequence, Value as _Value

    if os.path.exists(part_path):
        return (part_path, "cached", -1)

    try:
        cap = max_rows_per_file if (max_rows_per_file and max_rows_per_file > 0) else None
        with pa.OSFile(arrow_path, "rb") as src:
            reader = pa.ipc.open_file(src)
            if cap is None:
                table = reader.read_all()
            else:
                batches = []
                total = 0
                for i in range(reader.num_record_batches):
                    if total >= cap:
                        break
                    batch = reader.get_batch(i)
                    need = cap - total
                    if len(batch) > need:
                        batch = batch.slice(0, need)
                    batches.append(batch)
                    total += len(batch)
                table = (
                    pa.Table.from_batches(batches, schema=reader.schema)
                    if batches
                    else reader.read_all().slice(0, 0)
                )
        ds = _HFDataset(table)
        if len(ds) == 0:
            return (part_path, "empty", 0)

        def _transpose(batch):
            out = []
            for tgt in batch["target"]:
                arr = np_local.asarray(tgt, dtype=np_local.float32)  # [C, L]
                if arr.ndim == 1:
                    arr = arr[None, :]
                out.append(arr.T.tolist())                              # [L, C]
            return {"target": out}

        # num_proc=1 here — outer ProcessPoolExecutor already parallelises across
        # files, and HF map's own multiprocessing would fork-on-fork.
        ds = ds.map(_transpose, batched=True, remove_columns=ds.column_names)
        ds = ds.cast_column("target", _Sequence(_Sequence(_Value(dtype="float32"))))

        n_rows = len(ds)
        ds.save_to_disk(part_path)
        del ds
        return (part_path, "built", n_rows)
    except Exception as exc:
        return (part_path, f"failed: {exc!r}", 0)

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

# ---------------------------------------------------------------------------
# Optional offline preprocessing (Options A and B) — OFF by default
# ---------------------------------------------------------------------------
# Set CHRONOS_PRE_IMPUTE=1 to materialise NaN/Inf-imputed chronos splits to
# disk once (Option A). Skip the per-sample impute on the hot path afterwards.
# Set CHRONOS_PRE_WINDOW_STATS=1 to precompute (mean, std) for every training
# stub once and reuse them at __getitem__ (Option B). Both caches live under
# PREPROCESSED_SPLITS_DIR alongside the existing chronos splits.

def _impute_target_numpy(target: np.ndarray) -> np.ndarray:
    """NaN/Inf forward-fill, vectorised. Matches `_last_value_impute`'s torch
    behaviour on a numpy array. Returns a new array if anything changed,
    otherwise the input unchanged.
    """
    bad = ~np.isfinite(target)
    if not bad.any():
        return target

    if target.ndim == 1:
        T = target.shape[0]
        valid = (~bad).astype(np.int64)
        idx = np.maximum.accumulate(valid * (np.arange(T) + 1)) - 1
        leading = idx < 0
        idx_safe = np.maximum(idx, 0)
        out = np.where(bad, target[idx_safe], target).astype(target.dtype, copy=True)
        out[leading] = 0.0
        return out

    # 2D [T, C]: column-wise forward-fill
    T = target.shape[0]
    valid = (~bad).astype(np.int64)
    ar = (np.arange(T) + 1)[:, None]
    idx = np.maximum.accumulate(valid * ar, axis=0) - 1
    leading = idx < 0
    idx_safe = np.maximum(idx, 0)
    out = np.take_along_axis(target, idx_safe, axis=0)
    out = np.where(bad, out, target).astype(target.dtype, copy=True)
    out[leading] = 0.0
    return out


def _impute_target_row(row):
    """HF Dataset .map() callable. Imputes NaN/Inf in `target`."""
    target = np.asarray(row["target"], dtype=np.float32)
    fixed = _impute_target_numpy(target)
    if fixed is target:
        return row
    row["target"] = fixed.tolist()  # HF Dataset stores as Python list / Sequence
    return row

CHRONOS_DATASET_REGISTRY: Dict[str, List[str]] = {
    # Full registry restored (was trimmed to monash_australian_electricity only
    # while isolating the "model can't fit" bug). All 10 sources confirmed
    # present under BASE_DATA_PATH; missing ones are skipped gracefully.
    "dominick": ["target"],
    "monash_fred_md": ["target"],
    "electricity_15min": ["consumption_kW"],
    "ercot": ["target"],
    "exchange_rate": ["target"],
    "ushcn_daily": ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
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
        chronos_window_sampling: str = "stride",
        chronos_random_samples_per_row: int | None = None,
        chronos_univariate: bool = False,
        synthetic_data_dir: str | None = None,
        synthetic_max_rows_per_file: int | None = None,
        norm_std_floor: float = 1e-3,
        norm_clip: float | None = 20.0,
        **kwargs,
    ):
        self.max_samples_per_source = max_samples_per_source
        self.val_fraction = val_fraction
        self.data_seed = data_seed
        self.patch_size = int(patch_size)
        self.window_lens = [int(w) for w in WINDOW_LENS]
        # Per-file cap applied at LOAD time in _load_synthetic_arrow. None ->
        # read the entire file. With ~17 unified_*.arrow files at 100k+ rows
        # each, a small cap (e.g. 500) makes the build finish in seconds for
        # debug iteration.
        self.synthetic_max_rows_per_file = (
            int(synthetic_max_rows_per_file) if synthetic_max_rows_per_file else None
        )
        # Numerical safety for per-channel z-norm in _sample_phase1_window.
        # Without these, synthetic samples with a near-constant channel (raw
        # std ≈ 0; seen in ~15-25% of symtime rows and a handful of kernel/lmc
        # rows) divide by 1e-8 and produce z-scores in the thousands, which
        # poison MSE in joint training (occasional spikes to 80+). Defaults
        # match the FuturePatch override that protects phase1_chronos_mse.
        self.norm_std_floor = max(float(norm_std_floor), 1e-8)
        self.norm_clip = None if norm_clip is None else float(norm_clip)
        # When True, registry Chronos datasets are kept strictly univariate:
        # row-stacking into a multivariate series is skipped, and multi-key
        # datasets are exploded into one univariate [T, 1] series per key.
        self.chronos_univariate = bool(chronos_univariate)
        # When set, every ``unified_*.arrow`` file in this directory is loaded
        # as a multivariate [T, C] source and concatenated into the Phase-1
        # Chronos corpus alongside the registry datasets.
        self.synthetic_data_dir = (
            str(synthetic_data_dir) if synthetic_data_dir else None
        )
        # Stride between enumerated training windows (in timesteps). When set,
        # each HF row yields multiple training stubs — one per window start —
        # so len(dataset) scales with the number of windows rather than the
        # number of rows. Val/test still use one deterministic last-window stub.
        self.chronos_window_stride = (
            int(chronos_window_stride) if chronos_window_stride is not None else None
        )
        # Chronos-style sampling: "stride" enumerates a fixed grid of starts
        # (legacy behaviour). "random" emits N stubs per row WITHOUT a fixed
        # _window_start, and each __getitem__ picks a fresh random anchor in
        # _sample_phase1_window (same downstream slicing/normalisation).
        sampling = str(chronos_window_sampling).lower()
        if sampling not in ("stride", "random"):
            raise ValueError(
                f"chronos_window_sampling must be 'stride' or 'random', "
                f"got {chronos_window_sampling!r}"
            )
        self.chronos_window_sampling = sampling
        self.chronos_random_samples_per_row = (
            int(chronos_random_samples_per_row)
            if chronos_random_samples_per_row is not None and int(chronos_random_samples_per_row) > 0
            else None
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
        test = last test_frac. Short series (where T_train <= 0) are kept whole
        in train only and excluded from val/test.

        Vectorised via ``ds.map(batched=True, num_proc=N)`` so the per-row
        slicing happens across many CPU workers — the previous row-by-row
        Python loop was the dominant cache-build bottleneck for ~650k+
        univariate rows. Output count of val/test may differ from train when
        short series are present (intentional).
        """
        # Use the cache-build worker pool count, capped at 64 to bound overhead.
        _nproc = max(1, min(64, Phase1JointDataset._build_cache_num_proc()))
        _map_kwargs = dict(batched=True, remove_columns=ds.column_names)
        if _nproc > 1 and len(ds) >= 1024:
            _map_kwargs["num_proc"] = _nproc

        def _train_slice(batch):
            out = []
            for tgt in batch["target"]:
                T = len(tgt)
                T_test = max(1, int(T * test_frac))
                T_val = max(1, int(T * val_frac))
                T_train = T - T_val - T_test
                # short series: keep whole in train; otherwise slice [:T_train]
                out.append(tgt if T_train <= 0 else tgt[:T_train])
            return {"target": out}

        def _val_slice(batch):
            out = []
            for tgt in batch["target"]:
                T = len(tgt)
                T_test = max(1, int(T * test_frac))
                T_val = max(1, int(T * val_frac))
                T_train = T - T_val - T_test
                if T_train <= 0:
                    continue
                out.append(tgt[T_train:T_train + T_val])
            return {"target": out}

        def _test_slice(batch):
            out = []
            for tgt in batch["target"]:
                T = len(tgt)
                T_test = max(1, int(T * test_frac))
                T_val = max(1, int(T * val_frac))
                T_train = T - T_val - T_test
                if T_train <= 0:
                    continue
                out.append(tgt[T_train + T_val:])
            return {"target": out}

        train = ds.map(_train_slice, **_map_kwargs)
        val = ds.map(_val_slice, **_map_kwargs)
        test = ds.map(_test_slice, **_map_kwargs)
        return train, val, test

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

    def _normalize_to_target_univariate(
        self, ds: Dataset, target_keys: List[str]
    ) -> Dataset:
        """Emit each variate as an independent univariate ``[T, 1]`` series.

        Unlike :meth:`_normalize_to_target`, this never row-stacks rows that
        share a timeline into one multivariate series, and multi-key datasets
        are *exploded*: every key column becomes its own set of univariate
        rows. The output schema matches the rest of the pipeline —
        ``target`` is ``Sequence(Sequence(float32))`` with channel dim == 1.
        """
        per_key_parts: List[Dataset] = []
        _nproc = self._build_cache_num_proc()
        for key in target_keys:
            sub = ds
            if key != "target":
                sub = sub.rename_column(key, "target")
            sub = sub.select_columns(["target"])
            _map_kwargs: Dict[str, "object"] = dict(
                batched=True,
                remove_columns=sub.column_names,
            )
            if _nproc > 1 and len(sub) >= 1024:
                _map_kwargs["num_proc"] = _nproc
            sub = sub.map(
                lambda batch: {
                    "target": [[[float(v)] for v in seq] for seq in batch["target"]]
                },
                **_map_kwargs,
            )
            sub = sub.cast_column("target", Sequence(Sequence(Value(dtype="float32"))))
            per_key_parts.append(sub)
        if len(per_key_parts) == 1:
            return per_key_parts[0]
        return concatenate_datasets(per_key_parts)

    @staticmethod
    def _build_cache_num_proc() -> int:
        """Worker count for the parallel ``.map`` calls during cache build.

        Read from ``BUILD_CACHE_WORKERS`` env var (set explicitly in the
        build_cache_* sbatch scripts). Falls back to a conservative
        ``min(16, cpu_count)`` so single-process runs still benefit when
        ``cpu_count`` is high. ``1`` disables multiprocessing entirely
        (useful for debugging fork-related issues).
        """
        env = os.environ.get("BUILD_CACHE_WORKERS")
        if env:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        return max(1, min(16, os.cpu_count() or 1))

    def _load_synthetic_arrow(self, arrow_path: str) -> Dataset:
        """Load a unified synthetic MV arrow file as an HF Dataset, ``target`` [T, C].

        ``collect_synthetic.py`` writes ``target`` as ``[n_variates][length]``;
        downstream code (``_sample_phase1_window``) expects ``[T, C]`` (time
        first), so each row is transposed here. Output schema is a single
        ``target`` column of ``Sequence(Sequence(float32))``, matching the
        registry parts so the two can be concatenated.

        When ``self.synthetic_max_rows_per_file`` is set, only the first N
        record-batches that cumulatively hold N rows are read. This makes the
        debug-mode arrow load finish in milliseconds instead of minutes per
        100k-row file.

        The per-row transpose is the largest CPU cost here; ``.map`` is
        called with ``num_proc=BUILD_CACHE_WORKERS`` so a CPU-only cache-build
        job actually uses its allocated cores.
        """
        import pyarrow as pa

        cap = self.synthetic_max_rows_per_file
        with pa.OSFile(arrow_path, "rb") as src:
            reader = pa.ipc.open_file(src)
            if cap is None or cap <= 0:
                table = reader.read_all()
            else:
                # Read just enough batches to hit `cap` rows, then trim.
                batches = []
                total = 0
                for i in range(reader.num_record_batches):
                    if total >= cap:
                        break
                    batch = reader.get_batch(i)
                    need = cap - total
                    if len(batch) > need:
                        batch = batch.slice(0, need)
                    batches.append(batch)
                    total += len(batch)
                table = (
                    pa.Table.from_batches(batches, schema=reader.schema)
                    if batches
                    else reader.read_all().slice(0, 0)
                )
        ds = Dataset(table)
        if len(ds) == 0:
            return ds.select_columns(["target"]) if "target" in ds.column_names else ds

        def _transpose(batch):
            out = []
            for tgt in batch["target"]:
                arr = np.asarray(tgt, dtype=np.float32)  # [C, L]
                if arr.ndim == 1:
                    arr = arr[None, :]
                out.append(arr.T.tolist())  # [L, C]
            return {"target": out}

        _nproc = self._build_cache_num_proc()
        if _nproc > 1 and len(ds) >= 1024:
            ds = ds.map(
                _transpose,
                batched=True,
                num_proc=_nproc,
                remove_columns=ds.column_names,
            )
        else:
            ds = ds.map(_transpose, batched=True, remove_columns=ds.column_names)
        ds = ds.cast_column("target", Sequence(Sequence(Value(dtype="float32"))))
        return ds

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
            if self.chronos_univariate:
                print("  -> chronos_univariate=True: registry datasets kept "
                      "strictly univariate (no row-stacking, multi-key exploded).")
            parts_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "_parts")
            os.makedirs(parts_dir, exist_ok=True)

            chronos_part_paths: List[str] = []
            synth_part_paths: List[str] = []
            # Univariate parts are cached under a distinct name so a previously
            # built multivariate part for the same dataset is never reused.
            part_suffix = "_uv" if self.chronos_univariate else ""
            for name, keys in CHRONOS_DATASET_REGISTRY.items():
                part_path = os.path.join(parts_dir, f"{name}{part_suffix}")
                if not os.path.exists(part_path):
                    try:
                        ds = load_dataset(
                            f"{BASE_DATA_PATH}/{name}",
                            split="train",
                            trust_remote_code=True,
                            keep_in_memory=True,
                        )
                        if self.chronos_univariate:
                            ds = self._normalize_to_target_univariate(ds, keys)
                        else:
                            ds = self._normalize_to_target(ds, keys)
                        print(f"  -> '{name}': {len(ds)} series — saving…")
                        ds.save_to_disk(part_path)
                        del ds
                    except Exception as exc:
                        print(f"  -> WARNING: failed to load '{name}': {exc}")
                        continue
                else:
                    print(f"  -> cached '{name}{part_suffix}' found, skipping.")
                chronos_part_paths.append(part_path)

            # Synthetic multivariate sources: every unified_*.arrow in the dir.
            if self.synthetic_data_dir:
                synth_glob = os.path.join(self.synthetic_data_dir, "unified_*.arrow")
                synth_files = sorted(glob.glob(synth_glob))
                if not synth_files:
                    print(f"  -> WARNING: no unified_*.arrow under "
                          f"{self.synthetic_data_dir}")

                # File-level parallelism: process each synthetic .arrow file in
                # its own worker process. BUILD_CACHE_FILE_WORKERS controls
                # the outer fan-out; each worker uses num_proc=1 internally so
                # we don't nest multiprocessing. Defaults to min(8, #files)
                # which is plenty for our typical 17-file corpus.
                _file_workers_env = os.environ.get("BUILD_CACHE_FILE_WORKERS")
                if _file_workers_env:
                    try:
                        n_file_workers = max(1, int(_file_workers_env))
                    except ValueError:
                        n_file_workers = 1
                else:
                    n_file_workers = min(8, len(synth_files))

                synth_args = []
                for arrow_path in synth_files:
                    stem = os.path.splitext(os.path.basename(arrow_path))[0]
                    part_path = os.path.join(parts_dir, f"synthetic_{stem}")
                    synth_args.append((arrow_path, part_path, self.synthetic_max_rows_per_file))

                if n_file_workers > 1 and len(synth_args) > 1:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    print(f"  -> processing {len(synth_args)} synthetic files in "
                          f"parallel (BUILD_CACHE_FILE_WORKERS={n_file_workers})…")
                    with ProcessPoolExecutor(max_workers=n_file_workers) as pool:
                        futures = {
                            pool.submit(_build_one_synthetic_part, *args): args
                            for args in synth_args
                        }
                        for fut in as_completed(futures):
                            part_path, status, n_rows = fut.result()
                            stem = os.path.basename(part_path)
                            if status == "cached":
                                print(f"  -> cached synthetic '{stem}' found, skipping.")
                                synth_part_paths.append(part_path)
                            elif status == "built":
                                print(f"  -> synthetic '{stem}': {n_rows} MV series — built.")
                                synth_part_paths.append(part_path)
                            elif status == "empty":
                                print(f"  -> synthetic '{stem}': 0 rows — skipping.")
                            else:
                                print(f"  -> WARNING: synthetic '{stem}': {status}")
                else:
                    # Sequential fallback (file count ≤ 1 or workers=1).
                    for args in synth_args:
                        arrow_path, part_path, _max_rows = args
                        stem = os.path.basename(part_path)
                        part_path_, status, n_rows = _build_one_synthetic_part(*args)
                        if status == "cached":
                            print(f"  -> cached synthetic '{stem}' found, skipping.")
                            synth_part_paths.append(part_path_)
                        elif status == "built":
                            print(f"  -> synthetic '{stem}': {n_rows} MV series — built.")
                            synth_part_paths.append(part_path_)
                        elif status == "empty":
                            print(f"  -> synthetic '{stem}': 0 rows — skipping.")
                        else:
                            print(f"  -> WARNING: synthetic '{stem}': {status}")

            if not chronos_part_paths and not synth_part_paths:
                raise RuntimeError("No Chronos datasets could be loaded!")

            # Two different split strategies, then concatenate:
            #   * chronos_uv parts  -> temporal per-row slice (rows are variates;
            #     a row-level split would put rare sources entirely in train).
            #   * synthetic parts   -> row-level random split (each row is an
            #     independent sample; we want fully unseen series in val/test).
            split_pieces = {"train": [], "val": [], "test": []}

            if chronos_part_paths:
                ds_chr = concatenate_datasets(
                    [load_from_disk(p, keep_in_memory=False) for p in chronos_part_paths]
                )
                print(
                    f"[Phase1Joint] chronos: {len(ds_chr)} row(s) — temporal slice "
                    f"(train={int((1-VAL_FRAC-TEST_FRAC)*100)}%, "
                    f"val={int(VAL_FRAC*100)}%, test={int(TEST_FRAC*100)}%)."
                )
                tr_c, va_c, te_c = self._temporal_split_rows(
                    ds_chr, val_frac=VAL_FRAC, test_frac=TEST_FRAC
                )
                # Tag source so the validation loop can break out per-source MSE
                # for wandb logging (chronos_uv vs synthetic).
                tr_c = tr_c.add_column("source", ["chronos_uv"] * len(tr_c))
                va_c = va_c.add_column("source", ["chronos_uv"] * len(va_c))
                te_c = te_c.add_column("source", ["chronos_uv"] * len(te_c))
                split_pieces["train"].append(tr_c)
                split_pieces["val"].append(va_c)
                split_pieces["test"].append(te_c)

            if synth_part_paths:
                ds_syn = concatenate_datasets(
                    [load_from_disk(p, keep_in_memory=False) for p in synth_part_paths]
                )
                print(
                    f"[Phase1Joint] synthetic: {len(ds_syn)} row(s) — row-level "
                    f"random split with seed=42 "
                    f"(train≈{int((1-VAL_FRAC-TEST_FRAC)*100)}%, "
                    f"val≈{int(VAL_FRAC*100)}%, test≈{int(TEST_FRAC*100)}%)."
                )
                tv_s, te_s = ds_syn.train_test_split(
                    test_size=TEST_FRAC, seed=42
                ).values()
                tr_s, va_s = tv_s.train_test_split(
                    test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
                ).values()
                tr_s = tr_s.add_column("source", ["synthetic"] * len(tr_s))
                va_s = va_s.add_column("source", ["synthetic"] * len(va_s))
                te_s = te_s.add_column("source", ["synthetic"] * len(te_s))
                split_pieces["train"].append(tr_s)
                split_pieces["val"].append(va_s)
                split_pieces["test"].append(te_s)

            train = (
                concatenate_datasets(split_pieces["train"])
                if len(split_pieces["train"]) > 1 else split_pieces["train"][0]
            )
            val = (
                concatenate_datasets(split_pieces["val"])
                if len(split_pieces["val"]) > 1 else split_pieces["val"][0]
            )
            test = (
                concatenate_datasets(split_pieces["test"])
                if len(split_pieces["test"]) > 1 else split_pieces["test"][0]
            )

            os.makedirs(split_dir, exist_ok=True)
            # Use num_proc to fully exploit cache-build's allocated CPUs.
            # The combined train/val/test write is ~70 GB on /vast; with the
            # single-proc default it takes hours on these big builds.
            _save_nproc = max(1, min(64, Phase1JointDataset._build_cache_num_proc()))
            train.save_to_disk(os.path.join(split_dir, "train"), num_proc=_save_nproc)
            val.save_to_disk(os.path.join(split_dir, "val"),   num_proc=_save_nproc)
            test.save_to_disk(os.path.join(split_dir, "test"),  num_proc=_save_nproc)
            with open(done_flag, "w") as fh:
                fh.write("done\n")
            del split_pieces, train, val, test
            print(f"[Phase1Joint] Chronos splits saved to {split_dir}")

        if dist.is_initialized():
            dist.barrier()

        # ── Per-source split cache backfill ─────────────────────────────────
        # Phase1PretrainEvalDataset --source_filter loads `{split}_{source}`
        # directly when these exist; otherwise it falls back to ds.filter()
        # which is ~80+ minutes for 600k+ rows because it's a row-level
        # Python callback. We build per-source caches here once: rank-0 only,
        # vectorised via numpy mask + ds.select() (~seconds for 600k rows).
        # Skipped if already done; runs on both freshly-built and pre-existing
        # combined caches.
        per_source_done = os.path.join(split_dir, "_DONE_PER_SOURCE")
        if rank == 0 and not os.path.exists(per_source_done):
            print("[Phase1Joint] Building per-source split caches "
                  "(eval-only optimisation)…")
            try:
                for _split_name in ("train", "val", "test"):
                    _combined_path = os.path.join(split_dir, _split_name)
                    if not os.path.isdir(_combined_path):
                        continue
                    _ds_combined = load_from_disk(_combined_path, keep_in_memory=False)
                    if "source" not in _ds_combined.column_names:
                        print(f"[Phase1Joint]   {_split_name}: no 'source' column, "
                              "skipping (older cache built before source tagging).")
                        del _ds_combined
                        continue
                    _sources = np.asarray(_ds_combined["source"], dtype=object)
                    for _src in ("chronos_uv", "synthetic"):
                        _out_path = os.path.join(split_dir, f"{_split_name}_{_src}")
                        if os.path.isdir(_out_path):
                            print(f"[Phase1Joint]   {_out_path} already exists, skipping.")
                            continue
                        _idx = np.nonzero(_sources == _src)[0]
                        if _idx.size == 0:
                            print(f"[Phase1Joint]   {_split_name}/{_src}: 0 rows, skipping.")
                            continue
                        _sub = _ds_combined.select(_idx.tolist(), keep_in_memory=True)
                        # Match the parallelism used for the main split writes.
                        _backfill_nproc = max(1, min(32, Phase1JointDataset._build_cache_num_proc()))
                        _sub.save_to_disk(_out_path, num_proc=_backfill_nproc)
                        print(f"[Phase1Joint]   saved {_out_path} "
                              f"({len(_sub)}/{len(_ds_combined)} rows)")
                        del _sub
                    del _ds_combined
                with open(per_source_done, "w") as fh:
                    fh.write("done\n")
                print(f"[Phase1Joint] Per-source split caches done -> {per_source_done}")
            except Exception as exc:
                print(f"[Phase1Joint] WARN: per-source split build failed: {exc!r}; "
                      "eval will fall back to ds.filter() at run time.")
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

    # ------------------------------------------------------------------
    # Option A: one-time NaN/Inf imputation cached on disk
    # ------------------------------------------------------------------
    def _load_or_build_chronos_imputed_raw(self) -> Tuple[Dataset, Dataset, Dataset]:
        """If CHRONOS_PRE_IMPUTE=1, build (or reload) NaN/Inf-imputed chronos
        splits and return those. Otherwise return the raw splits unchanged.
        The hot-path `_last_value_impute` calls become near-free fast-path
        early-returns once the data is clean.
        """
        if not bool(int(os.environ.get("CHRONOS_PRE_IMPUTE", "0"))):
            return self._load_or_build_chronos_raw()

        rank = int(os.environ.get("RANK", 0))
        impute_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all_imputed")
        done_flag = os.path.join(impute_dir, "_DONE")

        if rank == 0 and not os.path.exists(done_flag):
            print("[Phase1Joint] Building NaN/Inf-imputed chronos splits (one-time)…")
            train_raw, val_raw, test_raw = self._load_or_build_chronos_raw()
            os.makedirs(impute_dir, exist_ok=True)
            n_proc = max(1, (os.cpu_count() or 1))
            for name, ds in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
                target_dir = os.path.join(impute_dir, name)
                if os.path.exists(target_dir):
                    continue
                imputed = ds.map(
                    _impute_target_row,
                    num_proc=n_proc,
                    desc=f"impute chronos.{name}",
                    load_from_cache_file=False,
                )
                imputed.save_to_disk(target_dir)
                del imputed
            with open(done_flag, "w") as fh:
                fh.write("done\n")
            print(f"[Phase1Joint] Imputed chronos splits saved to {impute_dir}")
            del train_raw, val_raw, test_raw

        if dist.is_initialized():
            dist.barrier()

        train = load_from_disk(os.path.join(impute_dir, "train"), keep_in_memory=True)
        val = load_from_disk(os.path.join(impute_dir, "val"), keep_in_memory=True)
        test = load_from_disk(os.path.join(impute_dir, "test"), keep_in_memory=True)
        return train, val, test

    # ------------------------------------------------------------------
    # Option B: per-stub window stats cached on disk
    # ------------------------------------------------------------------
    def _compute_or_load_window_stats(
        self,
        ds: Dataset,
        stubs: List[dict],
        split_name: str,
        history_len: int,
    ) -> "np.ndarray | None":
        """Return a (n_stubs, 2) float32 array of per-stub (mean, std), or None
        if Option B is disabled. Cached on disk under
        PREPROCESSED_SPLITS_DIR/window_stats_h{H}_p{P}_s{S}/{split}.npy.

        The stats correspond to the history window of length ``history_len``
        starting at ``stub['_window_start']``. For val/test stubs (no fixed
        start), we fall back to the deterministic last-window logic that
        ``_sample_phase1_window`` uses.
        """
        if not bool(int(os.environ.get("CHRONOS_PRE_WINDOW_STATS", "0"))):
            return None

        stride = self.chronos_window_stride or 0
        cache_dir = os.path.join(
            PREPROCESSED_SPLITS_DIR,
            f"window_stats_h{history_len}_p{self.patch_size}_s{stride}",
        )
        cache_path = os.path.join(cache_dir, f"{split_name}.npy")
        rank = int(os.environ.get("RANK", 0))

        if rank == 0 and not os.path.exists(cache_path):
            os.makedirs(cache_dir, exist_ok=True)
            print(
                f"[Phase1Joint] Building window stats for chronos.{split_name} "
                f"({len(stubs)} stubs, history_len={history_len})…"
            )
            out = np.empty((len(stubs), 2), dtype=np.float32)
            target_col = ds.data.column("target")
            chunks = list(target_col.chunks)
            chunk_offsets = np.cumsum([0] + [len(c) for c in chunks])

            def _row_target(raw_idx: int) -> np.ndarray:
                # Pull a single row from the underlying Arrow ListArray.
                ci = int(np.searchsorted(chunk_offsets[1:], raw_idx, side="right"))
                local = raw_idx - chunk_offsets[ci]
                vals = chunks[ci].value(local)
                return np.asarray(vals, dtype=np.float32)

            iterator = tqdm(
                stubs,
                desc=f"stats {split_name}",
                disable=(rank != 0),
                mininterval=2.0,
            )
            for i, stub in enumerate(iterator):
                target = _row_target(int(stub["_raw_idx"]))
                T = target.shape[0]
                if T < history_len + self.patch_size:
                    # Short series — use whatever we can; matches the fallback
                    # behaviour in _sample_phase1_window.
                    window = target[: max(1, T)]
                else:
                    ws = stub.get("_window_start")
                    if ws is None:
                        ws = max(0, T - history_len - self.patch_size)
                    ws = max(0, min(int(ws), T - history_len))
                    window = target[ws : ws + history_len]
                if window.size == 0:
                    out[i] = (0.0, 1.0)
                    continue
                mean = float(window.mean())
                std = float(max(window.std(), 1e-8))
                out[i] = (mean, std)

            np.save(cache_path, out)
            print(f"[Phase1Joint] Window stats saved to {cache_path}")

        if dist.is_initialized():
            dist.barrier()

        return np.load(cache_path, mmap_mode="r")

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
        pre_stats: tuple | None = None,
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

        ``pre_stats``: optional ``(mean, std)`` precomputed for this stub
        (Option B). When provided, skips the window's mean/std computation.
        """
        # OPTION B: always pad to `target_window_len = max(valid_window_lens)`.
        # Track `real_len` (real timesteps in the window) so downstream
        # encoder/loss masks can exclude padded patches. This way short series
        # contribute to training without poisoning supervision.
        T = series.shape[0]  # works for both [T] and [T, C]
        is_mv = series.dim() == 2
        target_window_len = max(self.valid_window_lens)
        patch_size = self.patch_size

        # Decide the real-data slice + its in-window start.
        # We always reserve `patch_size` real timesteps at the end of the
        # real slice for ts_future, so the splice at position n_real-1 has
        # a meaningful target.
        if T >= target_window_len + patch_size:
            # Plenty of data: random/last window of full target length.
            max_start = T - target_window_len - patch_size
            if window_start is not None:
                start = max(0, min(int(window_start), max_start))
            else:
                start = (
                    self.rng.randint(0, max_start)
                    if mode == "train" and max_start > 0 else max_start
                )
            real_len_in_window = target_window_len
            future_start_in_series = start + target_window_len
            has_future = True
        elif T > patch_size:
            # Some data but not enough for full target window. Use as much
            # real data as possible (leaving 1 patch for future), pad the rest.
            start = 0
            real_len_in_window = min(target_window_len, T - patch_size)
            future_start_in_series = real_len_in_window
            has_future = True
        else:
            # Very short or empty series: pad with last-value (or zero) and
            # accept that ts_future does not exist.
            start = 0
            real_len_in_window = max(0, T)
            future_start_in_series = real_len_in_window
            has_future = False

        # Build the (possibly padded) window of length target_window_len.
        if real_len_in_window > 0:
            real_slice = _last_value_impute(
                series[start: start + real_len_in_window]
            )
        else:
            real_slice = (
                torch.zeros(0, series.shape[1], dtype=torch.float32) if is_mv
                else torch.zeros(0, dtype=torch.float32)
            )

        pad_needed = target_window_len - real_slice.shape[0]
        if pad_needed > 0:
            if real_slice.shape[0] == 0:
                # No real data at all: pad with zeros.
                if is_mv:
                    C = series.shape[1] if is_mv else 1
                    pad_block = torch.zeros(pad_needed, C, dtype=torch.float32)
                else:
                    pad_block = torch.zeros(pad_needed, dtype=torch.float32)
            else:
                # Repeat the last real value.
                if is_mv:
                    pad_block = real_slice[-1:].expand(pad_needed, -1)
                else:
                    pad_scalar = float(real_slice[-1].item())
                    pad_block = real_slice.new_full((pad_needed,), pad_scalar)
            window = torch.cat([real_slice, pad_block], dim=0)
        else:
            window = real_slice[:target_window_len]
        window_len = target_window_len

        # Z-normalize using the REAL portion's stats only (avoid bias from
        # the constant padding tail) but apply to the full window so padded
        # tokens are also numerically tame (they get masked out anyway).
        _std_floor = max(float(self.norm_std_floor), 1e-8)
        real_portion = window[: max(1, real_len_in_window)]
        if pre_stats is not None and window.dim() == 1:
            norm_mean, norm_std = float(pre_stats[0]), float(pre_stats[1])
            norm_std = max(norm_std, _std_floor)
            window_norm = (window - norm_mean) / norm_std
        else:
            if window.dim() == 1:
                m_t = real_portion.mean()
                s_t = real_portion.std().clamp(min=_std_floor)
                window_norm = (window - m_t) / s_t
                norm_mean = m_t.item()
                norm_std = s_t.item()
            else:
                m_t = real_portion.mean(dim=0, keepdim=True)
                s_t = real_portion.std(dim=0, keepdim=True).clamp(min=_std_floor)
                window_norm = (window - m_t) / s_t
                norm_mean = m_t[0, 0].item()
                norm_std = s_t[0, 0].item()

        if self.norm_clip is not None:
            window_norm = window_norm.clamp(-self.norm_clip, self.norm_clip)

        # ts_future: the patch immediately after the real portion.
        ts_future = None
        if has_future and future_start_in_series + patch_size <= T:
            future_raw = _last_value_impute(
                series[future_start_in_series: future_start_in_series + patch_size]
            )
            if series.dim() == 1:
                ts_future = (
                    (future_raw - norm_mean) / max(norm_std, _std_floor)
                ).to(dtype=torch.float32).clone()
                if self.norm_clip is not None:
                    ts_future = ts_future.clamp(-self.norm_clip, self.norm_clip)
            else:
                # Per-channel future norm using real-portion stats (matches
                # the window normalization above).
                ch_mean = real_portion.mean(dim=0, keepdim=True)                       # [1, C]
                ch_std  = real_portion.std(dim=0, keepdim=True).clamp(min=_std_floor)  # [1, C]
                future_norm = (future_raw - ch_mean) / ch_std                          # [P, C]
                if self.norm_clip is not None:
                    future_norm = future_norm.clamp(-self.norm_clip, self.norm_clip)
                ts_future = future_norm.transpose(0, 1).to(dtype=torch.float32).contiguous()  # [C, P]

        return {
            "type": "chronos",
            "series": window_norm.numpy(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_patches": window_len // self.patch_size,
            # NEW: real timesteps in the window — downstream encoder/loss masks
            # exclude padded positions. Carried into _format_sample as
            # `time_series_lengths` so OpenTSLMSPFixPos2D / processor honor it.
            "real_len": int(real_len_in_window),
            "ts_future": ts_future,   # UV: [patch_size]; MV: [C, patch_size]; or None
        }

    def _process_chronos_row(
        self,
        row: dict,
        mode: str = "train",
        window_start: int | None = None,
        pre_stats: tuple | None = None,
    ) -> dict | None:
        # "target" stores [T] for univariate or [T, C] (nested list) for multivariate.
        series = torch.as_tensor(row["target"], dtype=torch.float32)
        return self._sample_phase1_window(
            series, mode, window_start=window_start, pre_stats=pre_stats,
        )

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

        Each stub is ``{"type": "chronos", "_raw_idx": i, "_chronos_split": split_name,
        "source": <source>}``. The actual window crop + normalisation happens
        on-demand in ``_format_sample``. ``source`` is propagated onto every
        stub so the task-weighted sampler can split chronos into chronos_uv vs
        synthetic *before* __getitem__ ever runs (the sampler operates on the
        raw stubs, not on formatted samples).

        When ``self.chronos_window_stride`` is set *and* this is the train split,
        each row is additionally expanded into one stub per window start so that
        ``len(dataset)`` scales with the total number of windows. Val/test keep
        the original one-stub-per-row semantics; their deterministic last-window
        logic lives in ``_sample_phase1_window``.

        Caching: the result is deterministic given (split, sampling config, cap,
        data signature). We cache to ``PREPROCESSED_SPLITS_DIR/_stub_cache/<key>.pkl``
        so reruns skip the multi-minute build entirely.
        Vectorisation: the random-anchor branch uses ``np.repeat`` + a single list
        comprehension instead of a per-row Python loop, ~100× faster than the
        previous ``for i in iter_indices: ... for _ in range(N): stubs.append(...)``.
        """
        import pickle  # local import keeps top-of-file lean

        stride = self.chronos_window_stride
        sampling = self.chronos_window_sampling if split_name == "train" else "stride"
        history_len = max(self.valid_window_lens)
        min_total = history_len + self.patch_size
        stride = max(1, int(stride)) if stride is not None else None
        n_per_row = self.chronos_random_samples_per_row or 1
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        # Per-row length lookup — also used as a quick data fingerprint in
        # the cache key. Reads outer ListArray offsets in one pass; doesn't
        # materialise any float rows.
        lengths_per_row: "np.ndarray | None"
        try:
            target_col = ds.data.column("target")
            lengths_per_row = np.empty(len(ds), dtype=np.int64)
            cursor = 0
            for chunk in target_col.chunks:
                offs = chunk.offsets.to_numpy(zero_copy_only=False)
                n_chunk = len(chunk)
                lengths_per_row[cursor:cursor + n_chunk] = offs[1:] - offs[:-1]
                cursor += n_chunk
            assert cursor == len(ds), (
                f"chunk-offset coverage mismatch: cursor={cursor} len(ds)={len(ds)}"
            )
        except Exception as exc:
            if rank == 0:
                print(
                    f"[{self.__class__.__name__}] Arrow-offset length lookup "
                    f"unavailable ({exc!r}); falling back to per-row reads."
                )
            lengths_per_row = None

        # ── Cache key ──────────────────────────────────────────────────────
        # Deterministic given the split, the active sampling mode + config,
        # and the data signature (len + sum of lengths catches corpus drift).
        data_sig = int(lengths_per_row.sum()) if lengths_per_row is not None else len(ds)
        cap_tag = int(cap) if cap else 0
        stride_tag = int(stride) if stride is not None else 0
        cache_key = (
            f"{split_name}_smp{sampling}_n{n_per_row}_str{stride_tag}_cap{cap_tag}"
            f"_pat{self.patch_size}_hist{history_len}_seed{self.data_seed}"
            f"_len{len(ds)}_sig{data_sig}"
        )
        cache_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "_stub_cache")
        cache_path = os.path.join(cache_dir, cache_key + ".pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    stubs = pickle.load(f)
                if rank == 0:
                    print(
                        f"[{self.__class__.__name__}] loaded cached stubs ({len(stubs)}) "
                        f"from {cache_path}"
                    )
                return stubs
            except Exception as exc:
                if rank == 0:
                    print(
                        f"[{self.__class__.__name__}] cache load failed ({exc!r}); "
                        f"rebuilding."
                    )

        # ── Build ──────────────────────────────────────────────────────────
        n = cap if cap else len(ds)
        n = min(n, len(ds))
        indices = self.rng.sample(range(len(ds)), n)

        if "source" in ds.column_names:
            sources_per_row = ds["source"]
        else:
            sources_per_row = None

        if (stride is None or split_name != "train") and sampling != "random":
            # Trivial single-stub-per-row case (val/test or default).
            if sources_per_row is not None:
                stubs = [
                    {"type": "chronos", "_raw_idx": int(i), "_chronos_split": split_name,
                     "source": sources_per_row[i]}
                    for i in indices
                ]
            else:
                stubs = [
                    {"type": "chronos", "_raw_idx": int(i), "_chronos_split": split_name,
                     "source": None}
                    for i in indices
                ]

        elif sampling == "random":
            # ── Vectorised random-anchor expansion (was the 2.5h Python loop) ──
            indices_arr = np.asarray(indices, dtype=np.int64)
            if lengths_per_row is not None:
                T_arr = lengths_per_row[indices_arr]
            else:
                # Slow fallback — only used when Arrow offsets aren't available.
                T_arr = np.array(
                    [
                        (len(ds[int(i)]["target"]) if True else 0)
                        for i in indices_arr
                    ],
                    dtype=np.int64,
                )
            is_long = T_arr >= min_total
            n_per_index = np.where(is_long, n_per_row, 1).astype(np.int64)
            raw_idx_per_stub = np.repeat(indices_arr, n_per_index)

            if sources_per_row is not None:
                src_arr = np.asarray(sources_per_row, dtype=object)
                source_per_stub = src_arr[raw_idx_per_stub].tolist()
            else:
                source_per_stub = [None] * len(raw_idx_per_stub)

            raw_idx_list = raw_idx_per_stub.tolist()
            stubs = [
                {"type": "chronos", "_raw_idx": int(i), "_chronos_split": split_name, "source": s}
                for i, s in zip(raw_idx_list, source_per_stub)
            ]

            if rank == 0:
                n_long = int(is_long.sum())
                n_short = int((~is_long).sum())
                print(
                    f"[{self.__class__.__name__}] sampling=random, "
                    f"samples_per_row={n_per_row}: "
                    f"{len(indices)} rows ({n_long} long / {n_short} short) "
                    f"-> {len(stubs)} stubs "
                    f"(history_len={history_len}, patch_size={self.patch_size})"
                )

        else:
            # ── Stride mode (legacy) — kept as the original per-row loop ──
            stubs = []
            iter_indices = tqdm(
                indices,
                desc=f"[{self.__class__.__name__}] expanding train stubs",
                disable=(rank != 0),
                mininterval=2.0,
            )
            for i in iter_indices:
                if lengths_per_row is not None:
                    T = int(lengths_per_row[i])
                else:
                    try:
                        T = len(ds[i]["target"])
                    except Exception:
                        T = 0
                src = sources_per_row[i] if sources_per_row is not None else None
                if T < min_total:
                    stubs.append({
                        "type": "chronos",
                        "_raw_idx": int(i),
                        "_chronos_split": split_name,
                        "source": src,
                    })
                    continue
                max_start = T - min_total
                starts = list(range(0, max_start + 1, stride))
                if not starts or starts[-1] != max_start:
                    starts.append(max_start)
                for s in starts:
                    stubs.append({
                        "type": "chronos",
                        "_raw_idx": int(i),
                        "_chronos_split": split_name,
                        "_window_start": int(s),
                        "source": src,
                    })
            if rank == 0:
                print(
                    f"[{self.__class__.__name__}] train stride={stride}: "
                    f"{len(indices)} rows -> {len(stubs)} window stubs "
                    f"(history_len={history_len}, patch_size={self.patch_size})"
                )

        # ── Write cache (atomic rename — safe across concurrent ranks) ─────
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp_path = cache_path + f".tmp.{rank}.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump(stubs, f, protocol=5)
            os.replace(tmp_path, cache_path)
            if rank == 0:
                print(
                    f"[{self.__class__.__name__}] cached stubs ({len(stubs)}) "
                    f"to {cache_path}"
                )
        except Exception as exc:
            if rank == 0:
                print(
                    f"[{self.__class__.__name__}] WARNING: stub cache write failed: {exc!r}"
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
            tr, va, te = self._load_or_build_chronos_imputed_raw()
            # Make __getitem__ return the "target" column as a numpy array
            # rather than a Python list. torch.tensor on a Python list of 1024
            # floats walks each element through Python (~100 µs); on a numpy
            # buffer it's a single C-level copy (~1 µs).
            for ds in (tr, va, te):
                if hasattr(ds, "set_format"):
                    try:
                        ds.set_format(
                            type="numpy", columns=["target"], output_all_columns=True
                        )
                    except Exception as exc:
                        print(
                            f"[{self.__class__.__name__}] could not set numpy "
                            f"format on chronos cache ({exc!r}); falling back "
                            f"to default Python-list output."
                        )
            Phase1JointDataset._chronos_raw_cache = {"train": tr, "validation": va, "test": te}
        cache = Phase1JointDataset._chronos_raw_cache

        # ── TSQA (small, eager) ──
        tsqa_t, tsqa_v, tsqa_te = self._load_tsqa_lists(cap)

        # ── Chronos stubs (lazy) ──
        c_t  = self._make_chronos_stubs(cache["train"],      "train",      cap)
        c_v  = self._make_chronos_stubs(cache["validation"], "validation", cap)
        c_te = self._make_chronos_stubs(cache["test"],       "test",       cap)

        # Option B: optionally precompute per-stub (mean, std) and inject into
        # each stub. _sample_phase1_window will use these when present and
        # skip the per-call mean/std torch ops.
        if bool(int(os.environ.get("CHRONOS_PRE_WINDOW_STATS", "0"))):
            history_len = max(self.valid_window_lens)
            for ds_split, stubs_split, name in (
                (cache["train"],      c_t,  "train"),
                (cache["validation"], c_v,  "validation"),
                (cache["test"],       c_te, "test"),
            ):
                stats = self._compute_or_load_window_stats(
                    ds_split, stubs_split, name, history_len
                )
                if stats is None:
                    continue
                for i, stub in enumerate(stubs_split):
                    stub["_pre_norm_mean"] = float(stats[i, 0])
                    stub["_pre_norm_std"]  = float(stats[i, 1])

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
        chronos_source = None
        if row.get("type") == "chronos" and "_raw_idx" in row:
            # Lazy stub: fetch the raw series and process it now
            split = row["_chronos_split"]
            raw_row = Phase1JointDataset._chronos_raw_cache[split][row["_raw_idx"]]
            # Carry source ("chronos_uv" | "synthetic" | None for old splits)
            # forward so the val loop can break out per-source MSE for wandb.
            chronos_source = raw_row.get("source") if hasattr(raw_row, "get") else None
            if chronos_source is None and isinstance(raw_row, dict):
                chronos_source = raw_row.get("source")
            window_start = row.get("_window_start")
            # Option B: if we precomputed (mean, std) for this stub, forward
            # them so _sample_phase1_window can skip the per-call .mean()/.std().
            pre_stats = None
            if "_pre_norm_mean" in row and "_pre_norm_std" in row:
                pre_stats = (row["_pre_norm_mean"], row["_pre_norm_std"])
            processed = self._process_chronos_row(
                raw_row, split, window_start=window_start, pre_stats=pre_stats,
            )
            if processed is None:
                # Defensive fallback: synthesize the smallest valid window rather
                # than returning an empty sample that would later break collation.
                series = torch.tensor(raw_row["target"], dtype=torch.float32)
                processed = self._sample_phase1_window(series, "validation")
                if processed is None:
                    series = torch.zeros(2 * self.patch_size, dtype=torch.float32)
                    processed = self._sample_phase1_window(series, "validation")
            if chronos_source is not None:
                processed["source"] = chronos_source
            row = processed

        # super() calls _get_text_time_series_prompt_list, which sets
        # row["_norm_mean"] and row["_norm_std"] as a side-effect.
        d = super()._format_sample(row)
        d["type"]      = row.get("type", "unknown")
        if row.get("source") is not None:
            d["source"] = row["source"]
        elif chronos_source is not None:
            d["source"] = chronos_source
        d["norm_mean"] = row.get("_norm_mean", row.get("norm_mean", 0.0))
        d["norm_std"]  = row.get("_norm_std",  row.get("norm_std",  1.0))
        if row.get("ts_future") is not None:
            d["ts_future"] = row["ts_future"]
        if d.get("time_series"):
            d["ts_already_normalized"] = True
            # For multivariate [L, C] series the norm stats track channel-0 only;
            # pad_and_apply_batch will compute per-channel stats for channels 1..C-1.
            d["time_series_norm_stats"] = [(d["norm_mean"], d["norm_std"])]
            # OPTION B: tell the encoder/loss the real (unpadded) length so that
            # padded patches at the tail of the window are masked out.
            # _sample_phase1_window sets row["real_len"]; if missing, the model
            # falls back to full length (legacy behavior).
            if row.get("real_len") is not None:
                d["time_series_lengths"] = [int(row["real_len"])]
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
            n_timesteps = series.shape[0]
            modality = "multivariate" if n_channels > 1 else "univariate"
            ch_word = "channels" if n_channels > 1 else "channel"
            prompt_text = (
                f"This is a {modality} time series with {n_channels} {ch_word} "
                f"and {n_timesteps} time steps. "
                f"It has mean {row['_norm_mean']:.4f} and std {row['_norm_std']:.4f}. <ts>"
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

        n_channels = series.shape[1] if series.dim() == 2 else 1
        n_timesteps = series.shape[0]
        modality = "multivariate" if n_channels > 1 else "univariate"
        ch_word = "channels" if n_channels > 1 else "channel"
        return [TextTimeSeriesPrompt(
            f"This is a {modality} time series with {n_channels} {ch_word} "
            f"and {n_timesteps} time steps. "
            f"It has mean {mean_val:.4f} and std {std_val:.4f}.",
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

            n_channels = series_tensor.shape[1] if series_tensor.dim() == 2 else 1
            n_timesteps = series_tensor.shape[0]
            modality = "multivariate" if n_channels > 1 else "univariate"
            ch_word = "channels" if n_channels > 1 else "channel"
            text_prompt = (
                f"This is a {modality} time series with {n_channels} {ch_word} "
                f"and {n_timesteps} time steps. "
                f"It has mean {mean:.4f} and std {std:.4f}:"
            )
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
            tr, va, te = self._load_or_build_chronos_imputed_raw()
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
        pre_stats: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Numerically stable z-normalisation for next-patch regression.

        Handles both univariate ``[T]`` and multivariate ``[T, C]`` history
        windows. For MV, stats are per-channel (matching the parent class and
        the eval dataset), so the channel structure is preserved.

        ``pre_stats``: optional ``(mean, std)`` precomputed for this stub
        (Option B). Univariate only; falls back to in-call stats on MV.
        """
        history_raw = history_raw.float()
        future_raw = future_raw.float()

        if history_raw.dim() == 1:
            if pre_stats is not None:
                norm_mean = float(pre_stats[0])
                norm_std = max(float(pre_stats[1]), self.norm_std_floor)
                history_norm = (history_raw - norm_mean) / norm_std
                future_norm = (future_raw - norm_mean) / norm_std
            else:
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
        pre_stats: tuple | None = None,
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
            elif mode == "train" and max_start > 0:
                # Chronos-style: uniform anchor in [0, max_start]. Use torch's
                # RNG so each DataLoader worker draws an independent stream
                # (torch.utils.data.DataLoader auto-seeds torch's RNG per
                # worker; self.rng would share state across forked workers).
                # `torch.randint(low, high)` excludes `high`, so add 1 to make
                # the range [0, max_start] inclusive, matching the prior
                # `self.rng.randint(0, max_start)` semantics.
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start = max_start

            history_raw = _last_value_impute(series[start : start + history_len])
            future_raw = _last_value_impute(
                series[start + history_len : start + history_len + self.patch_size]
            )

        history_norm, future_norm, norm_mean, norm_std = self._normalize_history_and_future(
            history_raw,
            future_raw,
            pre_stats=pre_stats,
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
            tr, va, te = self._load_or_build_chronos_imputed_raw()
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

        n_channels = input_series.shape[1] if input_series.dim() == 2 else 1
        modality = "multivariate" if n_channels > 1 else "univariate"
        ch_word = "channels" if n_channels > 1 else "channel"

        return [
            TextTimeSeriesPrompt(
                f"You are given a {modality} time series with {n_channels} {ch_word} "
                f"and {n_in * self.patch_size} time steps "
                f"(mean={mean:.4f}, std={std:.4f}):",
                input_series.numpy(),
            ),
            TextTimeSeriesPrompt(
                f"Predict the next {n_out * self.patch_size} time steps "
                f"with {n_channels} {ch_word}:",
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
