# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Phase1JointDataset.py
---------------------
Joint training dataset for Phase 1, combining three tasks:
  1. Chronos forecasting (pretraining on time-series prediction)
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_FRAC = 0.1
VAL_FRAC = 0.1

PATCH_SIZE = 16
PRED_LEN = 96           # Forecast horizon in timesteps
HISTORY_LENS = [96, 192, 384, 416]

CHRONOS_DATASET_REGISTRY: Dict[str, List[str]] = {
    "dominick": ["target"],
    "monash_fred_md": ["target"],
    "electricity_15min": ["consumption_kW"],
    "ercot": ["target"],
    "exchange_rate": ["target"],
    "ushcn_daily": ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
    "weatherbench_daily": ['target'],
    "weatherbench_weekly": ['target'],
    "wiki_daily_100k": ['target'],
}

# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Phase1JointDataset(PretrainDataset):
    """Joint Phase-1 dataset: Chronos forecasting + TSQA MCQ + M4 captioning.

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
        pred_len: int = PRED_LEN,
        val_fraction: float = 1.0,
        **kwargs,
    ):
        self.max_samples_per_source = max_samples_per_source
        self.val_fraction = val_fraction
        self.data_seed = data_seed
        self.patch_size = int(patch_size)
        self.pred_len = int(pred_len)
        self.history_lens = [int(h) for h in HISTORY_LENS]

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be > 0, got {self.pred_len}")
        if self.pred_len % self.patch_size != 0:
            raise ValueError(
                f"pred_len ({self.pred_len}) must be divisible by patch_size ({self.patch_size})"
            )

        self.valid_history_lens = [
            h for h in self.history_lens if h > 0 and h % self.patch_size == 0
        ]
        if not self.valid_history_lens:
            raise ValueError(
                f"No valid history lengths for patch_size={self.patch_size}. "
                f"Configured HISTORY_LENS={self.history_lens}"
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
        else:
            keys = target_keys
            ds = ds.map(
                lambda batch: {"target": [seq for k in keys for seq in batch[k]]},
                batched=True,
                remove_columns=ds.column_names,
            )
        ds = ds.cast_column("target", Sequence(feature=Value(dtype="float32")))
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
            del ds_full, train_val, train, val, test
            print(f"[Phase1Joint] Chronos splits saved to {split_dir}")

        if dist.is_initialized():
            dist.barrier()

        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=False)
        val = load_from_disk(os.path.join(split_dir, "val"), keep_in_memory=False)
        test = load_from_disk(os.path.join(split_dir, "test"), keep_in_memory=False)
        return train, val, test

    @staticmethod
    def _znorm(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Z-normalize a 1-D tensor. Returns (normalized, mean, std).
        Uses a consistent epsilon=1e-8 across all tasks."""
        if x.dim() > 1:
            x = x.reshape(-1)          # flatten multivariate to 1-D for scalar stats
        m = x.mean()
        s = x.std().clamp(min=1e-8)
        return (x - m) / s, m.item(), s.item()

    def _process_chronos_row(self, row: dict, mode: str = "train") -> dict | None:
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = len(series)

        history_len = (
            self.rng.choice(self.valid_history_lens)
            if mode == "train"
            else max(self.valid_history_lens)
        )
        window_needed = history_len + self.pred_len
        if T < window_needed:
            return None

        start = self.rng.randint(0, T - window_needed) if mode == "train" else T - window_needed
        window = series[start: start + window_needed]
        window = _last_value_impute(window)

        history = window[:history_len]
        future  = window[history_len:]

        # Normalize using history stats so future is in the same scale as input.
        # Store stats so predictions can be de-normalized back to raw values.
        _, norm_mean, norm_std = self._znorm(history)
        history_norm = (history - norm_mean) / norm_std
        future_norm  = (future  - norm_mean) / norm_std

        return {
            "type": "chronos",
            "history": history_norm.tolist(),
            "future": future_norm.tolist(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_history_patches": history_len // self.patch_size,
            "n_future_patches": self.pred_len // self.patch_size,
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

    def _make_chronos_stubs(self, ds: Dataset, split_name: str, cap) -> List[dict]:
        """Build lightweight Chronos index stubs — NO row processing, NO disk access.

        Each stub is just {"type": "chronos", "_raw_idx": i, "_chronos_split": split_name}.
        The actual window crop + normalisation happens on-demand in _format_sample.
        Too-short series are handled gracefully in the _format_sample fallback.
        """
        n = cap if cap else len(ds)
        n = min(n, len(ds))
        indices = self.rng.sample(range(len(ds)), n)
        return [{"type": "chronos", "_raw_idx": i, "_chronos_split": split_name} for i in indices]

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
            processed = self._process_chronos_row(raw_row, split)
            if processed is None:
                # Series length borderline — fall back to minimum history_len
                series = torch.tensor(raw_row["target"], dtype=torch.float32)
                history_len = min(self.valid_history_lens)
                window_needed = history_len + self.pred_len
                if len(series) < window_needed:
                    # Truly too short; return a minimal valid dict so the
                    # training loop can filter it via valid_mask / zero loss.
                    return {"type": "chronos", "prompt": "", "answer": "",
                            "norm_mean": 0.0, "norm_std": 1.0,
                            "time_series": [], "time_series_lengths": []}
                start = len(series) - window_needed
                history = series[start: start + history_len]
                future  = series[start + history_len: start + window_needed]
                m, s = history.mean().item(), history.std().clamp(min=1e-8).item()
                processed = {
                    "type": "chronos",
                    "history": ((history - m) / s).tolist(),
                    "future":  ((future  - m) / s).tolist(),
                    "norm_mean": m, "norm_std": s,
                    "n_history_patches": history_len // self.patch_size,
                    "n_future_patches":  self.pred_len // self.patch_size,
                }
            row = processed

        # super() calls _get_text_time_series_prompt_list, which sets
        # row["_norm_mean"] and row["_norm_std"] as a side-effect.
        d = super()._format_sample(row)
        d["type"]      = row.get("type", "unknown")
        d["norm_mean"] = row.get("_norm_mean", row.get("norm_mean", 0.0))
        d["norm_std"]  = row.get("_norm_std",  row.get("norm_std",  1.0))
        return d

    def _get_answer(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Answer"]
        # chronos: ground truth as space-separated floats
        future_str = " ".join(f"{v:.4f}" for v in row["future"])
        return f"Answer: <ts> {future_str} </ts>"

    def _get_pre_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Question"]
        return f"Given {row['n_history_patches']} time series patches"

    def _get_post_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return "Predict the " + row["Task"] + " Answer:"
        return f"</ts>, please predict the next {row['n_future_patches']} patches"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if row["type"] == "chronos":
            # Already z-normalized in _process_chronos_row using history stats.
            history = torch.tensor(row["history"], dtype=torch.float32)
            row["_norm_mean"] = row.get("norm_mean", 0.0)
            row["_norm_std"]  = row.get("norm_std",  1.0)
            return [TextTimeSeriesPrompt("<ts>", history.tolist())]

        # tsqa
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)
        if series.dim() > 1:
            series = series.mean(dim=-1)
        normalized, norm_mean, norm_std = self._znorm(series)
        row["_norm_mean"] = norm_mean
        row["_norm_std"]  = norm_std
        return [TextTimeSeriesPrompt(
            f"This is the time series, it has mean {norm_mean:.4f} and std {norm_std:.4f}.",
            normalized.tolist(),
        )]


class Phase1JointExtendDataset(Phase1JointDataset):
    """Extended Phase-1 dataset: Chronos forecasting + TSQA MCQ + M4 captioning.

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
                normalized = ((series_tensor - mean) / std).tolist()
            else:
                normalized = (series_tensor - mean).tolist()
            row["_norm_mean"] = mean
            row["_norm_std"] = std if std > 0 else 1.0
            text_prompt = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:"
            return [TextTimeSeriesPrompt(text_prompt, normalized)]
        return super()._get_text_time_series_prompt_list(row)
