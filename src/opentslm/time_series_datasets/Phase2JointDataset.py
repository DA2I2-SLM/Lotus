# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Phase2JointDataset.py
---------------------
Joint training dataset for Phase 2, combining:
  1. Chronos forecasting — LLM text output: "<ts> v1 v2 ... </ts>"
     Prompt: "<dataset_desc>. You are given T timesteps: <ts> [patches] </ts>.
              Predict the next Y timesteps"
  2. HAR Chain-of-Thought QA  (stage3_cot)
  3. SleepEDF Chain-of-Thought QA  (stage4_sleep_cot)
  4. ECG Chain-of-Thought QA  (stage5_ecg_cot)

The Chronos component uses a separate registry and preprocessed split cache
from Phase1JointDataset, so the two phases never interfere.
"""

import os
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    DownloadMode,
)

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.ChronosDataset import (
    BASE_DATA_PATH,
    CHRONOS_DATASET_REGISTRY,
    PREPROCESSED_SPLITS_DIR,
    _last_value_impute,
)
from opentslm.time_series_datasets.Phase1JointDataset import Phase1JointDataset

# ---------------------------------------------------------------------------
# Phase 2 Chronos registry — diverse datasets different from Phase 1
# ---------------------------------------------------------------------------

PHASE2_CHRONOS_REGISTRY: Dict[str, List[str]] = {
    "dominick":          ["target"],
    "monash_fred_md":    ["target"],
    "electricity_15min": ["consumption_kW"],
    "ercot":             ["target"],
    "exchange_rate":     ["target"],
    "ushcn_daily":       ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"],
    "weatherbench_daily":  ["target"],
    "weatherbench_weekly": ["target"],
    "wiki_daily_100k":     ["target"],
}


def _load_dataset_description(dataset_name: str, fallback: str = "time series data") -> str:
    """Load description from BASE_DATA_PATH/<dataset_name>/description.txt.
    Returns ``fallback`` if the file does not exist."""
    desc_path = os.path.join(BASE_DATA_PATH, dataset_name, "description.txt")
    try:
        with open(desc_path) as f:
            return f.read().strip()
    except OSError:
        return fallback


# Build at import time; falls back gracefully if files are missing.
PHASE2_DATASET_DESCRIPTIONS: Dict[str, str] = {
    name: _load_dataset_description(name)
    for name in PHASE2_CHRONOS_REGISTRY
}

# Separate on-disk cache from Phase 1 (different registry → different split)
PHASE2_SPLITS_DIR = os.path.join(PREPROCESSED_SPLITS_DIR, "_splits_phase2")

# Constants (inherit from Phase1JointDataset where not overridden)
TEST_FRAC = 0.1
VAL_FRAC  = 0.1


class Phase2JointDataset(Phase1JointDataset):
    """Joint Phase-3 dataset: Chronos forecasting (LLM text output) +
    HAR CoT + SleepEDF CoT + ECG CoT.

    Chronos uses the new prompt format:
        "<dataset_desc>. You are given T timesteps: <ts>[patches]</ts>.
         Predict the next Y timesteps"
    Answer: "<ts> v1 v2 ... </ts>"

    CoT datasets are loaded eagerly and stored as pre-formatted dicts.
    """

    # Separate class-level cache from Phase1JointDataset
    _phase2_chronos_raw_cache: Dict = {}

    # ------------------------------------------------------------------
    # Chronos split building — Phase 2 registry + dataset_name column
    # ------------------------------------------------------------------

    def _load_or_build_chronos_raw(self):
        """Build / load the Phase 2 Chronos HF Dataset splits.

        Adds a ``dataset_name`` column so the prompt can include a
        human-readable description of the source dataset.
        """
        split_dir  = PHASE2_SPLITS_DIR
        done_flag  = os.path.join(split_dir, "_DONE")

        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if rank == 0 and not os.path.exists(done_flag):
            print("[Phase2Joint] Building Chronos splits for Phase 2 ...")
            all_parts: List[Dataset] = []

            for dataset_name, columns in PHASE2_CHRONOS_REGISTRY.items():
                ds_path = os.path.join(PREPROCESSED_SPLITS_DIR, dataset_name)
                if not os.path.exists(os.path.join(ds_path, "_DONE")):
                    print(f"  [Phase3] Skipping {dataset_name} — not preprocessed yet")
                    continue
                try:
                    # Each preprocessed dataset stores splits as subdirectories
                    # (train/ val/ test/) — load and concatenate all three so we
                    # can do our own phase-2 split afterwards.
                    split_parts = [
                        load_from_disk(os.path.join(ds_path, s), keep_in_memory=False)
                        for s in ("train", "val", "test")
                        if os.path.exists(os.path.join(ds_path, s))
                    ]
                    ds_full = (
                        concatenate_datasets(split_parts)
                        if len(split_parts) > 1 else split_parts[0]
                    )
                    for col in columns:
                        if col not in ds_full.column_names:
                            continue
                        # Keep only the target column, rename to "target" if needed
                        ds_col = ds_full.select_columns([col])
                        if col != "target":
                            ds_col = ds_col.rename_column(col, "target")
                        # Add dataset_name column for prompt description
                        ds_col = ds_col.add_column(
                            "dataset_name", [dataset_name] * len(ds_col)
                        )
                        all_parts.append(ds_col)
                except Exception as e:
                    print(f"  [Phase3] Could not load {dataset_name}: {e}")
                    continue

            if not all_parts:
                raise RuntimeError(
                    "[Phase2Joint] No Chronos datasets loaded. "
                    "Ensure PHASE2_CHRONOS_REGISTRY datasets are preprocessed."
                )

            ds_full = concatenate_datasets(all_parts)
            ds_full = ds_full.shuffle(seed=42)

            n       = len(ds_full)
            n_test  = max(1, int(n * TEST_FRAC))
            n_val   = max(1, int(n * VAL_FRAC))
            n_train = n - n_test - n_val

            train = ds_full.select(range(n_train))
            val   = ds_full.select(range(n_train, n_train + n_val))
            test  = ds_full.select(range(n_train + n_val, n))

            os.makedirs(split_dir, exist_ok=True)
            train.save_to_disk(os.path.join(split_dir, "train"))
            val.save_to_disk(os.path.join(split_dir, "val"))
            test.save_to_disk(os.path.join(split_dir, "test"))
            with open(done_flag, "w") as fh:
                fh.write("done\n")
            print(f"[Phase2Joint] Chronos splits saved → {split_dir}")

        if is_distributed:
            dist.barrier()

        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=False)
        val   = load_from_disk(os.path.join(split_dir, "val"),   keep_in_memory=False)
        test  = load_from_disk(os.path.join(split_dir, "test"),  keep_in_memory=False)
        return train, val, test

    # ------------------------------------------------------------------
    # CoT dataset loader — loads HAR / Sleep / ECG eagerly
    # ------------------------------------------------------------------

    def _load_cot_splits(self, cap) -> Tuple[list, list, list]:
        """Instantiate each CoT dataset, collect pre-formatted samples."""
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

        cot_specs = [
            (HARCoTQADataset,      "har"),
            (SleepEDFCoTQADataset, "sleep"),
            (ECGQACoTQADataset,    "ecg"),
        ]

        result: Dict[str, list] = {"train": [], "validation": [], "test": []}

        for ds_cls, task_type in cot_specs:
            for split_name in ("train", "validation", "test"):
                try:
                    ds = ds_cls(split_name, self.EOS_TOKEN)
                except Exception as e:
                    print(f"[Phase2Joint] Could not load {ds_cls.__name__} {split_name}: {e}")
                    continue

                n = min(cap, len(ds)) if cap else len(ds)
                indices = self.rng.sample(range(len(ds)), n)

                for i in indices:
                    try:
                        formatted = ds[i]          # calls ds._format_sample(row)
                    except Exception:
                        continue
                    formatted["type"]          = task_type
                    formatted["_preformatted"] = True
                    result[split_name].append(formatted)

            print(
                f"[Phase2Joint] {ds_cls.__name__}: "
                f"train={len([r for r in result['train']     if r['type'] == task_type])}, "
                f"val={len([r for r in result['validation']  if r['type'] == task_type])}, "
                f"test={len([r for r in result['test']       if r['type'] == task_type])}"
            )

        return result["train"], result["validation"], result["test"]

    # ------------------------------------------------------------------
    # _load_splits — Chronos (Phase 2 registry) + CoT datasets
    # ------------------------------------------------------------------

    def _load_splits(self) -> Tuple[list, list, list]:
        cap = self.max_samples_per_source

        # ── Phase-3 Chronos cache (separate from Phase1JointDataset) ──
        if not Phase2JointDataset._phase2_chronos_raw_cache:
            tr, va, te = self._load_or_build_chronos_raw()
            Phase2JointDataset._phase2_chronos_raw_cache = {
                "train": tr, "validation": va, "test": te
            }
        cache = Phase2JointDataset._phase2_chronos_raw_cache

        # ── Chronos stubs (lazy) ──
        c_t  = self._make_chronos_stubs(cache["train"],      "train",      cap)
        c_v  = self._make_chronos_stubs(cache["validation"], "validation", cap)
        c_te = self._make_chronos_stubs(cache["test"],       "test",       cap)

        # ── CoT datasets (eager, pre-formatted) ──
        cot_t, cot_v, cot_te = self._load_cot_splits(cap)

        # Subsample val per source to preserve task ratio
        if self.val_fraction < 1.0:
            c_v   = c_v[:max(1,   int(len(c_v)   * self.val_fraction))]
            cot_v = cot_v[:max(1, int(len(cot_v) * self.val_fraction))]

        def _merge(*lists):
            combined = []
            for lst in lists:
                combined.extend(lst)
            self.rng.shuffle(combined)
            return combined

        train = _merge(c_t,  cot_t)
        val   = _merge(c_v,  cot_v)
        test  = _merge(c_te, cot_te)

        print(
            f"[Phase2JointDataset] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  chronos — train={len(c_t)} (lazy), val={len(c_v)} (lazy), test={len(c_te)} (lazy)\n"
            f"  cot     — train={len(cot_t)}, val={len(cot_v)}, test={len(cot_te)}"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # _format_sample — dispatch: pre-formatted CoT or Chronos stub
    # ------------------------------------------------------------------

    def _process_chronos_row(self, row: dict, mode: str = "train") -> dict | None:
        """Prepare Chronos rows for true history-to-future forecasting."""
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
        future = window[history_len:]

        _, norm_mean, norm_std = self._znorm(history)
        history_norm = (history - norm_mean) / norm_std
        future_norm = (future - norm_mean) / norm_std

        return {
            "type": "chronos",
            "history": history_norm.tolist(),
            "future": future_norm.tolist(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_history_patches": history_len // self.patch_size,
            "n_future_patches": self.pred_len // self.patch_size,
        }

    def _format_sample(self, row):
        # Pre-formatted CoT sample (HAR / Sleep / ECG) — return as-is
        if row.get("_preformatted"):
            return row

        # Chronos stub — lazy resolution using Phase 2 cache
        if row.get("type") == "chronos" and "_raw_idx" in row:
            split   = row["_chronos_split"]
            raw_row = Phase2JointDataset._phase2_chronos_raw_cache[split][row["_raw_idx"]]
            processed = self._process_chronos_row(raw_row, split)

            if processed is None:
                # Too short — try with minimum history
                series = torch.tensor(raw_row["target"], dtype=torch.float32)
                history_len   = min(self.valid_history_lens)
                window_needed = history_len + self.pred_len
                if len(series) < window_needed:
                    return {"type": "chronos", "_preformatted": True,
                            "pre_prompt": "", "post_prompt": "", "answer": "",
                            "time_series": [], "time_series_lengths": [],
                            "norm_mean": 0.0, "norm_std": 1.0}
                start   = len(series) - window_needed
                history = series[start: start + history_len]
                future  = series[start + history_len: start + window_needed]
                m, s    = history.mean().item(), history.std().clamp(min=1e-8).item()
                processed = {
                    "type":             "chronos",
                    "history":          ((history - m) / s).tolist(),
                    "future":           ((future  - m) / s).tolist(),
                    "norm_mean":        m,
                    "norm_std":         s,
                    "n_history_patches": history_len // self.patch_size,
                    "n_future_patches":  self.pred_len // self.patch_size,
                    "dataset_name":     raw_row.get("dataset_name", "unknown"),
                }

            processed["dataset_name"] = raw_row.get("dataset_name", "unknown")
            row = processed

        from opentslm.time_series_datasets.PretrainDataset import PretrainDataset
        d = PretrainDataset._format_sample(self, row)
        d["type"]      = row.get("type", "unknown")
        d["norm_mean"] = row.get("_norm_mean", row.get("norm_mean", 0.0))
        d["norm_std"]  = row.get("_norm_std",  row.get("norm_std",  1.0))
        if row.get("type") == "chronos" and "future" in row:
            # Required by compute_phase2_loss for the MSE decoder path
            d["ts_already_normalized"] = True
            d["time_series_norm_stats"] = [(d["norm_mean"], d["norm_std"])]
            d["is_forecasting"] = True
            d["ts_future"] = torch.tensor(row["future"], dtype=torch.float32)
        return d

    # ------------------------------------------------------------------
    # Prompt methods — Chronos uses new descriptive format
    # ------------------------------------------------------------------

    def _get_pre_prompt(self, row) -> str:
        if row.get("type") == "chronos":
            desc = PHASE2_DATASET_DESCRIPTIONS.get(
                row.get("dataset_name", ""), "time series"
            )
            T = len(row["history"])
            return f"This is {desc} data. You are given {T} time series timesteps:"
        return ""   # CoT rows are pre-formatted; this won't be called

    def _get_post_prompt(self, row) -> str:
        if row.get("type") == "chronos":
            return f". Predict the next {self.pred_len} timesteps"
        return ""

    def _get_answer(self, row) -> str:
        if row.get("type") == "chronos":
            future_str = " ".join(f"{v:.4f}" for v in row["future"])
            return f"<ts> {future_str} </ts>"
        return ""

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if row.get("type") == "chronos":
            history = torch.tensor(row["history"], dtype=torch.float32)
            row["_norm_mean"] = row.get("norm_mean", 0.0)
            row["_norm_std"]  = row.get("norm_std",  1.0)
            return [TextTimeSeriesPrompt("<ts>", history.tolist())]
        return []

    def _get_raw_ts(self, row):
        if row.get("type") == "chronos":
            return row.get("history", [])
        return []
