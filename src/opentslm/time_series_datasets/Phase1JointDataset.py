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
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import (
    Dataset,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.ChronosDataset import (
    BASE_DATA_PATH,
    CHRONOS_DATASET_REGISTRY,
    PREPROCESSED_SPLITS_DIR,
    _last_value_impute,
)
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.m4.m4_loader import load_all_m4_data, create_combined_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_FRAC = 0.1
VAL_FRAC = 0.1

PATCH_SIZE = 4
PRED_LEN = 96           # 24 patches × patch_size 4
HISTORY_LENS = [96, 192, 384, 416]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Phase1JointDataset(QADataset):
    """Joint Phase-1 dataset: Chronos forecasting + TSQA MCQ + M4 captioning.

    Each split is balanced to the size of the smallest sub-split so that no
    single task dominates training.  Rows are tagged with a ``"type"`` key
    (``"chronos"``, ``"tsqa"``, or ``"m4"``) to drive dispatch in the format
    methods.

    Args:
        max_samples_per_source: If set, cap each data source (chronos, tsqa, m4)
            to this many samples per split before balancing.  Useful for quick
            smoke-tests (e.g. ``max_samples_per_source=100``).
    """

    def __init__(self, *args, max_samples_per_source: int | None = None, **kwargs):
        self.max_samples_per_source = max_samples_per_source
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

    def _process_chronos_row(self, row: dict, mode: str = "train") -> dict | None:
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = len(series)

        history_len = random.choice(HISTORY_LENS) if mode == "train" else max(HISTORY_LENS)
        window_needed = history_len + PRED_LEN
        if T < window_needed:
            return None

        start = random.randint(0, T - window_needed) if mode == "train" else T - window_needed
        window = series[start: start + window_needed]
        window = _last_value_impute(window)

        history = window[:history_len]
        future  = window[history_len:]

        mean = history.mean()
        std  = history.std().clamp(min=1e-4)
        history_norm = (history - mean) / std
        future_norm  = (future  - mean) / std

        return {
            "type": "chronos",
            "history": history_norm.tolist(),
            "future": future_norm.tolist(),
            "n_history_patches": history_len // PATCH_SIZE,
            "n_future_patches": PRED_LEN // PATCH_SIZE,
        }

    def _sample_and_process_chronos(self, ds: Dataset, n: int, mode: str) -> List[dict]:
        n = min(n, len(ds))
        indices = random.sample(range(len(ds)), n)
        rows = ds.select(indices)
        result = []
        for row in rows:
            processed = self._process_chronos_row(row, mode)
            if processed is not None:
                result.append(processed)
        return result

    def _load_splits(self) -> Tuple[list, list, list]:
        # ---- TSQA (stage1_mcq) — rank-0-first to avoid NFS filelock errors ----
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if rank == 0:
            tsqa_full = load_dataset("ChengsenWang/TSQA", split="train")
        if is_distributed:
            dist.barrier()
        if rank != 0:
            tsqa_full = load_dataset("ChengsenWang/TSQA", split="train")

        tsqa_tv, tsqa_test = tsqa_full.train_test_split(test_size=TEST_FRAC, seed=42).values()
        tsqa_train, tsqa_val = tsqa_tv.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()
        cap = getattr(self, "max_samples_per_source", None)

        def _cap(lst):
            return random.sample(lst, min(cap, len(lst))) if cap else lst

        tsqa_train_list = _cap([{"type": "tsqa", **r} for r in tsqa_train])
        tsqa_val_list   = _cap([{"type": "tsqa", **r} for r in tsqa_val])
        tsqa_test_list  = _cap([{"type": "tsqa", **r} for r in tsqa_test])

        # ---- M4 (stage2_captioning) — loads from local files, no lock issues ----
        data_dict = load_all_m4_data()
        m4_train_raw, m4_val_raw, m4_test_raw = create_combined_dataset(data_dict, seed=42)
        m4_train_list = _cap([{"type": "m4", **r} for r in m4_train_raw])
        m4_val_list   = _cap([{"type": "m4", **r} for r in m4_val_raw])
        m4_test_list  = _cap([{"type": "m4", **r} for r in m4_test_raw])

        # ---- Chronos forecasting ----
        chronos_train_raw, chronos_val_raw, chronos_test_raw = self._load_or_build_chronos_raw()
        chronos_n_train = cap if cap else min(len(tsqa_train_list), len(m4_train_list))
        chronos_n_val   = cap if cap else min(len(tsqa_val_list),   len(m4_val_list))
        chronos_n_test  = cap if cap else min(len(tsqa_test_list),  len(m4_test_list))
        chronos_train_list = self._sample_and_process_chronos(chronos_train_raw, chronos_n_train, "train")
        chronos_val_list   = self._sample_and_process_chronos(chronos_val_raw,   chronos_n_val,   "validation")
        chronos_test_list  = self._sample_and_process_chronos(chronos_test_raw,  chronos_n_test,  "test")

        # ---- Balance all three to the smallest, combine, shuffle ----
        def balance_and_combine(tsqa: list, m4: list, chronos: list) -> list:
            n = min(len(tsqa), len(m4), len(chronos))
            combined = (
                random.sample(tsqa, n)
                + random.sample(m4, n)
                + random.sample(chronos, n)
            )
            random.shuffle(combined)
            return combined

        train = balance_and_combine(tsqa_train_list, m4_train_list, chronos_train_list)
        val   = balance_and_combine(tsqa_val_list,   m4_val_list,   chronos_val_list)
        test  = balance_and_combine(tsqa_test_list,  m4_test_list,  chronos_test_list)

        print(
            f"[Phase1JointDataset] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  tsqa    — train={len(tsqa_train_list)}, val={len(tsqa_val_list)}, test={len(tsqa_test_list)}\n"
            f"  m4      — train={len(m4_train_list)}, val={len(m4_val_list)}, test={len(m4_test_list)}\n"
            f"  chronos — train={len(chronos_train_list)}, val={len(chronos_val_list)}, test={len(chronos_test_list)}"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Format dispatch
    # ------------------------------------------------------------------

    def _format_sample(self, row):
        d = super()._format_sample(row)
        d["type"] = row.get("type", "unknown")
        return d

    def _get_answer(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Answer"]
        if row["type"] == "m4":
            return row["caption"]
        # chronos: ground truth as space-separated floats
        future_str = " ".join(f"{v:.4f}" for v in row["future"])
        return f"Answer: <ts> {future_str} </ts>"

    def _get_pre_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return row["Question"]
        if row["type"] == "m4":
            return "You are an expert in time series analysis."
        return f"Given {row['n_history_patches']} time series patches"

    def _get_post_prompt(self, row) -> str:
        if row["type"] == "tsqa":
            return "Predict the " + row["Task"] + " Answer:"
        if row["type"] == "m4":
            return "Please generate a detailed caption for this time-series, describing it as accurately as possible."
        return f"</ts>, please predict the next {row['n_future_patches']} patches"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if row["type"] == "chronos":
            history = torch.tensor(row["history"], dtype=torch.float32)
            return [TextTimeSeriesPrompt("<ts>", history.tolist())]

        if row["type"] == "tsqa":
            series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)
            means = series.mean(dim=0, keepdim=True)
            stds  = series.std(dim=0, keepdim=True)
            series = (series - means) / (stds + 1e-8)
            mean_val = means.flatten()[0].item()
            std_val  = stds.flatten()[0].item()
            return [TextTimeSeriesPrompt(
                f"This is the time series, it has mean {mean_val:.4f} and std {std_val:.4f}.",
                series.tolist(),
            )]

        # m4
        series = row["series"]
        if isinstance(series, list):
            series_tensor = torch.tensor(series, dtype=torch.float32)
        else:
            series_tensor = series
        mean = series_tensor.mean()
        std  = series_tensor.std()
        normalized = (series_tensor - mean) / std if std > 0 else series_tensor - mean
        return [TextTimeSeriesPrompt(
            f"This is the time series, it has mean {mean:.4f} and std {std:.4f}:",
            normalized.tolist(),
        )]
