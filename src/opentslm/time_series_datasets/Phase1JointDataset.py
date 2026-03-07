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

    def __init__(
        self,
        *args,
        max_samples_per_source: int | None = None,
        data_seed: int = 42,
        **kwargs,
    ):
        self.max_samples_per_source = max_samples_per_source
        self.data_seed = data_seed
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

        history_len = self.rng.choice(HISTORY_LENS) if mode == "train" else max(HISTORY_LENS)
        window_needed = history_len + PRED_LEN
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
            "n_history_patches": history_len // PATCH_SIZE,
            "n_future_patches": PRED_LEN // PATCH_SIZE,
        }

    def _sample_and_process_chronos(self, ds: Dataset, n: int, mode: str) -> List[dict]:
        n = min(n, len(ds))
        indices = self.rng.sample(range(len(ds)), n)
        rows = ds.select(indices)
        result = []
        for row in rows:
            processed = self._process_chronos_row(row, mode)
            if processed is not None:
                result.append(processed)
        return result

    def _load_tsqa_and_chronos(self):
        """Load and return (tsqa_train, tsqa_val, tsqa_test, chronos_train, chronos_val, chronos_test).

        Shared by Phase1JointDataset and Phase1JointExtendDataset.
        """
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
            return self.rng.sample(lst, min(cap, len(lst))) if cap else lst

        tsqa_train_list = _cap([{"type": "tsqa", **r} for r in tsqa_train])
        tsqa_val_list   = _cap([{"type": "tsqa", **r} for r in tsqa_val])
        tsqa_test_list  = _cap([{"type": "tsqa", **r} for r in tsqa_test])

        chronos_train_raw, chronos_val_raw, chronos_test_raw = self._load_or_build_chronos_raw()
        chronos_n_train = cap if cap else len(tsqa_train_list)
        chronos_n_val   = cap if cap else len(tsqa_val_list)
        chronos_n_test  = cap if cap else len(tsqa_test_list)
        chronos_train_list = self._sample_and_process_chronos(chronos_train_raw, chronos_n_train, "train")
        chronos_val_list   = self._sample_and_process_chronos(chronos_val_raw,   chronos_n_val,   "validation")
        chronos_test_list  = self._sample_and_process_chronos(chronos_test_raw,  chronos_n_test,  "test")

        return (
            tsqa_train_list, tsqa_val_list, tsqa_test_list,
            chronos_train_list, chronos_val_list, chronos_test_list,
        )

    def _load_splits(self) -> Tuple[list, list, list]:
        tsqa_train_list, tsqa_val_list, tsqa_test_list, \
        chronos_train_list, chronos_val_list, chronos_test_list = self._load_tsqa_and_chronos()

        # ---- Balance TSQA + Chronos to the smallest, combine, shuffle ----
        def balance_and_combine(tsqa: list, chronos: list) -> list:
            n = min(len(tsqa), len(chronos))
            combined = self.rng.sample(tsqa, n) + self.rng.sample(chronos, n)
            self.rng.shuffle(combined)
            return combined

        train = balance_and_combine(tsqa_train_list, chronos_train_list)
        val   = balance_and_combine(tsqa_val_list,   chronos_val_list)
        test  = balance_and_combine(tsqa_test_list,  chronos_test_list)

        print(
            f"[Phase1JointDataset] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  tsqa    — train={len(tsqa_train_list)}, val={len(tsqa_val_list)}, test={len(tsqa_test_list)}\n"
            f"  chronos — train={len(chronos_train_list)}, val={len(chronos_val_list)}, test={len(chronos_test_list)}"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Format dispatch
    # ------------------------------------------------------------------

    def _format_sample(self, row):
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
            # Already z-normalized in _process_chronos_row using history stats.
            # Norm stats are in row["norm_mean"] / row["norm_std"].
            history = torch.tensor(row["history"], dtype=torch.float32)
            row["_norm_mean"] = row.get("norm_mean", 0.0)
            row["_norm_std"]  = row.get("norm_std",  1.0)
            return [TextTimeSeriesPrompt("<ts>", history.tolist())]

        if row["type"] == "tsqa":
            series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)
            # Collapse multivariate to univariate (encoder only accepts [T] or [T,1])
            if series.dim() > 1:
                series = series.mean(dim=-1)
            normalized, norm_mean, norm_std = self._znorm(series)
            row["_norm_mean"] = norm_mean
            row["_norm_std"]  = norm_std
            return [TextTimeSeriesPrompt(
                f"This is the time series, it has mean {norm_mean:.4f} and std {norm_std:.4f}.",
                normalized.tolist(),
            )]

        # m4
        series = row["series"]
        if not isinstance(series, torch.Tensor):
            series = torch.tensor(series, dtype=torch.float32)
        if series.dim() > 1:
            series = series.mean(dim=-1)
        normalized, norm_mean, norm_std = self._znorm(series)
        row["_norm_mean"] = norm_mean
        row["_norm_std"]  = norm_std
        return [TextTimeSeriesPrompt(
            f"This is the time series, it has mean {norm_mean:.4f} and std {norm_std:.4f}:",
            normalized.tolist(),
        )]


class Phase1JointExtendDataset(Phase1JointDataset):
    """Extended Phase-1 dataset: Chronos forecasting + TSQA MCQ + M4 captioning.

    Same as Phase1JointDataset but adds M4 captioning as a third task.
    All three splits are balanced to the smallest sub-split.
    CE loss is applied to TSQA and M4; MSE to Chronos.
    """

    def _load_splits(self) -> Tuple[list, list, list]:
        # Load TSQA + Chronos via parent
        tsqa_train_list, tsqa_val_list, tsqa_test_list, \
        chronos_train_list, chronos_val_list, chronos_test_list = self._load_tsqa_and_chronos()

        # ---- M4 (stage2_captioning) ----
        cap = getattr(self, "max_samples_per_source", None)

        def _cap(lst):
            return self.rng.sample(lst, min(cap, len(lst))) if cap else lst

        data_dict = load_all_m4_data()
        m4_train_raw, m4_val_raw, m4_test_raw = create_combined_dataset(data_dict, seed=42)
        m4_train_list = _cap([{"type": "m4", **r} for r in m4_train_raw])
        m4_val_list   = _cap([{"type": "m4", **r} for r in m4_val_raw])
        m4_test_list  = _cap([{"type": "m4", **r} for r in m4_test_raw])

        # ---- Balance all three, combine, shuffle ----
        def balance_and_combine(tsqa: list, chronos: list, m4: list) -> list:
            n = min(len(tsqa), len(chronos), len(m4))
            combined = (
                self.rng.sample(tsqa, n)
                + self.rng.sample(chronos, n)
                + self.rng.sample(m4, n)
            )
            self.rng.shuffle(combined)
            return combined

        train = balance_and_combine(tsqa_train_list, chronos_train_list, m4_train_list)
        val   = balance_and_combine(tsqa_val_list,   chronos_val_list,   m4_val_list)
        test  = balance_and_combine(tsqa_test_list,  chronos_test_list,  m4_test_list)

        print(
            f"[Phase1JointExtendDataset] train={len(train)}, val={len(val)}, test={len(test)}\n"
            f"  tsqa    — train={len(tsqa_train_list)}, val={len(tsqa_val_list)}, test={len(tsqa_test_list)}\n"
            f"  chronos — train={len(chronos_train_list)}, val={len(chronos_val_list)}, test={len(chronos_test_list)}\n"
            f"  m4      — train={len(m4_train_list)}, val={len(m4_val_list)}, test={len(m4_test_list)}"
        )
        return train, val, test
