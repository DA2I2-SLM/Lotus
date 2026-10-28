# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import ast
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
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_FRAC = 0.1
VAL_FRAC = 0.1



def _chatts_data_path() -> str:
    """ChatTS align_256 split under $FTS_CHATTS_DIR (raises when unset)."""
    from opentslm.time_series_datasets.constants import get_chatts_dir
    return os.path.join(get_chatts_dir(), "process", "align_256")

# Patch size used to count "patches" in the forecasting prompt
PATCH_SIZE = 4

# Fixed prediction horizon (Chronos-2 style: one canonical forecast length).
PRED_LEN = 96  # 24 patches × patch_size 4

# Chronos-2 uses multiple fixed context lengths drawn uniformly per sample.
# All values must be multiples of PATCH_SIZE; max + PRED_LEN ≤ MAX_SEQ_LEN (512).
HISTORY_LENS = [96, 192, 384, 416]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_value_count(_key: str, dataset: Dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )
    for value in dataloader:
        print(value)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class Stage2AllDataset(QADataset):
    """Mixed dataset for Stage-2 training.

    Combines:
    1. ChatTS alignment data (text + time-series Q&A).
    2. Chronos time-series forecasting data, formatted as:

       Input  : "Given {X} time series patches <ts> [history] </ts>,
                 please predict the next {Y} patches"
       Output : "Answer: <ts> {space-separated float values} </ts>"

    Both sub-datasets are balanced to the size of the smaller one per split.
    """

    # ------------------------------------------------------------------
    # Chronos loading helpers
    # ------------------------------------------------------------------

    def _normalize_to_target(self, ds: Dataset, target_keys: List[str]) -> Dataset:
        """Normalise dataset columns to a single ``target`` column."""
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
        """Return (train, val, test) HuggingFace Datasets with a ``target`` column.

        Reuses ChronosDataset's cached splits when available; builds them from
        scratch otherwise.  Follows the same rank-0-writes / barrier / all-read
        pattern as :class:`ChronosDataset`.
        """
        rank = int(os.environ.get("RANK", 0))
        split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all")
        done_flag = os.path.join(split_dir, "_DONE")

        if rank == 0 and not os.path.exists(done_flag):
            print("[Stage2] Building Chronos splits (this runs once)…")
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
            print(f"  -> concatenated: {len(ds_full)} series total")

            train_val, test = ds_full.train_test_split(
                test_size=TEST_FRAC, seed=42
            ).values()
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
            print(f"[Stage2] Chronos splits saved to {split_dir}")

        if dist.is_initialized():
            dist.barrier()

        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=False)
        val = load_from_disk(os.path.join(split_dir, "val"), keep_in_memory=False)
        test = load_from_disk(os.path.join(split_dir, "test"), keep_in_memory=False)
        return train, val, test

    def _process_chronos_row(self, row: dict, mode: str = "train") -> dict | None:
        """Convert a raw Chronos row to a forecasting dict (Chronos-2 style).

        History length is sampled uniformly from ``HISTORY_LENS`` on every
        call (train) or fixed to the longest option (val/test).  Prediction
        length is always ``PRED_LEN = 96`` values.  Returns ``None`` when the
        series is too short for the requested window.
        """
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = len(series)

        # --- Sample a fixed context length (Chronos-2: multiple fixed lengths) ---
        history_len = random.choice(HISTORY_LENS) if mode == "train" else max(HISTORY_LENS)
        window_needed = history_len + PRED_LEN

        if T < window_needed:
            return None  # series too short for this history/pred combination

        # --- Crop: random window (train) or last window (val/test) ---
        if mode == "train":
            start = random.randint(0, T - window_needed)
        else:
            start = T - window_needed
        window = series[start : start + window_needed]

        # Forward-fill NaN / Inf
        window = _last_value_impute(window)

        history = window[:history_len]
        future  = window[history_len:]  # exactly PRED_LEN = 96 values

        # Normalise using history statistics only (future is unknown at inference)
        mean = history.mean()
        std  = history.std().clamp(min=1e-4)
        history_norm = (history - mean) / std
        future_norm  = (future  - mean) / std

        return {
            "type": "forecasting",
            "history": history_norm.numpy(),
            "future": future_norm.numpy(),
            "n_history_patches": history_len // PATCH_SIZE,
            "n_future_patches":  PRED_LEN     // PATCH_SIZE,  # always 24
        }

    def _sample_and_process_chronos(
        self, ds: Dataset, n: int, mode: str
    ) -> List[dict]:
        """Randomly sample *n* rows from *ds* and convert to forecasting dicts."""
        n = min(n, len(ds))
        indices = random.sample(range(len(ds)), n)
        rows = ds.select(indices)
        result = []
        for row in rows:
            processed = self._process_chronos_row(row, mode)
            if processed is not None:
                result.append(processed)
        return result

    # ------------------------------------------------------------------
    # QADataset interface
    # ------------------------------------------------------------------

    def _format_sample(self, row: dict) -> dict:
        """Extend the base dict with forecasting-specific fields.

        Adds two keys consumed by the model's ``compute_phase2_loss``:
          - ``is_forecasting`` (bool): True for Chronos forecasting rows.
          - ``ts_future`` (1-D float Tensor | None): ground-truth future values
            for MSE loss; None for ChatTS rows.
        """
        d = super()._format_sample(row)
        if row.get("type") == "forecasting":
            d["is_forecasting"] = True
            d["ts_future"] = torch.as_tensor(row["future"], dtype=torch.float32)
        else:
            d["is_forecasting"] = False
            d["ts_future"] = None
        return d

    def _load_splits(self) -> Tuple[list, list, list]:
        # ---- ChatTS data ----
        chatts_full = load_dataset(
            _chatts_data_path(), split="train", trust_remote_code=True
        )
        chatts_tv, chatts_test = chatts_full.train_test_split(
            test_size=TEST_FRAC, seed=42
        ).values()
        chatts_train, chatts_val = chatts_tv.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        # # only take 500 samples from each split to keep the total dataset size manageable
        # chatts_train = chatts_train.shuffle(seed=42).select(range(500))
        # chatts_val = chatts_val.shuffle(seed=42).select(range(500))
        # chatts_test = chatts_test.shuffle(seed=42).select(range(500))

        chatts_train_list = [{"type": "chatts", **r} for r in chatts_train]
        chatts_val_list = [{"type": "chatts", **r} for r in chatts_val]
        chatts_test_list = [{"type": "chatts", **r} for r in chatts_test]

        # ---- Chronos forecasting data ----
        chronos_train_raw, chronos_val_raw, chronos_test_raw = (
            self._load_or_build_chronos_raw()
        )

        # Sample to match ChatTS split sizes, then process
        chronos_train_list = self._sample_and_process_chronos(
            chronos_train_raw, len(chatts_train_list), "train"
        )
        chronos_val_list = self._sample_and_process_chronos(
            chronos_val_raw, len(chatts_val_list), "validation"
        )
        chronos_test_list = self._sample_and_process_chronos(
            chronos_test_raw, len(chatts_test_list), "test"
        )

        # ---- Balance to the smaller set, then combine and shuffle ----
        def balance_and_combine(chatts: list, chronos: list) -> list:
            n = min(len(chatts), len(chronos))
            combined = random.sample(chatts, n) + random.sample(chronos, n)
            random.shuffle(combined)
            return combined

        train = balance_and_combine(chatts_train_list, chronos_train_list)
        val   = balance_and_combine(chatts_val_list,   chronos_val_list)
        test  = balance_and_combine(chatts_test_list,  chronos_test_list)

        print(f"[Stage2AllDataset] train={len(train)}, val={len(val)}, test={len(test)}")
        print(f"  chatts  — train={len(chatts_train_list)}, val={len(chatts_val_list)}, test={len(chatts_test_list)}")
        print(f"  chronos — train={len(chronos_train_list)}, val={len(chronos_val_list)}, test={len(chronos_test_list)}")

        return train, val, test

    def _get_answer(self, row) -> str:
        if row["type"] == "chatts":
            return row["output"]
        # Forecasting: future patch values are injected as continuous embeddings by
        # compute_phase2_loss, not as text floats, so only the text prefix is returned.
        return "Answer: "

    def _get_pre_prompt(self, row) -> str:
        if row["type"] == "chatts":
            return row["input"]
        return f"Given {row['n_history_patches']} time series patches"

    def _get_post_prompt(self, row) -> str:
        if row["type"] == "chatts":
            return "Answer:"
        return f"</ts>, please predict the next {row['n_future_patches']} patches"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # ---- Forecasting sample ----
        if row["type"] == "forecasting":
            history = torch.as_tensor(row["history"], dtype=torch.float32)
            return [TextTimeSeriesPrompt("<ts>", history.numpy())]

        # ---- ChatTS sample (original behaviour) ----
        ts = row["timeseries"]
        if isinstance(ts, str):
            ts = ast.literal_eval(ts)

        series = torch.tensor(ts, dtype=torch.float32)

        if series.shape[0] != 1:
            prompts = []
            assert series.shape[1] > series.shape[0], (
                f"Expected more than 1 time series, but got shape {series.shape}"
            )
            for i in range(series.shape[0]):
                mean = series[i].mean().item()
                std = series[i].std().item()
                normalized = (series[i] - mean) / (std + 1e-8)
                prompt_text = (
                    f"This is time series {i}, it has mean {mean:.4f} and std {std:.4f}"
                )
                prompts.append(TextTimeSeriesPrompt(prompt_text, normalized.numpy()))
            return prompts
        else:
            series = series.squeeze(0)
            mean = series.mean().item()
            std = series.std().item()
            normalized = (series - mean) / (std + 1e-8)
            prompt_text = (
                f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            )
            return [TextTimeSeriesPrompt(prompt_text, normalized.numpy())]


if __name__ == "__main__":
    import textwrap

    EOS = "</s>"
    ds_train = Stage2AllDataset("train", EOS_TOKEN=EOS)
    ds_val   = Stage2AllDataset("validation", EOS_TOKEN=EOS)
    print(f"Train samples: {len(ds_train)}")
    print(f"Val   samples: {len(ds_val)}\n")
    ds = ds_train

    # Find and print one example of each type
    found = {"chatts": None, "forecasting": None}
    for i in range(len(ds)):
        sample = ds[i]
        # Recover the source type from the pre_prompt heuristic
        pre = sample["pre_prompt"]
        kind = "forecasting" if pre.startswith("Given ") else "chatts"
        if found[kind] is None:
            found[kind] = (i, sample)
        if all(v is not None for v in found.values()):
            break

    SEP = "=" * 70

    # for kind, (idx, sample) in found.items():
    #     print(SEP)
    #     print(f"  [{kind.upper()}]  index={idx}")
    #     print(SEP)

    #     print(f"pre_prompt  : {sample['pre_prompt']}")
    #     print(f"post_prompt : {sample['post_prompt']}")

    #     for j, (ts_text, ts_arr) in enumerate(
    #         zip(sample["time_series_text"], sample["time_series"])
    #     ):
    #         ts_preview = ts_arr[:8].tolist() if hasattr(ts_arr, "tolist") else list(ts_arr)[:8]
    #         print(f"ts[{j}] text  : {ts_text}")
    #         print(f"ts[{j}] shape : {len(ts_arr)} values  |  first 8: {[f'{v:.3f}' for v in ts_preview]}")

    #     answer = sample["answer"]
    #     answer_preview = answer
    #     print(f"answer      : {answer_preview}")
    #     print()
