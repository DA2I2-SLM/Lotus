# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
from typing import List, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from opentslm.prompt.text_time_series_prompt_multivariate import TextTimeSeriesPromptMultiVariate
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader
import torch


TEST_FRAC = 0.1
VAL_FRAC = 0.1


def get_value_count(key: str, dataset: Dataset):
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


class ERA5Dataset(QADataset):
    """
    Multivariate ERA5 from a (T, C) numpy array.
    Each dataset item is a window across all channels, patched along time.
    """

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        path = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/ERA5/ERA5-Large/pretrain.npy"

        # Use mmap to avoid loading the whole file into RAM
        full = np.load(path, mmap_mode="r")
        
        # only take 12 channels
        full = full[:, :12]

        # Ensure (T, C) = (time, channels)
        # If your file is (C, T), flip it.
        if full.shape[0] < full.shape[1]:
            full = full.T

        T, C = full.shape  # e.g. (116880, 4920)


        prompt = "This is a timeseries."

        # --------- choose these to match your model setup ----------
        CONTEXT_LEN = 512   # history length (timesteps)
        PRED_LEN    = 96     # future length (timesteps). Set 0 if you do not need numeric targets.
        PATCH_SIZE  = 1     # patch length (timesteps)
        STRIDE      = 256     # sliding stride (timesteps)
        # ----------------------------------------------------------

        if CONTEXT_LEN % PATCH_SIZE != 0:
            raise ValueError("CONTEXT_LEN must be divisible by PATCH_SIZE")

        max_start = T - (CONTEXT_LEN + PRED_LEN)
        if max_start < 0:
            raise ValueError(f"T={T} is smaller than CONTEXT_LEN+PRED_LEN={CONTEXT_LEN+PRED_LEN}")

        # 1) HF dataset holds only indices so train_test_split works
        starts = np.arange(0, max_start + 1, STRIDE, dtype=np.int64)
        ds_idx = Dataset.from_dict({"start": starts})

        # 2) split indices
        train_val, test = ds_idx.train_test_split(test_size=TEST_FRAC, seed=42).values()
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        # 3) map index -> real example (multivariate + patched)
        def transform(example):
            starts = example["start"]
            # Handle both scalar and list starts
            if isinstance(starts, (int, np.integer)):
                starts = [int(starts)]

            xs, ins, outs = [], [], []
            for s in starts:
                s = int(s)
                x = np.asarray(full[s : s + CONTEXT_LEN, :], dtype=np.float32)  # (T_window, C)
                xs.append(x)
                ins.append(prompt)
                outs.append("")

            return {"input": ins, "timeseries": xs, "output": outs}

        train = train.with_transform(transform)
        val   = val.with_transform(transform)
        test  = test.with_transform(transform)

        return train, val, test

    def _get_answer(self, row) -> str:
        return row["output"]
    
    def _get_pre_prompt(self, row) -> str:
        return row["input"]
    
    @property
    def column_names(self):
        return ["input", "timeseries", "output"]
    
    def _get_text_time_series_prompt_list(self, row):
        ts = row["timeseries"]

        series = torch.tensor(ts, dtype=torch.float32)  # now safe

        means = series.mean(dim=1, keepdim=True)  # per-channel
        stds  = series.std(dim=1, keepdim=True)
        series = (series - means) / (stds + 1e-8)

        return [TextTimeSeriesPromptMultiVariate(
            f"This is a multivariate time series with {series.shape[0]} channels.",
            series.tolist()
        )]

    def _get_post_prompt(self, row) -> str:
        # NOTE: This one is totally managable
        if "Task" not in row.keys():
            return ""
        return "Predict the " + row["Task"] + " Answer:"






if __name__ == "__main__":
    train = ERA5Dataset("train", "")
    val = ERA5Dataset("validation", "")
    test = ERA5Dataset("test", "")

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
    #         batch, patch_size=4
    #     ),
    # )

    from collections import Counter

    train_values = [
        (el[0], el[1] / len(train))
        for el in Counter(map(lambda x: x["post_prompt"], train)).items()
    ]
    print("train", train_values)
    val_values = [
        (el[0], el[1] / len(val))
        for el in Counter(map(lambda x: x["post_prompt"], val)).items()
    ]
    print("val", val_values)
    test_values = [
        (el[0], el[1] / len(test))
        for el in Counter(map(lambda x: x["post_prompt"], test)).items()
    ]
    print("test", test_values)
