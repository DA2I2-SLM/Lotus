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


class ChatTSDataset(QADataset):
    """
    Manually processed for ChatTSDataset
    """
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # ds_full = load_dataset("ChatTSRepo/ChatTS-Training-Dataset", "sft", split="train")
        path = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/ERA5/ERA5-Large/pretrain.npy"
        ds_full = np.load()


        train_val, test = ds_full.train_test_split(test_size=TEST_FRAC, seed=42).values()
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        return train, val, test

    def _get_answer(self, row) -> str:
        return row["output"]
    
    def _get_pre_prompt(self, row) -> str:
        return row["input"]
    
    @property
    def column_names(self):
        return ["input", "timeseries", "output"]
    
    def _get_text_time_series_prompt_list(self, row):
        try:
            ts = row["timeseries"]

            series = torch.tensor(ts, dtype=torch.float32)  # now safe

            # ragged check: all channels must have same length
            lens = [len(ch) for ch in ts if isinstance(ch, list)]
            if len(lens) != len(ts) or len(set(lens)) != 1:
                return [TextTimeSeriesPromptMultiVariate(
                    f"This is a multivariate time series with 4 channels.",
                    [1,2,3,4]
                )]

            means = series.mean(dim=1, keepdim=True)  # per-channel
            stds  = series.std(dim=1, keepdim=True)
            series = (series - means) / (stds + 1e-8)

            return [TextTimeSeriesPromptMultiVariate(
                f"This is a multivariate time series with {series.shape[0]} channels.",
                series.tolist()
            )]
        except Exception as e:
            print(e)
            return [TextTimeSeriesPromptMultiVariate(
                f"This is a multivariate time series with 4 channels.",
                [1,2,3,4]
            )]

    def _get_post_prompt(self, row) -> str:
        # NOTE: This one is totally managable
        if "Task" not in row.keys():
            return ""
        return "Predict the " + row["Task"] + " Answer:"






if __name__ == "__main__":
    train = ChatTSDataset("train", "")
    val = ChatTSDataset("validation", "")
    test = ChatTSDataset("test", "")

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
