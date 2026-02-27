# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple 
from datasets import Dataset, load_dataset, Sequence
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader
import torch
import ast
from tqdm import tqdm

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
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        # 1) Load the single built‑in "train" split (≈ 7 k rows)
        ds_full = load_dataset("/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS/process/align_256", split="train", trust_remote_code=True)

        # 2) First carve out the test split
        train_val, test = ds_full.train_test_split(
            test_size=TEST_FRAC, seed=42
        ).values()

        # 3) From the remaining data take validation
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()

        return train, val, test

    def _get_answer(self, row) -> str:
        return row["output"]

    def _get_pre_prompt(self, row) -> str:
        return row["input"]

    def _get_post_prompt(self, row) -> str:
        return "Answer:"
        # return "Predict the " + row["Task"] + " Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # context is [C, ctx]
        ts = row["timeseries"]

        if isinstance(ts, str):
            ts = ast.literal_eval(ts)   # -> list / nested list
        
        series = torch.tensor(ts, dtype=torch.float32)


        # 0) If series is 1D: treat as univariate [T]
        if series.dim() == 1:
            mean = series.mean().item()
            std = series.std().item()
            normalized_series = (series - mean) / (std + 1e-8)
            prompt_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            return [TextTimeSeriesPrompt(prompt_text, normalized_series.tolist())]

        # 1) If series is not 2D at this point, fail loudly (unexpected shape)
        if series.dim() != 2:
            raise ValueError(f"Expected series to be 1D or 2D, got shape {tuple(series.shape)}")

        # 2) Handle degenerate 2D univariate cases: [1, T] or [T, 1]
        if series.shape[0] == 1 and series.shape[1] >= 1:
            s = series.squeeze(0)  # [T]
            mean = s.mean().item()
            std = s.std().item()
            normalized_series = (s - mean) / (std + 1e-8)
            prompt_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            return [TextTimeSeriesPrompt(prompt_text, normalized_series.tolist())]

        if series.shape[1] == 1 and series.shape[0] >= 1:
            s = series.squeeze(1)  # [T]
            mean = s.mean().item()
            std = s.std().item()
            normalized_series = (s - mean) / (std + 1e-8)
            prompt_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            return [TextTimeSeriesPrompt(prompt_text, normalized_series.tolist())]

        if series.shape[0] <= series.shape[1]:
            # likely already [C, T]
            series_ct = series
        else:
            # likely [T, C] -> transpose to [C, T]
            series_ct = series.transpose(0, 1)

        prompts = []
        if series_ct.shape[0] > 50:
            print("TOO LARGE C DIM: ", series_ct.shape)
        for i in range(series_ct.shape[0]):
            ch = series_ct[i]  # [T]
            mean = ch.mean().item()
            std = ch.std().item()
            normalized_series = (ch - mean) / (std + 1e-8)
            prompt_text = f"This is time series {i}, it has mean {mean:.4f} and std {std:.4f}"
            prompts.append(TextTimeSeriesPrompt(prompt_text, normalized_series.tolist()))

        return prompts


if __name__ == "__main__":
    train = ChatTSDataset("train", "")
    val = ChatTSDataset("validation", "")
    test = ChatTSDataset("test", "")

    dataloader = DataLoader(
        train,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )
    batch = next(iter(dataloader))
    print('batch: ', len(batch))
    print("Batch keys:", batch[0].keys())
    for key, items in batch[0].items():
        if isinstance(items, torch.Tensor):
            print(f"  {key}: Tensor of shape {items.shape}")
        else:
            print(f"  {key}: {items}")
