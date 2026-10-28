# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple 
from datasets import Dataset, load_dataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate
)
from torch.utils.data import DataLoader
import os
import torch
import ast


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
        import torch.distributed as dist

        # In distributed training, let rank 0 download/cache first to avoid
        # concurrent filelock failures on NFS filesystems (errno 37 ENOLCK).
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        from opentslm.time_series_datasets.constants import get_chatts_dir
        _path = os.path.join(get_chatts_dir(), "process", "align_256")
        if rank == 0:
            ds_full = load_dataset(_path, split="train", trust_remote_code=True)
        if is_distributed:
            dist.barrier()
        if rank != 0:
            ds_full = load_dataset(_path, split="train", trust_remote_code=True)

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

        if series.shape[0] != 1:
            prompts = []
            assert series.shape[1] > series.shape[0], f"Expected more than 1 time series, but got shape {series.shape}"

            for i in range(series.shape[0]):
                mean = series[i].mean(dim=0, keepdim=True).item()
                std = series[i].std(dim=0, keepdim=True).item()
                normalized_series = (series[i] - mean) / (std + 1e-8)
                prompt_text = f"This is time series {i}, it has mean {mean:.4f} and std {std:.4f}"
                prompts.append(TextTimeSeriesPrompt(prompt_text, normalized_series.tolist()))
            return prompts
        else:
            if series.shape[0] == 1:
                series = series.squeeze(0)  # shape: [L]
            mean = series.mean(dim=0, keepdim=True).item()
            std = series.std(dim=0, keepdim=True).item()
            normalized_series = (series - mean) / (std + 1e-8)
            prompt_text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            return [TextTimeSeriesPrompt(prompt_text, normalized_series.tolist())]

