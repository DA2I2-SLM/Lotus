# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
from typing import List, Tuple


from datasets import Dataset, load_dataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
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


class TSQADataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        import torch.distributed as dist

        # In distributed training, let rank 0 download/cache first to avoid
        # concurrent filelock failures on NFS filesystems (errno 37 ENOLCK).
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if rank == 0:
            ds_full = load_dataset("ChengsenWang/TSQA", split="train")
        if is_distributed:
            dist.barrier()
        if rank != 0:
            ds_full = load_dataset("ChengsenWang/TSQA", split="train")

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
        return row["Answer"]

    def _get_pre_prompt(self, row) -> str:
        return row["Question"]

    def _get_post_prompt(self, row) -> str:
        # return "Answer:"
        return "Predict the " + row["Task"] + " Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(json.loads(row["Series"]), dtype=torch.float32)

        means = series.mean(dim=0, keepdim=True)  # shape: (n_series, 1)
        stds = series.std(dim=0, keepdim=True)  # shape: (n_series, 1)
        series = (series - means) / (stds + 1e-8)  # broadcasts to (n_series, length)
        # TSQA has always only one time series
        # Make tensor indexing more robust
        mean_val = means.flatten()[0].item()
        std_val = stds.flatten()[0].item()

        # Prompt mirrors Phase1JointDataset tsqa branch — must stay aligned so
        # the model sees the same shape descriptors at train and eval time.
        n_channels = series.shape[1] if series.dim() == 2 else 1
        n_timesteps = series.shape[0]
        modality = "multivariate" if n_channels > 1 else "univariate"
        ch_word = "channels" if n_channels > 1 else "channel"
        return [TextTimeSeriesPrompt(
            f"This is a {modality} time series with {n_channels} {ch_word} "
            f"and {n_timesteps} time steps. "
            f"It has mean {mean_val:.4f} and std {std_val:.4f}.",
            series.tolist(),
        )]


if __name__ == "__main__":
    train = TSQADataset("train", "")
    val = TSQADataset("validation", "")
    test = TSQADataset("test", "")

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
    # from collections import Counter

    # train_values = [
    #     (el[0], el[1] / len(train))
    #     for el in Counter(map(lambda x: x["post_prompt"], train)).items()
    # ]
    # print("train", train_values)
    # val_values = [
    #     (el[0], el[1] / len(val))
    #     for el in Counter(map(lambda x: x["post_prompt"], val)).items()
    # ]
    # print("val", val_values)
    # test_values = [
    #     (el[0], el[1] / len(test))
    #     for el in Counter(map(lambda x: x["post_prompt"], test)).items()
    # ]
    # print("test", test_values)
