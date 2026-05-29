# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
import random
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset, load_from_disk, Sequence, concatenate_datasets, Value
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.PretrainDataset import PretrainDataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import os

TEST_FRAC = 0.1
VAL_FRAC = 0.1

# Chronos uses context_length=512 (T5 default).
# Training sequences are constructed by slicing random windows of this size.
MAX_SEQ_LEN = 512

# Maps dataset name -> list of target column keys to use as training data
CHRONOS_DATASET_REGISTRY: Dict[str, List[str]] = {
    # Trimmed to a single clean dataset to isolate the "model can't fit" bug.
    # Other registry entries are too noisy / heterogeneous for sanity testing.
    # "dominick": ["target"],
    # "monash_fred_md": ["target"],
    # "electricity_15min": ["consumption_kW"],
    # "ercot": ["target"],
    # "exchange_rate": ["target"],
    # "ushcn_daily": ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'],
    # "weatherbench_daily": ['target'],
    # "weatherbench_weekly": ['target'],
    # "wiki_daily_100k": ['target'],
    "monash_australian_electricity": ['target'],

}

BASE_DATA_PATH = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets"

# Directory on shared filesystem where preprocessed splits are saved once
# by global rank 0 and loaded by all other ranks.
# Must be writable by the user — BASE_DATA_PATH may be read-only.
# Overridable via CHRONOS_PREPROCESSED_SPLITS_DIR so concurrent runs that
# would otherwise race on writes can point at separate cache directories.
PREPROCESSED_SPLITS_DIR = os.environ.get(
    "CHRONOS_PREPROCESSED_SPLITS_DIR",
    "_preprocessed_chronos_splits_mv",
)


def _last_value_impute(series: torch.Tensor) -> torch.Tensor:
    """
    Forward-fill NaN/Inf values with the last valid observation (per channel).
    Matches Chronos's ``LastValueImputation`` strategy for causal models.
    Any leading NaNs (no prior valid value) are filled with 0.

    Vectorised: cummax of (valid * 1-based-index) gives, at each position, the
    most recent valid index + 1 (0 if none). Subtracting 1 yields a gather
    index where -1 marks "no prior valid value", which we then zero out.
    """
    bad = torch.isnan(series) | torch.isinf(series)
    if not bad.any():
        return series

    if series.ndim == 1:
        T = series.size(0)
        valid = (~bad).to(torch.long)
        idx = torch.cummax(valid * (torch.arange(T, device=series.device) + 1), dim=0).values - 1
        leading = idx < 0
        gathered = series[idx.clamp(min=0)]
    else:
        T = series.size(0)
        valid = (~bad).to(torch.long)
        ar = (torch.arange(T, device=series.device) + 1).unsqueeze(-1)
        idx = torch.cummax(valid * ar, dim=0).values - 1
        leading = idx < 0
        gathered = torch.gather(series, dim=0, index=idx.clamp(min=0))

    out = torch.where(bad, gathered, series)
    return torch.where(leading, torch.zeros_like(out), out)


def _random_window_crop_and_normalize(
    series: torch.Tensor,
    max_len: int = MAX_SEQ_LEN,
    mode: str = "train",
) -> Tuple[torch.Tensor, float, float]:
    """
    Chronos-style windowing: slice a random contiguous window of length
    ``max_len`` from the series, impute missing values via forward-fill,
    then z-normalize using *window-local* stats.

    Parameters
    ----------
    series : torch.Tensor
        1-D or 2-D (time, channels) tensor.
    max_len : int
        Maximum window size to extract.
    mode : str
        "train" -> random start index (data diversity across epochs).
        "validation" / "test" -> deterministic, takes the last ``max_len`` steps.

    Returns
    -------
    window : torch.Tensor
        Cropped, imputed, and normalized tensor.
    mean_val : float
        Window mean (first channel, before normalization).
    std_val : float
        Window std  (first channel, before normalization).
    """
    T = series.shape[0]

    if T <= max_len:
        window = series
    elif mode == "train":
        start = random.randint(0, T - max_len)
        window = series[start : start + max_len]
    else:
        window = series[-max_len:]

    # Chronos-style last-value imputation for missing data
    window = _last_value_impute(window)

    # Window-local normalization (per-channel for multivariate)
    means = window.mean(dim=0, keepdim=True)
    stds = window.std(dim=0, keepdim=True).clamp(min=1e-4)
    window = (window - means) / stds

    mean_val = means.flatten()[0].item()
    std_val = stds.flatten()[0].item()
    return window, mean_val, std_val


class ChronosDataset(PretrainDataset):
    """
    A dataset wrapper for pre-training, using Chronos trained datasets.

    Applies Chronos-style random window cropping: during training, a random
    contiguous window of ``max_seq_len`` steps is sampled from each series.
    During validation/test the last ``max_seq_len`` steps are used.

    Parameters
    ----------
    mode : str
        One of "train", "validation", or "test".
    dataset_name : str
        Key into CHRONOS_DATASET_REGISTRY selecting which dataset to load,
        or "all" to concatenate every registered dataset.
    base_path : str, optional
        Override the base directory for dataset files.
    max_seq_len : int, optional
        Context window size (default 512, matching Chronos T5 default).
    """

    def __init__(
        self,
        mode: str,
        EOS_TOKEN: str,
        dataset_name: str = "all",
        base_path: str = BASE_DATA_PATH,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        if dataset_name != "all" and dataset_name not in CHRONOS_DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available datasets: {list(CHRONOS_DATASET_REGISTRY.keys())} or 'all'."
            )
        self.dataset_name = dataset_name
        # When loading all datasets, columns are normalized to "target"
        self.target_keys = ["target"] if dataset_name == "all" else CHRONOS_DATASET_REGISTRY[dataset_name]
        self.base_path = base_path
        self.max_seq_len = max_seq_len
        self.mode = mode
        super().__init__(mode, EOS_TOKEN)

    def _normalize_to_target(self, ds: Dataset, target_keys: List[str]) -> Dataset:
        """Normalize to a single 'target' column for concatenation.

        Univariate datasets: rename the column.
        Multivariate datasets: expand each key into a separate row so that
        channel-independent processing (one [T] series per prompt) is preserved.
        """
        if len(target_keys) == 1:
            key = target_keys[0]
            if key != "target":
                ds = ds.rename_column(key, "target")
            ds = ds.select_columns(["target"])
        else:
            # Expand: each target_key becomes its own row
            keys = target_keys  # capture for lambda
            ds = ds.map(
                lambda batch: {"target": [seq for key in keys for seq in batch[key]]},
                batched=True,
                remove_columns=ds.column_names,
            )
        ds = ds.cast_column("target", Sequence(feature=Value(dtype="float32")))
        return ds

    def _load_all_datasets(self) -> Dataset:
        """Load and concatenate all registered datasets, normalizing columns.

        Called only on global rank 0. Each dataset is loaded, normalized,
        saved to disk, and freed before the next one is loaded — only one
        dataset occupies RAM at a time, preventing CPU OOM on large corpora.
        """
        parts_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "_parts")
        os.makedirs(parts_dir, exist_ok=True)

        part_paths = []
        for name, keys in CHRONOS_DATASET_REGISTRY.items():
            part_path = os.path.join(parts_dir, name)
            try:
                if not os.path.exists(part_path):
                    ds = load_dataset(
                        f"{self.base_path}/{name}",
                        split="train",
                        trust_remote_code=True,
                        keep_in_memory=True,  # preprocess in RAM, no mmap
                    )
                    ds = self._normalize_to_target(ds, keys)
                    print(f"  -> Loaded '{name}': {len(ds)} series — saving to disk...")
                    ds.save_to_disk(part_path)
                    del ds  # free RAM before loading the next dataset
                else:
                    print(f"  -> Cached part '{name}' found at {part_path}, skipping.")
                part_paths.append(part_path)
            except Exception as e:
                print(f"  -> WARNING: Failed to load '{name}': {e}")

        if not part_paths:
            raise RuntimeError("No datasets were successfully loaded!")

        parts = [load_from_disk(p, keep_in_memory=False) for p in part_paths]
        ds_full = concatenate_datasets(parts)
        print(f"  -> Concatenated total: {len(ds_full)} series")
        return ds_full

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        rank = int(os.environ.get("RANK", 0))

        # Path where the preprocessed splits live on shared storage.
        # Written once by global rank 0; read by all ranks.
        split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, self.dataset_name)
        done_flag = os.path.join(split_dir, "_DONE")

        # ── Global rank 0: load, preprocess in RAM, split, save once ──
        if rank == 0:
            # Check if splits were already prepared (e.g. from a previous run)
            if os.path.exists(done_flag):
                print(f"[Rank 0] Preprocessed splits already exist at {split_dir}, skipping.")
            else:
                print(f"[Rank 0] Loading dataset '{self.dataset_name}' (in-memory)...")
                if self.dataset_name == "all":
                    ds_full = self._load_all_datasets()
                else:
                    ds_full = load_dataset(
                        f"{self.base_path}/{self.dataset_name}",
                        split="train",
                        trust_remote_code=True,
                        keep_in_memory=True,  # preprocess in RAM
                    )
                    print(f"  -> Loaded '{self.dataset_name}': {len(ds_full)} series")

                # Deterministic splits (same seed → reproducible)
                train_val, test = ds_full.train_test_split(
                    test_size=TEST_FRAC, seed=42
                ).values()
                train, val = train_val.train_test_split(
                    test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
                ).values()

                # Save clean Arrow files to shared storage
                os.makedirs(split_dir, exist_ok=True)
                train.save_to_disk(os.path.join(split_dir, "train"))
                val.save_to_disk(os.path.join(split_dir, "val"))
                test.save_to_disk(os.path.join(split_dir, "test"))

                # Write a marker so future runs skip preprocessing
                with open(done_flag, "w") as f:
                    f.write("done\n")

                # Free RAM
                del ds_full, train_val, train, val, test
                print(f"[Rank 0] Saved splits to {split_dir}")

        # ── Barrier: all ranks wait for rank 0 to finish writing ──
        if dist.is_initialized():
            dist.barrier()

        # keep_in_memory=True: Arrow mmaps are not fork-safe with DataLoader workers.
        train = load_from_disk(os.path.join(split_dir, "train"), keep_in_memory=True)
        val = load_from_disk(os.path.join(split_dir, "val"), keep_in_memory=True)
        test = load_from_disk(os.path.join(split_dir, "test"), keep_in_memory=True)

        print(
            f"[Rank {rank}] Dataset '{self.dataset_name}': "
            f"{len(train)} train, {len(val)} val, {len(test)} test"
        )

        return train, val, test

    def _get_raw_ts(self, row):
        """Return the time series data by reading all target keys from the row."""
        if len(self.target_keys) == 1:
            return row[self.target_keys[0]]
        # For multiple target keys, stack them into a single tensor
        return [row[key] for key in self.target_keys]

    def _get_answer(self, row) -> str:
        # Language modelling: no explicit answer, just terminate with EOS
        return self.EOS_TOKEN

    def _get_pre_prompt(self, row) -> str:
        return ""

    def _get_post_prompt(self, row) -> str:
        return self.EOS_TOKEN

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        if len(self.target_keys) == 1:
            series = torch.tensor(row[self.target_keys[0]], dtype=torch.float32)
        else:
            series = torch.stack(
                [torch.tensor(row[key], dtype=torch.float32) for key in self.target_keys],
                dim=1,
            )

        window, mean_val, std_val = _random_window_crop_and_normalize(
            series, max_len=self.max_seq_len, mode=self.mode,
        )

        # Channel-independence: one prompt per channel (Chronos/PatchTST style).
        # Prompt text mirrors the shape descriptors used in Phase1JointDataset's
        # chronos branch so the model sees the same channel/timestep language
        # at train and eval time.
        if window.ndim == 2 and window.size(1) > 1:
            n_channels = window.size(1)
            n_timesteps = window.size(0)
            prompts = []
            for ch in range(n_channels):
                ch_data = window[:, ch]  # [T] — already normalized per-channel
                prompts.append(TextTimeSeriesPrompt(
                    f"This is channel {ch} of {n_channels} of a multivariate "
                    f"time series with {n_timesteps} time steps. <ts>",
                    ch_data.tolist(),
                ))
            return prompts

        # Univariate — single prompt
        n_timesteps = window.shape[0]
        return [TextTimeSeriesPrompt(
            f"This is a univariate time series with 1 channel "
            f"and {n_timesteps} time steps. "
            f"It has mean {mean_val:.4f} and std {std_val:.4f}. <ts>",
            window.tolist(),
        )]

    def _format_sample(self, row):
        d = super()._format_sample(row)
        if d.get("time_series"):
            # _random_window_crop_and_normalize already returns normalized inputs.
            d["ts_already_normalized"] = True
        return d


if __name__ == "__main__":
    # Example: iterate over all registered datasets
    for name in CHRONOS_DATASET_REGISTRY:
        print(f"Dataset: {name}, target keys: {CHRONOS_DATASET_REGISTRY[name]}")

    # Load a specific dataset
    dataset_name = "solar"
    train = ChronosDataset("train", "", dataset_name=dataset_name)
    val = ChronosDataset("validation", "", dataset_name=dataset_name)
    test = ChronosDataset("test", "", dataset_name=dataset_name)

    dataloader = DataLoader(
        train,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )
    batch = next(iter(dataloader))
    print("batch: ", len(batch))
    print("Batch keys:", batch[0].keys())
    for key, items in batch[0].items():
        if isinstance(items, torch.Tensor):
            print(f"  {key}: Tensor of shape {items.shape}")
        else:
            print(f"  {key}: {items}")
