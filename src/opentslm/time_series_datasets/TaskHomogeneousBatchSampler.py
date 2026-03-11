# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List

from torch.utils.data import Sampler


class TaskHomogeneousDistributedBatchSampler(Sampler[List[int]]):
    """Yield local batches where every batch contains exactly one task type.

    Expected dataset contract:
    - ``dataset.rows`` exists (as used by PretrainDataset) and each row is a dict
      with a ``"type"`` key (e.g., ``"chronos"``, ``"tsqa"``, ``"m4"``).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas - 1}], got {rank}"
            )
        if not hasattr(dataset, "rows"):
            raise ValueError(
                "TaskHomogeneousDistributedBatchSampler expects dataset.rows "
                "to exist for task-type indexing."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self._indices_by_task = self._build_task_index()
        self._batches: List[List[int]] = []
        self._refresh_batches()

    def _build_task_index(self) -> Dict[str, List[int]]:
        task_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.dataset.rows):
            task = row.get("type", "unknown") if isinstance(row, dict) else "unknown"
            task_to_indices[task].append(idx)
        return dict(task_to_indices)

    def _adjust_global_indices(self, indices: List[int]) -> List[int]:
        """Pad/trim so count is divisible by batch_size * num_replicas."""
        if not indices:
            return []

        global_unit = self.batch_size * self.num_replicas
        n = len(indices)

        if self.drop_last:
            keep = (n // global_unit) * global_unit
            return indices[:keep]

        total = int(math.ceil(n / global_unit) * global_unit)
        if total == n:
            return indices
        return indices + indices[: total - n]

    def _refresh_batches(self) -> None:
        rng = random.Random(self.seed + self.epoch)
        local_batches: List[List[int]] = []

        for task in sorted(self._indices_by_task.keys()):
            task_indices = list(self._indices_by_task[task])
            if self.shuffle:
                rng.shuffle(task_indices)

            adjusted = self._adjust_global_indices(task_indices)
            if not adjusted:
                continue

            local_indices = adjusted[self.rank : len(adjusted) : self.num_replicas]
            if len(local_indices) < self.batch_size:
                continue
            if len(local_indices) % self.batch_size != 0:
                local_indices = local_indices[
                    : (len(local_indices) // self.batch_size) * self.batch_size
                ]

            local_batches.extend(
                local_indices[i : i + self.batch_size]
                for i in range(0, len(local_indices), self.batch_size)
            )

        if self.shuffle:
            rng.shuffle(local_batches)

        self._batches = local_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._refresh_batches()

    def __iter__(self) -> Iterable[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
