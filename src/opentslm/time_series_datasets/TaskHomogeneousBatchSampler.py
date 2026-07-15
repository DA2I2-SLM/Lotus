# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

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
        max_tokens_per_batch: Optional[int] = None,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        task_key_fn: Optional[Callable[[Any], str]] = None,
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
        self.max_tokens_per_batch = int(max_tokens_per_batch) if max_tokens_per_batch is not None else None
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        # Optional custom row→task-key function. The default groups by the
        # row's ``type`` field, preserving existing behaviour. Callers that
        # want finer-grained groupings (e.g. splitting chronos by source into
        # chronos_uv / synthetic) pass their own callable here.
        self.task_key_fn: Callable[[Any], str] = (
            task_key_fn
            if task_key_fn is not None
            else (lambda r: r.get("type", "unknown") if isinstance(r, dict) else "unknown")
        )

        self._indices_by_task = self._build_task_index()
        self._batches: List[List[int]] = []
        self._refresh_batches()

    def _build_task_index(self) -> Dict[str, List[int]]:
        task_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.dataset.rows):
            task = self.task_key_fn(row)
            if task is None:
                task = "unknown"
            task_to_indices[str(task)].append(idx)
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

    def _get_seq_len(self, row) -> int:
        if isinstance(row, dict):
            return int(row.get("_seq_len", self.batch_size * 32))
        return self.batch_size * 32

    def _token_budget_batches(
        self,
        local_indices: List[int],
        target_n: int,
        rng: random.Random,
    ) -> List[List[int]]:
        """Pack local_indices into variable-size batches respecting max_tokens_per_batch.

        Sorts by estimated sequence length, packs greedily so that
        len(batch) * max_len_in_batch <= max_tokens_per_batch and
        len(batch) <= batch_size. A single sample that alone exceeds the budget
        still forms a B=1 batch (cannot be split); the existing OOM handler
        catches it if it truly overflows GPU memory.

        Returns exactly min(len(packed), target_n) batches after shuffling,
        so all DDP ranks produce the same batch count for this task.
        """
        rows = self.dataset.rows
        lengths = [(idx, self._get_seq_len(rows[idx])) for idx in local_indices]
        lengths.sort(key=lambda x: x[1])  # shorter-first for greedy packing

        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for idx, est_len in lengths:
            new_max = max(cur_max, est_len)
            if cur and (
                (len(cur) + 1) * new_max > self.max_tokens_per_batch
                or len(cur) >= self.batch_size
            ):
                batches.append(cur)
                cur = [idx]
                cur_max = est_len
            else:
                cur.append(idx)
                cur_max = new_max
        if cur:
            batches.append(cur)

        rng.shuffle(batches)  # shuffle before trim so no length-bias in the kept set
        return batches[:target_n]

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

            if self.max_tokens_per_batch is not None:
                target_n = len(local_indices) // self.batch_size
                local_batches.extend(
                    self._token_budget_batches(local_indices, target_n, rng)
                )
            else:
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


class TaskWeightedDistributedBatchSampler(TaskHomogeneousDistributedBatchSampler):
    """Task-homogeneous sampler with explicit per-task sampling weights.

    Each task is assigned a weight; the number of batches drawn from each task
    is proportional to its weight rather than its dataset size.  Smaller tasks
    are oversampled (their indices are repeated) to match the target count;
    larger tasks are undersampled (indices are subsampled without replacement).

    The full dataset is never modified — only the sampling frequency changes.

    Example (Transfusion-style 50/50 split):
        task_weights = {"chronos": 1, "tsqa": 1, "m4": 1}
        → each task contributes equal batches per epoch regardless of size.

    Args:
        task_weights: dict mapping task type string to a positive float weight.
            Tasks not listed default to weight 1.0.
            Weights are normalised internally so only ratios matter.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        task_weights: Optional[Dict[str, float]] = None,
        max_tokens_per_batch: Optional[int] = None,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        task_key_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self._task_weights: Dict[str, float] = task_weights or {}
        super().__init__(
            dataset,
            batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            task_key_fn=task_key_fn,
        )

    def _refresh_batches(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        # Determine target batch count per task.
        # Base: largest task's natural batch count (so the largest task is
        # never oversampled, only smaller tasks are brought up to match).
        task_natural = {}  # natural local batch count without weighting
        for task, indices in self._indices_by_task.items():
            adjusted = self._adjust_global_indices(list(indices))
            local = adjusted[self.rank : len(adjusted) : self.num_replicas]
            task_natural[task] = len(local) // self.batch_size

        if not task_natural:
            self._batches = []
            return

        # Normalise weights; default missing tasks to 1.0
        weights = {t: self._task_weights.get(t, 1.0) for t in task_natural}
        total_w = sum(weights.values())
        norm_w  = {t: w / total_w for t, w in weights.items()}

        # Target: each task's share of total batches proportional to its weight.
        # Total pool = sum of natural counts (keeps epoch length reasonable).
        # Use max(natural, 1) so tiny tasks (e.g. 5 chronos_uv rows that get
        # trimmed to 0 by drop_last across 8 ranks) still contribute one batch
        # to the natural total — otherwise their weight gets normalised away
        # and they're effectively dropped from training.
        total_natural = sum(max(n, 1) for n in task_natural.values())
        target_batches = {t: max(1, round(norm_w[t] * total_natural)) for t in task_natural}

        local_batches: List[List[int]] = []
        for task in sorted(self._indices_by_task.keys()):
            task_indices = list(self._indices_by_task[task])
            if self.shuffle:
                rng.shuffle(task_indices)

            adjusted   = self._adjust_global_indices(task_indices)
            local_all  = adjusted[self.rank : len(adjusted) : self.num_replicas]
            n_target   = target_batches[task] * self.batch_size

            if len(local_all) == 0:
                # Tiny task — distributed sharding (drop_last × global_unit)
                # erased it. Fall back to cross-rank replication of the raw
                # (already shuffled) indices: every rank draws oversampled
                # copies from the same small pool. Different ranks see
                # different random anchors per __getitem__ for chronos stubs,
                # so the gradient is still varied across ranks, just drawn
                # from the same series pool.
                if not task_indices:
                    continue
                local_all = task_indices

            # Repeat or truncate to hit the target count exactly.
            if len(local_all) < n_target:
                # Oversample: tile and trim
                repeats = math.ceil(n_target / len(local_all))
                local_indices = (local_all * repeats)[:n_target]
                if self.shuffle:
                    rng.shuffle(local_indices)
            else:
                # Undersample: take first n_target (already shuffled)
                local_indices = local_all[:n_target]

            if self.max_tokens_per_batch is not None:
                local_batches.extend(
                    self._token_budget_batches(local_indices, target_batches[task], rng)
                )
            else:
                # Trim to batch boundary
                trim = (len(local_indices) // self.batch_size) * self.batch_size
                local_batches.extend(
                    local_indices[i : i + self.batch_size]
                    for i in range(0, len(local_indices[:trim]), self.batch_size)
                )

        if self.shuffle:
            rng.shuffle(local_batches)

        self._batches = local_batches
