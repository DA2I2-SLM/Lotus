import json
from typing import List, Tuple, Literal, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset


LANGTIME_PATH = "/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS/process"

TEST_FRAC = 0.1
VAL_FRAC = 0.1

PROMPT = (
    "This is a multivariate time series dataset. "
    "The following data contains multiple time series variables recorded over time."
)

SERIES_KEYS = [
    "target", "targets",
    "values", "value",
    "series", "time_series",
    "y", "observations",
    "past_values", "future_values",
    "data",  # sometimes nested dict
]


def _extract_series_array(row: dict) -> np.ndarray:
    """
    Return float32 np.ndarray that is either:
      - 1D: [T] (one univariate series)
      - 2D: [T, C] or [C, T] (already multivariate)
    """
    # direct keys + nested dict fallback
    for k in SERIES_KEYS:
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, dict):
            # nested keys
            for kk in SERIES_KEYS:
                if kk in v:
                    v = v[kk]
                    break
        try:
            arr = np.asarray(v, dtype=np.float32)
        except Exception:
            continue
        if arr.ndim in (1, 2) and arr.size > 0:
            return arr

    # heuristic fallback: first list-like numeric field
    for _, v in row.items():
        if isinstance(v, (list, np.ndarray)) and len(v) > 0:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim in (1, 2) and arr.size > 0:
                    return arr
            except Exception:
                pass

    raise KeyError(f"No usable series field. keys={list(row.keys())}")


class LangTimeMultivariateQADataset(QADataset):
    """
    QADataset version of your MemoryEfficientMultivariateDataset.

    It pre-samples a finite pool of windows:
      samples_per_subset = batch_size * batches_per_load

    It fixes datasets that don't have 'target' by using a robust extractor.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        subsets: List[str] = ["process"],
        batch_size: int = 4,
        batches_per_load: int = 1,
        context_length: int = 256,
        prediction_length: int = 16,
        seed: int = 42,
        normalize: bool = True,
        max_variates_per_subset: int = 4,
        answer_format: str = "json",  # "empty" or "json"
    ):
        # store config
        self.subsets = list(subsets)
        self.batch_size = int(batch_size)
        self.batches_per_load = int(batches_per_load)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.total_seq_len = self.context_length + self.prediction_length
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.max_variates_per_subset = int(max_variates_per_subset)
        self.answer_format = answer_format

        # IMPORTANT: QADataset caches at the CLASS level.
        signature = (
            tuple(self.subsets),
            self.batch_size,
            self.batches_per_load,
            self.context_length,
            self.prediction_length,
            self.seed,
            self.normalize,
            self.max_variates_per_subset,
            self.answer_format,
        )
        if getattr(self.__class__, "_signature", None) != signature:
            for attr in ["loaded", "_train_dataset", "_validation_dataset", "_test_dataset"]:
                if hasattr(self.__class__, attr):
                    delattr(self.__class__, attr)
            self.__class__._signature = signature

        super().__init__(split=split, EOS_TOKEN=EOS_TOKEN, format_sample_str=False)

    def _load_single_dataset(self, name: str) -> Optional[np.ndarray]:
        """
        Same role as your original function:
          load one subset and build matrix [T, C] float32.
        """
        try:
            ds = load_dataset(LANGTIME_PATH, name, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"Skipping {name}: cannot load: {e}")
            return None

        if ds.num_rows == 0:
            print(f"Skipping {name}: empty dataset")
            return None

        # infer format from first row
        try:
            a0 = _extract_series_array(ds[0])
        except Exception as e:
            print(f"Skipping {name}: cannot parse first row: {e}. columns={ds.column_names}")
            return None

        # Case A: already multivariate matrix in each row (2D)
        if a0.ndim == 2:
            mat = a0
            # heuristic to ensure [T, C]
            if mat.shape[0] < mat.shape[1] and mat.shape[0] <= 64:
                mat = mat.T
            if mat.shape[0] < self.total_seq_len:
                return None
            return mat.astype(np.float32, copy=False)

        # Case B: each row is one univariate series (1D), stack many rows into [T, C]
        rng = np.random.default_rng(self.seed + (abs(hash(name)) % 10_000))
        idxs = np.arange(ds.num_rows)
        rng.shuffle(idxs)
        idxs = idxs[: self.max_variates_per_subset]

        all_series = []
        for i in idxs:
            try:
                arr = _extract_series_array(ds[int(i)])
            except Exception:
                continue
            if arr.ndim != 1:
                continue
            if len(arr) < self.total_seq_len:
                continue
            all_series.append(arr)

        if not all_series:
            print(f"Skipping {name}: no usable 1D series. columns={ds.column_names}")
            return None

        min_len = min(len(s) for s in all_series)
        if min_len < self.total_seq_len:
            return None

        matrix = np.stack([s[:min_len] for s in all_series], axis=0).T  # [T, C]
        return matrix.astype(np.float32, copy=False)

    def _build_rows(self) -> List[dict]:
        rng = np.random.default_rng(self.seed)
        rows: List[dict] = []
        samples_per_subset = self.batch_size * self.batches_per_load  # mimic your old semantics

        for name in self.subsets:
            matrix = self._load_single_dataset(name)
            if matrix is None:
                continue

            T, C = matrix.shape
            max_start = T - self.total_seq_len
            if max_start < 0:
                continue

            starts = rng.integers(0, max_start + 1, size=samples_per_subset)
            for s in starts:
                window = matrix[int(s) : int(s) + self.total_seq_len, :]  # [total, C]
                ctx = window[: self.context_length, :].T                  # [C, ctx]
                fut = window[self.context_length :, :].T                  # [C, pred]

                rows.append(
                    {
                        "dataset_name": name,
                        "num_variates": int(C),
                        "context": ctx.tolist(),
                        "future": fut.tolist(),
                    }
                )

        if not rows:
            raise RuntimeError("No rows were built. Check subsets, schema, and seq lengths.")
        return rows

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        ds_full = Dataset.from_list(self._build_rows())

        train_val, test = ds_full.train_test_split(test_size=TEST_FRAC, seed=self.seed).values()
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=self.seed
        ).values()

        return train, val, test

    def _get_pre_prompt(self, row) -> str:
        return PROMPT

    def _get_post_prompt(self, row) -> str:
        return f"Predict the next {self.prediction_length} steps for each variable. Put the values in lists. Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        # context is [C, ctx]
        series = torch.tensor(row["context"], dtype=torch.float32)

        if self.normalize:
            means = series.mean(dim=1, keepdim=True)  # [C, 1]
            stds = series.std(dim=1, keepdim=True)    # [C, 1]
            series = (series - means) / (stds + 1e-8)

            text = (
                f"This is the multivariate context window with {series.shape[0]} variables. "
                f"Avg mean {means.mean().item():.4f}, avg std {stds.mean().item():.4f}."
            )
        else:
            text = f"This is the multivariate context window with {series.shape[0]} variables."
        return [TextTimeSeriesPrompt(text, series.tolist())]

    def _get_answer(self, row) -> str:
        if self.answer_format == "empty":
            return ""  # QADataset will append EOS_TOKEN
        if self.answer_format == "json":
            fut = np.asarray(row["future"], dtype=np.float32)
            print("shape of fut: ", fut.shape)
            return json.dumps(np.round(fut, 4).tolist())
        raise ValueError(f"Unknown answer_format={self.answer_format}")
        