

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Literal

import ast
import torch
from datasets import Dataset, load_dataset  # type: ignore
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.prompt.text_time_series_prompt_language_modeling import LMPrompt
from opentslm.time_series_datasets.QADataset import QADataset

from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.LMDataset import LMDataset
import sys
sys.path.append("/home/s223540177/FoundationalTS")
from llm_timeseries_finetune.serialize import serialize_arr, SerializerSettings
import numpy as np

from torch.utils.data import DataLoader

@dataclass
class PlainTextDataset(QADataset):
    """QA dataset that serializes time series into plain text.

    This dataset uses the standard question‑answer format but treats
    the associated time‑series data as part of the prompt.  Each
    one‑dimensional channel of the multivariate series is **scaled**
    following the LLMTIME preprocessing: values are shifted by an
    offset ``b = \min(x) - \beta (\max(x) - \min(x))`` and then
    divided by ``a``, the ``\alpha``‑th percentile of the shifted
    series, so that the chosen percentile becomes 1.  The scaled
    series is then serialized into a string using the provided
    ``SerializerSettings``.  The resulting strings are appended to
    the pre‑prompt text and separated by newlines.  The post‑prompt
    text (e.g. ``"Answer:"``) is added afterwards.  The resulting
    prompt can be directly tokenized by a language model without
    additional time‑series processing.

    Parameters
    ----------
    split : {"train", "test", "validation"}
        Which portion of the dataset to load.
    eos_token : str
        End‑of‑sequence token appended to the answer.
    settings : SerializerSettings, optional
        Configuration controlling how the time series are serialized.
        If omitted, a default configuration (base‑10, three decimal
        places, comma separators) is used.
    dataset_path : str, optional
        Optional override for the on‑disk dataset to load.  Useful for
        experimenting with different sources.  Defaults to the
        ``ChatTS align_256`` dataset used in the original paper.
    alpha : float, optional
        Percentile threshold used for scaling.  See ``b`` and ``a``
        definitions above.  Defaults to 0.3.
    beta : float, optional
        Offset factor used in computing ``b``.  Defaults to 0.0.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        settings: Optional[SerializerSettings] = None,
        dataset_path: str = "/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS/process/align_256",
        alpha: float = 0.3,
        beta: float = 0.0,
    ) -> None:
        # Set default serializer settings if none provided
        self.settings = settings or SerializerSettings()
        self.alpha = alpha
        self.beta = beta

        # Override the dataset path for loading if the default differs
        self.dataset_path = dataset_path

        # Define a wrapper to rescale and serialise each 1D time series
        def ts_formatter(time_series: torch.Tensor) -> str:
            arr = torch.as_tensor(time_series, dtype=torch.float32).numpy()
            if arr.size == 0:
                return serialize_arr(arr, self.settings)
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            b = arr_min - self.beta * (arr_max - arr_min)
            shifted = arr - b
            try:
                a = float(np.nanpercentile(shifted, self.alpha * 100))
            except Exception:
                a = 0.0
            if not np.isfinite(a) or a <= 0:
                a = 0.01
            a = max(a, 0.01)
            rescaled = (arr - b) / a
            return serialize_arr(rescaled, self.settings)

        # We will use the string formatting mode provided by QADataset.  To
        # achieve this, we call the parent constructor with
        # ``format_sample_str=True`` and supply ``ts_formatter`` as the
        # custom time_series_format_function.  This causes each
        # ``TextTimeSeriesPrompt`` to be converted into a line of text
        # representing the normalised and serialised series.  The
        # ``QADataset`` base class will automatically join the pre‑prompt,
        # the time‑series prompts (their ``text`` attributes), the
        # serialised time series strings returned by ``ts_formatter``, and
        # the post‑prompt into a single ``prompt`` field.
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            time_series_format_function=ts_formatter,
        )

    # ------------------------------------------------------------------
    # Methods required by QADataset
    # ------------------------------------------------------------------
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and partition the underlying ChatTS dataset.

        We load the full ``train`` split from disk and then carve out
        validation and test subsets using fixed fractions.  The
        ``dataset_path`` attribute can be customised at construction
        time to point to a different HuggingFace dataset or local
        directory.
        """
        ds_full = load_dataset(
            self.dataset_path, split="train", trust_remote_code=True
        )

        # NOTE: get 100 random samples for quick testing
        # ds_full = ds_full.shuffle(seed=42).select(range(100))
        
        # Carve out test and validation splits using fixed fractions
        train_val, test = ds_full.train_test_split(test_size=TEST_FRAC, seed=42).values()
        train, val = train_val.train_test_split(
            test_size=VAL_FRAC / (1 - TEST_FRAC), seed=42
        ).values()
        return train, val, test

    def _get_answer(self, row) -> str:
        return row["output"]

    def _get_pre_prompt(self, row) -> str:
        return row["input"]

    def _get_post_prompt(self, row) -> str:
        # Always request an answer following the context
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List:
        """
        Extract and scale the multivariate time series for this row.

        Each channel is scaled using the same affine transformation as in
        LLMTIME: shift by ``b = min(x) - beta * (max(x) - min(x))`` and
        divide by ``a``, the alpha‑th percentile of the shifted series.
        The resulting scaled series is stored for potential downstream use
        and a descriptive string summarises the scaling.
        """
        ts = row["timeseries"]
        if isinstance(ts, str):
            ts = ast.literal_eval(ts)
        series = torch.tensor(ts, dtype=torch.float32)
        prompts = []
        
        for chan in series:
            arr = chan.numpy()
            if arr.size == 0:
                scaled = arr
                mean_after = float("nan")
                std_after = float("nan")
            else:
                arr_min = float(np.nanmin(arr))
                arr_max = float(np.nanmax(arr))
                b = arr_min - self.beta * (arr_max - arr_min)
                shifted = arr - b
                try:
                    a = float(np.nanpercentile(shifted, self.alpha * 100))
                except Exception:
                    a = 0.0
                if not np.isfinite(a) or a <= 0:
                    a = 0.01
                a = max(a, 0.01)
                scaled = (arr - b) / a
                mean_after = float(np.nanmean(scaled))
                std_after = float(np.nanstd(scaled))
            description = (
                ""
            )
            prompts.append(TextTimeSeriesPrompt(description, scaled.tolist()))
        return prompts


# Fractions used for splitting the dataset.  We define them below the
# class so they can be adjusted without editing class code.
TEST_FRAC = 0.1
VAL_FRAC = 0.1

if __name__ == "__main__":
    train = PlainTextDataset("train", "")
    val = PlainTextDataset("validation", "")
    test = PlainTextDataset("test", "")

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
