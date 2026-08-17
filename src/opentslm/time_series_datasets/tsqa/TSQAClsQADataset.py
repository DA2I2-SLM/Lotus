"""TSQA (ChengsenWang/TSQA) as an OpenTSLM QADataset.

TSQA is a multiple-choice time-series QA dataset (~48k, single split). Each row:
  Task, Size, Question (MCQ w/ definitions), Answer ('(a)'), Label ('constant trend'), Series (list).
We frame it as label-classification: pre_prompt = the Question (definitions/context),
answer = the Label text, so eval scores accuracy/F1 over the discrete Label set — the
same protocol as HAR/Sleep/ECG. The single split is carved 90/5/5 (deterministic).
"""
import ast
from functools import lru_cache
from typing import List, Literal, Tuple

import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset


def _parse_series(v):
    if isinstance(v, str):
        return ast.literal_eval(v)
    return list(v)


class TSQAClsQADataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        from datasets import load_dataset
        ds = load_dataset("ChengsenWang/TSQA", split="train")
        rows = list(ds)
        # deterministic 90/5/5 split (shuffle with a fixed seed for class mixing)
        import random
        rng = random.Random(1234)
        rng.shuffle(rows)
        n = len(rows)
        n_tr = int(0.90 * n)
        n_va = int(0.95 * n)
        return rows[:n_tr], rows[n_tr:n_va], rows[n_va:]

    def _get_answer(self, row) -> str:
        # answer = the Label text (the correct choice), so eval matches over labels.
        return str(row["Label"]).strip()

    def _get_pre_prompt(self, row) -> str:
        q = str(row["Question"]).strip()
        return (
            "You are given a time series. Use the following question and definitions "
            "to determine the correct category.\n" + q
        )

    def _get_post_prompt(self, _row) -> str:
        return "\nAnswer with the single correct label."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(_parse_series(row["Series"]), dtype=torch.float32).flatten()
        mean = series.mean()
        std = series.std()
        if std < 1e-8:
            std = torch.tensor(1.0)
        series_norm = (series - mean) / std
        text = f"This is the time series, mean {mean.item():.4f} std {std.item():.4f}."
        return [TextTimeSeriesPrompt(text, series_norm.tolist())]

    @staticmethod
    @lru_cache(maxsize=1)
    def get_labels() -> List[str]:
        from datasets import load_dataset
        ds = load_dataset("ChengsenWang/TSQA", split="train")
        return sorted({str(r).strip() for r in ds["Label"]})
