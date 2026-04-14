# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal, Tuple

import numpy as np
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class PretrainDataset(Dataset, ABC):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        """
        Initializes the dataset by loading and formatting the specified data split.
        Args:
            split (Literal["train", "test", "validation"]): The dataset split to load. Must be one of "train", "test", or "validation".
            EOS_TOKEN (str): End-of-sequence token to be used in formatting.
            format_sample_str (bool, optional): If True, applies a string formatting function to each sample. Defaults to False.
            time_series_format_function (Callable[[np.ndarray], str] | None, optional): Optional function to format time series data as strings. Used only if `format_sample_str` is True.
        Raises:
            RuntimeError: If the provided split is not one of "train", "test", or "validation".
        Notes:
            - The datasets for each split are loaded and formatted only once per class.
            - The formatted datasets are cached as class attributes for subsequent initializations.
        """
        
        self.EOS_TOKEN = EOS_TOKEN
        self.format_sample_str = format_sample_str
        self.time_series_format_function = time_series_format_function

        # Keep caches subclass-local. Using hasattr() here is incorrect because
        # it treats a parent class cache as if it belonged to the subclass.
        # That can make a Chronos-only subclass accidentally reuse the joint
        # dataset rows cached on Phase1JointDataset, pulling in TSQA/M4 paths.
        if "loaded" not in self.__class__.__dict__:
            train, val, test = self._load_splits()

            # Store raw rows so each split instance formats with its own mode/state
            # at __getitem__ time. This fixes two issues:
            #   1. val/test were formatted with train-mode windowing (wrong random crop)
            #   2. train samples were cached with a fixed random crop (no epoch diversity)
            self.__class__._train_rows = train
            self.__class__._val_rows = val
            self.__class__._test_rows = test
            self.__class__.loaded = True

        match split:
            case "train":
                self.rows = self.__class__._train_rows
            case "validation":
                self.rows = self.__class__._val_rows
            case "test":
                self.rows = self.__class__._test_rows
            case _:
                raise RuntimeError(
                    "Split must be a literal of 'train', 'test', or 'validation'"
                )


    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass


    @abstractmethod
    def _get_answer(self, row) -> str:
        pass


    @abstractmethod
    def _get_raw_ts(self, row) -> List:
        pass


    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        pass


    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        pass


    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        pass


    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN

        return PromptWithAnswer(
            TextPrompt(self._get_pre_prompt(row).strip()),
            self._get_text_time_series_prompt_list(row),
            TextPrompt(self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()


    def _format_sample_str(
        self, time_series_format_function: Callable[[np.ndarray], str] | None, row
    ):
        
        self.place_holders = "<ts>"
        def fallback_timeseries_formatter(time_series: np.ndarray) -> str:
            # Fallback formatter for time series data
        
            return np.array2string(
                time_series,
                separator=" ",
                formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
                threshold=sys.maxsize,
                max_line_width=sys.maxsize,
            ).removeprefix("[").removesuffix("]")
  
        if not time_series_format_function:
            time_series_format_function = fallback_timeseries_formatter

        # Create the prompt chunks: pre-prompt, time series prompts, and post-prompt
        prompt_chunks = [self._get_pre_prompt(row).strip()]

        if self.place_holders in prompt_chunks[0]:  # assume the texts are all formatted
            pre = prompt_chunks[0]
            cursor = 0  # where we start searching for the next placeholder

            for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
                prompt_chunks.append(text_time_series_prompt.get_text())

                time_series = time_series_format_function(
                    text_time_series_prompt.get_time_series()
                )

                idx = pre.find(self.place_holders, cursor)
                if idx == -1:
                    raise ValueError(
                        f"Not enough placeholders '{self.place_holders}' in pre-prompt. "
                        f"Needed at least {len(self._get_text_time_series_prompt_list(row))}."
                    )

                insert_pos = idx + len(self.place_holders)
                pre = pre[:insert_pos] + time_series + pre[insert_pos:]

                # Advance cursor past what we just inserted, so next TS goes to the next placeholder
                cursor = insert_pos + len(time_series)

            prompt_chunks[0] = pre
            prompt_chunks.append(self._get_post_prompt(row).strip())
        else:
            for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
                prompt_chunks.append(text_time_series_prompt.get_text())
                time_series = time_series_format_function(
                    text_time_series_prompt.get_time_series()
                )
                prompt_chunks.append(time_series)

            prompt_chunks.append(self._get_post_prompt(row).strip())

        return {"prompt": "\n".join(prompt_chunks), "answer": self._get_answer(row)}


    def __len__(self):
        return len(self.rows)


    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.format_sample_str:
            return partial(self._format_sample_str, self.time_series_format_function)(row)
        return self._format_sample(row)
