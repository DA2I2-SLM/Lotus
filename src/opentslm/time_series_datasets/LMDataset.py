from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Tuple, List
import numpy as np

from opentslm.time_series_datasets.QADataset import QADataset


class LMDataset(QADataset):
    """Base dataset class for language modeling.

    This class inherits from :class:`QADataset` but overrides the
    formatting logic to return a single text sequence per sample.  It
    forces ``format_sample_str=True`` in the constructor, so a
    ``time_series_format_function`` must be provided if your dataset
    includes numerical time series.  If ``format_sample_str=False`` is
    requested, a ``ValueError`` is raised to prevent accidental use
    with the QA formatting.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = True,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        # We require format_sample_str=True for language modelling so that
        # the base class uses our overridden _format_sample_str.  If
        # format_sample_str=False is passed explicitly, raise an error.
        if not format_sample_str:
            raise ValueError(
                "LMDataset only supports format_sample_str=True. "
                "Use QADataset for QA tasks."
            )
        # Store the time series formatting function for use in formatting
        self._lm_time_series_format_function = time_series_format_function
        # Call QADataset constructor with format_sample_str=True and our
        # provided time_series_format_function.  This will trigger the
        # caching of formatted samples using our overridden
        # _format_sample_str.
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            format_sample_str=True,
            time_series_format_function=time_series_format_function,
        )

    # def _format_sample_str(self, time_series_format_function, row):  # type: ignore[override]
    #     """Return a single text string combining prompt and answer.

    #     The base :meth:`QADataset._format_sample_str` returns a dict with
    #     keys ``"prompt"`` and ``"answer"``.  This override appends the
    #     EOS token to the answer if necessary, concatenates the prompt
    #     and answer with a newline, and returns the result under the key
    #     ``"text"``.  If ``time_series_format_function`` is ``None``, a
    #     fallback formatter will be used by the base implementation.
    #     """
    #     # Get the base prompt/answer dict by calling the parent implementation.
    #     base = super()._format_sample_str(time_series_format_function, row)
    #     answer = base["answer"]
    #     # Append EOS if missing
    #     if not answer.endswith(self.EOS_TOKEN):
    #         answer = answer + self.EOS_TOKEN


    #     # # Concatenate prompt and answer with a newline
    #     # if '\n\n' in base['prompt']:
    #     #     # replace with \n
    #     #     base['prompt'] = base['prompt'].replace('\n\n', 'HAHAHAHAHAHAHAHAHAHAHAHAHAH')
            
    #     text = f"{base['prompt']}\n{answer}"
        
    #     return {"text": text}




    def _format_sample_str(
        self, time_series_format_function: Callable[[np.ndarray], str] | None, row
    ):
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

        for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
            prompt_chunks.append(text_time_series_prompt.get_text())
            time_series = time_series_format_function(
                text_time_series_prompt.get_time_series()
            )
            prompt_chunks.append(time_series)

        prompt_chunks.append(self._get_post_prompt(row).strip())


        return {"prompt": "\n".join(prompt_chunks), "answer": self._get_answer(row)}
