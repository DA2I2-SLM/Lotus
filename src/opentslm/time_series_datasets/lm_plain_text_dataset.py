"""
Language modeling dataset built on top of the plain text QA dataset.

This module defines a :class:`LanguageModelingPlainTextDataset` which
extends :class:`PlainTextDataset` to produce samples formatted for
language modeling.  Each entry in the underlying QA dataset
consists of a ``prompt`` (composed of a pre‑prompt, serialized time
series and a post‑prompt such as ``"Answer:"``) and an ``answer``.

In language modeling, the objective is to predict every next token in
the sequence, rather than only the answer tokens.  This wrapper
concatenates the ``prompt`` and ``answer`` fields into a single
string and ensures that the answer ends with the configured
end‑of‑sequence token.  The resulting dictionary returned by
``__getitem__`` contains a single key ``"text"`` suitable for
tokenization by a language model.

Example
-------
>>> from llm_timeseries_finetune.serialize import SerializerSettings
>>> from llm_timeseries_finetune.lm_plain_text_dataset import LanguageModelingPlainTextDataset
>>> settings = SerializerSettings(base=10, prec=3, time_sep=", ", bit_sep="")
>>> ds = LanguageModelingPlainTextDataset("train", eos_token="<eos>", settings=settings)
>>> sample = ds[0]
>>> print(sample["text"])

This will output a single string containing the question, the serialized
time‑series context, the ``"Answer:"`` cue, the answer and the EOS
token, separated by newlines.  When used with a Hugging Face
``DataCollatorForLanguageModeling``, no further formatting is needed.
"""

from __future__ import annotations

from typing import Literal, Optional

from .plain_text_dataset import PlainTextDataset
from .serialize import SerializerSettings


class LanguageModelingPlainTextDataset(PlainTextDataset):
    """Wraps :class:`PlainTextDataset` to produce language modeling samples.

    This class inherits from :class:`PlainTextDataset` and overrides
    the string formatting logic so that each sample contains a single
    ``"text"`` entry combining the prompt and answer.  It ensures
    that the answer ends with the EOS token and inserts a newline
    between the prompt and answer.  The scaling hyperparameters
    ``alpha`` and ``beta`` are forwarded to the base class.

    Parameters
    ----------
    split : {"train", "test", "validation"}
        Which portion of the dataset to load.
    eos_token : str
        End‑of‑sequence token appended to the answer if not already present.
    settings : SerializerSettings, optional
        Serialization settings controlling how numeric values are
        converted to text.  If omitted, defaults to base‑10 with
        three decimal places and comma separators.
    dataset_path : str, optional
        Path to the underlying HuggingFace dataset or local directory.
        Defaults to the ChatTS align_256 used in the original work.
    alpha : float, optional
        Percentile threshold used for scaling each time series channel.
        Passed directly to :class:`PlainTextDataset`.  Defaults to 0.3.
    beta : float, optional
        Offset factor used in scaling each time series channel.
        Passed directly to :class:`PlainTextDataset`.  Defaults to 0.0.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        eos_token: str,
        settings: Optional[SerializerSettings] = None,
        dataset_path: str = "/scratch/Projects/rl_timeseries/pretraining_data/langtime/ChatTS/process/align_256",
        alpha: float = 0.3,
        beta: float = 0.0,
    ) -> None:
        super().__init__(
            split=split,
            eos_token=eos_token,
            settings=settings,
            dataset_path=dataset_path,
            alpha=alpha,
            beta=beta,
        )

    def _format_sample_str(self, time_series_format_function, row):  # type: ignore[override]
        """Format a sample as a single text string for language modeling.

        This method overrides the base implementation to concatenate
        the prompt and answer into a single string.  It adds the EOS
        token to the answer if not already present and separates the
        prompt and answer with a newline.
        """
        # Call the base implementation to get the formatted prompt/answer
        base_result = super()._format_sample_str(time_series_format_function, row)
        # Ensure the answer ends with the EOS token
        answer = base_result["answer"]
        if not answer.endswith(self.EOS_TOKEN):
            answer = answer + self.EOS_TOKEN
        # Concatenate prompt and answer
        text = f"{base_result['prompt']}\n{answer}"
        return {"text": text}