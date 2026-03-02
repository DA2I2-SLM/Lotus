"""
Dataset utilities for converting time series data into textual sequences.

This module defines a simple :class:`TimeSeriesDataset` class that reads raw
time‑series data from a CSV or JSON file and uses the serialization routines
implemented in :mod:`serialize` to transform each sequence into a string.  The
strings are then tokenized on the fly using a Hugging Face tokenizer inside
the training script.  The dataset class is intentionally lightweight to
facilitate streaming large datasets and allows for custom preprocessing
pipelines.

Input format
------------

The dataset loader assumes that each entry in the source file represents a
single univariate time series.  Supported formats include:

* **CSV** – Each row contains a comma‑separated list of floating point
  numbers.  Missing values may be encoded as empty strings or ``NaN``.
* **JSON** – A JSON file containing either a list of lists (e.g.
  ``[[1.0, 2.0, 3.0], [4.5, 5.0]]``) or a list of objects with a key
  ``"values"`` whose value is a list of numbers.

To construct a :class:`TimeSeriesDataset`, pass the path to the data file and
a :class:`serialize.SerializerSettings` object that specifies how to encode
numbers.  The dataset returns a dictionary with a single key ``"text"`` and
the serialized sequence as the value.  When used in conjunction with
`transformers` data collators, this enables straightforward fine‑tuning of
language models.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Any

import numpy as np

from serialize import serialize_arr, SerializerSettings


class TimeSeriesDataset:
    """PyTorch dataset that converts numeric sequences to serialized text.

    Parameters
    ----------
    data_path : str or Path
        Path to a CSV or JSON file containing time series data.  See
        module docstring for supported formats.
    settings : SerializerSettings
        Serialization parameters controlling the base, precision and
        separators used to convert numbers into text【716284818101681†L112-L170】.
    limit : Optional[int], optional
        If provided, limit the number of records read from the file.  This
        can be useful for quick experiments or when slicing large datasets.
    """

    def __init__(self, data_path: Path, settings: SerializerSettings,
                 limit: Optional[int] = None) -> None:
        self.data_path = Path(data_path)
        self.settings = settings
        self.limit = limit
        # Load data once during initialization to determine length
        self._sequences: List[np.ndarray] = list(self._load_sequences())

    def _load_sequences(self) -> Iterable[np.ndarray]:
        ext = self.data_path.suffix.lower()
        count = 0
        if ext == ".csv":
            with open(self.data_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if self.limit is not None and count >= self.limit:
                        break
                    # Handle missing values and whitespace
                    values: List[float] = []
                    for cell in row:
                        cell = cell.strip()
                        if cell == "" or cell.lower() == "nan":
                            values.append(np.nan)
                        else:
                            try:
                                values.append(float(cell))
                            except ValueError:
                                raise ValueError(
                                    f"Could not parse value '{cell}' on line {count+1}")
                    yield np.array(values, dtype=float)
                    count += 1
        elif ext == ".json":
            with open(self.data_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if self.limit is not None and count >= self.limit:
                        break
                    if isinstance(item, list):
                        arr = np.array([float(x) if x is not None else np.nan for x in item], dtype=float)
                    elif isinstance(item, dict) and "values" in item:
                        arr = np.array([float(x) if x is not None else np.nan for x in item["values"]], dtype=float)
                    else:
                        raise ValueError("JSON entries must be lists or objects with a 'values' key")
                    yield arr
                    count += 1
            else:
                raise ValueError("JSON file must contain a list at the top level")
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: .csv, .json")

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Any:
        seq = self._sequences[idx]
        text = serialize_arr(seq, self.settings)
        return {"text": text}