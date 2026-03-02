"""
Training script for fine‑tuning Gemma and Llama 3 models on time‑series data.

This module provides a command line interface to fine‑tune a causal language
model on sequences of serialized time‑series values.  It uses the
serialization routines defined in :mod:`serialize` to convert arrays of real
numbers into text and employs the Hugging Face `transformers` library for
model loading and training.  The script is intentionally simple and avoids
framework‑specific training abstractions so that it can be customised
according to the user's needs.

Usage
-----

To fine‑tune a Gemma or Llama 3 model on a CSV file of time series, run

.. code-block:: bash

    python train.py \
        --model_name google/gemma-2b \
        --data_path path/to/data.csv \
        --output_dir ./finetuned-gemma \
        --epochs 3 --batch_size 4

Replace ``google/gemma-2b`` with the desired model, e.g. ``meta-llama/Llama-3-8B``.

Implementation details
----------------------

The dataset loader uses :class:`dataset.TimeSeriesDataset` to read numeric
sequences and produces serialized strings.  These strings are tokenized by
``AutoTokenizer.from_pretrained(model_name)``.  Tokenization truncates or
pads sequences to a configurable maximum length.  During training, the
``DataCollatorForLanguageModeling`` supplies input IDs and labels such that
the model is trained to predict the next token given the previous context.

Because time‑series values are encoded as sequences of digit and separator
tokens, long histories can easily exceed the context length of the base
model.  The user should choose a suitable ``--max_length`` to avoid
truncation.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np  # noqa: F401  # used by serialize imports

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, DatasetDict

from serialize import SerializerSettings
from dataset import TimeSeriesDataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Fine‑tune a causal LM on serialized time series")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model identifier, e.g. google/gemma-2b or meta-llama/Llama-3-8B")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a CSV or JSON file containing time series data")
    parser.add_argument("--output_dir", type=str, default="./finetuned-model",
                        help="Directory where the fine‑tuned model checkpoints will be saved")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the AdamW optimizer")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (in tokens) to feed into the model")
    parser.add_argument("--train_fraction", type=float, default=0.9,
                        help="Fraction of the data to use for training (remainder used for evaluation)")
    parser.add_argument("--base", type=int, default=10, help="Base for digit representation")
    parser.add_argument("--prec", type=int, default=3, help="Number of fractional digits to encode")
    parser.add_argument("--bit_sep", type=str, default="",
                        help="Separator between individual digits (empty string joins digits)")
    parser.add_argument("--time_sep", type=str, default=", ",
                        help="Separator between time steps")
    parser.add_argument("--plus_sign", type=str, default="",
                        help="Prefix string for positive numbers when signed=True")
    parser.add_argument("--minus_sign", type=str, default="-",
                        help="Prefix string for negative numbers when signed=True")
    parser.add_argument("--max_val", type=float, default=1e7,
                        help="Largest absolute value expected in the data (controls integer digit width)")
    parser.add_argument("--no_signed", action="store_true",
                        help="If set, assume all values are non‑negative and do not encode the sign")
    return parser.parse_args()


def load_dataset(data_path: Path, settings: SerializerSettings, train_fraction: float) -> DatasetDict:
    """Load a CSV/JSON file into a Hugging Face ``Dataset`` and split it."""
    ts_dataset = TimeSeriesDataset(data_path, settings)
    # Extract serialized text strings
    texts: List[str] = [item["text"] for item in ts_dataset]
    ds = Dataset.from_dict({"text": texts})
    # Split into train and eval
    if train_fraction < 1.0:
        split = ds.train_test_split(test_size=1.0 - train_fraction, shuffle=True)
        return split
    else:
        return DatasetDict({"train": ds, "test": Dataset.from_dict({"text": []})})


def main():
    args = parse_args()
    # Prepare serialization settings
    settings = SerializerSettings(
        base=args.base,
        prec=args.prec,
        signed=not args.no_signed,
        fixed_length=False,
        max_val=args.max_val,
        time_sep=args.time_sep,
        bit_sep=args.bit_sep,
        plus_sign=args.plus_sign,
        minus_sign=args.minus_sign,
        half_bin_correction=True,
        decimal_point="",
        missing_str="NaN",
    )
    data_path = Path(args.data_path)
    # Load the raw dataset and split into train/test
    dataset_dict = load_dataset(data_path, settings, args.train_fraction)
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure EOS token is defined; if not, assign pad token as EOS
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tokenize the text fields
    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
    tokenized = dataset_dict.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    # Build data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Load model (weights in float16 if available)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # If the tokenizer added new tokens, resize the model embeddings
    model.resize_token_embeddings(len(tokenizer))
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch" if len(tokenized["test"]) > 0 else "no",
        save_strategy="epoch",
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
    )
    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test") if len(tokenized.get("test", [])) > 0 else None,
        data_collator=data_collator,
    )
    # Kick off training
    trainer.train()
    # Save the final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()