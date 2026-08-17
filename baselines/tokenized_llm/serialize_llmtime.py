# SPDX-License-Identifier: MIT
"""LLMTime serialisation, ported from Gruver et al. (2023).

Source: https://github.com/ngruver/llmtime  (data/serialize.py). This is a
faithful port of `vec_num2repr` / `serialize_arr` and their `SerializerSettings`
defaults, so the "Tokenized time-series" baseline uses the published encoding
rather than an approximation of it.

Why it differs from serialize.py (our own baseline) and why that matters:

  ours     : ", ".join(f"{v:.2f}")        ->  "[0.12, -0.45, 1.03]"
  llmtime  : digit-level, no decimal pt   ->  " 1 2 3 , -4 5 0 , 1 0 3 0 ,"

Gruver et al. space out individual digits (`bit_sep=' '`) and drop the decimal
point entirely (`decimal_point=''`), letting position carry the precision
(`prec=3`). That is deliberate: it forces a BPE tokenizer to emit one token per
digit instead of merging "0.12" into arbitrary multi-character pieces, which is
what makes the numbers comparable across magnitudes. It also costs roughly 2x
the tokens of a plain "%.2f" rendering, which matters a great deal here — see
the decimation note below.

Scaling: LLMTime rescales each series before serialising so values land in a
range the fixed-precision digit encoding can represent. We reproduce that with
their default alpha=0.99/beta=0.3 offset-and-scale, applied PER SERIES, and we
record the scaler so the encoding is reversible if anyone needs to check it.

NOT reproduced: their `half_bin_correction` matters only for deserialisation
(reading model output back into numbers). These baselines are scored on text
answers, not on decoded forecasts, so it is unused here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class SerializerSettings:
    """Gruver et al.'s defaults, verbatim."""
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = " ,"
    bit_sep: str = " "
    plus_sign: str = ""
    minus_sign: str = " -"
    half_bin_correction: bool = True
    decimal_point: str = ""
    missing_str: str = " Nan"


def vec_num2repr(val: np.ndarray, base: int, prec: int, max_val: float):
    """Port of llmtime data/serialize.py::vec_num2repr.

    base=10, prec=2:  0.5 -> 50 ; 3.52 -> 352 ; 12.5 -> 1250
    """
    base = float(base)
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val = val - digit * base ** (max_bit_pos - i - 1)
    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val = val - digit * base ** (-i - 1)
        digits = np.concatenate([before_decimals, np.stack(after_decimals, axis=-1)], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


def serialize_arr(arr: np.ndarray, settings: SerializerSettings = SerializerSettings()) -> str:
    """Port of llmtime data/serialize.py::serialize_arr."""
    finite = arr[~np.isnan(arr)]
    if finite.size and np.max(np.abs(finite)) > settings.max_val:
        raise ValueError(f"value exceeds max_val={settings.max_val}; rescale first")

    if not settings.signed:
        plus_sign = minus_sign = ""
    else:
        plus_sign, minus_sign = settings.plus_sign, settings.minus_sign

    sign_arr, digits_arr = vec_num2repr(
        np.where(np.isnan(arr), np.zeros_like(arr), arr),
        base=settings.base, prec=settings.prec, max_val=settings.max_val)
    ismissing = np.isnan(arr)

    def tokenize(a):
        return "".join([settings.bit_sep + str(b) for b in a])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            nz = np.where(digits != 0)[0]
            digits = np.array([0]) if len(nz) == 0 else digits[nz[0]:]
            if len(settings.decimal_point):
                p = settings.prec
                digits = np.concatenate(
                    [digits[:-p], np.array([settings.decimal_point]), digits[-p:]])
        d = tokenize(digits)
        sep = plus_sign if sign == 1 else minus_sign
        bit_strs.append(settings.missing_str if missing else sep + d)
    return settings.time_sep.join(bit_strs) + settings.time_sep


def scale_series(values: Sequence[float], alpha: float = 0.99, beta: float = 0.3
                 ) -> Tuple[np.ndarray, float, float]:
    """LLMTime's per-series offset-and-scale, so values fit the digit encoding.

    Returns (scaled, offset, scale) so the transform is invertible.
    """
    v = np.asarray(list(values), dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v, 0.0, 1.0
    offset = float(np.min(finite)) - beta * float(np.max(finite) - np.min(finite) + 1e-12)
    shifted = v - offset
    q = np.quantile(np.abs(shifted[np.isfinite(shifted)]), alpha)
    scale = float(q) if q > 1e-12 else 1.0
    return shifted / scale, offset, scale


def format_series_llmtime(values: Sequence[float], precision: int = 3) -> str:
    """Drop-in replacement for serialize.format_series, using LLMTime's encoding.

    `precision` maps onto SerializerSettings.prec so the caller's existing
    knob still means "digits after the point".
    """
    scaled, _off, _sc = scale_series(values)
    return serialize_arr(scaled, SerializerSettings(prec=precision))


if __name__ == "__main__":
    # Round-trip the examples in Gruver et al.'s own docstring.
    s, d = vec_num2repr(np.array([0.5, 3.52, 12.5]), base=10, prec=2, max_val=1e7)
    got = [int("".join(map(str, row))) for row in d]
    assert got == [50, 352, 1250], got
    print("vec_num2repr matches the paper's docstring examples:", got)
    print("serialize_arr([1.0, -2.5, 0.125]) ->",
          repr(serialize_arr(np.array([1.0, -2.5, 0.125]))))
    print("format_series_llmtime([0.1,0.2,0.3]) ->",
          repr(format_series_llmtime([0.1, 0.2, 0.3])))
