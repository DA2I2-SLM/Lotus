"""
Utility functions to convert numerical time‑series data to and from textual
representations.  These routines are adapted from the open‑source `llmtime`
project, which demonstrated how to represent real‑valued sequences as strings
for large language models (LLMs).  Each number is expressed in a chosen
integer base with a fixed precision and separators are inserted between
digits and time steps.  The resulting string can be fed into a language model
and later deserialized back into floating point values.

The functions below are simplified for fine‑tuning open models such as
Gemma and Llama 3.  They use NumPy for array manipulations and follow the
same API as the original code.  See the `SerializerSettings` class for a
description of the configuration parameters.

The original implementation from `llmtime` (NeurIPS 2023) serializes
numbers by converting them to a vector of digits and joining them with
custom separators.  The reimplementation here
preserves that behaviour while avoiding dependencies on JAX or thread pools.
"""

from dataclasses import dataclass
from functools import partial
import numpy as np
from typing import Callable, Tuple, List, Optional


def vec_num2repr(val: np.ndarray, base: int, prec: int, max_val: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert floating point numbers to a digit representation.

    Parameters
    ----------
    val : np.ndarray
        Input array of numbers to represent.  Absolute values larger than
        ``max_val`` will raise an assertion in the caller.
    base : int
        Base of the representation (e.g. 10 for decimal).  Must be >= 2.
    prec : int
        Number of digits to encode after the decimal point.
    max_val : float
        Upper bound on the absolute value used to determine the number of
        digits before the decimal point.

    Returns
    -------
    sign : np.ndarray
        Array of ±1 indicating the sign of each element in ``val``.
    digits : np.ndarray
        2‑D array where each row contains the digits for one value.  The
        number of columns is determined by the magnitude of the largest
        number and the requested precision.
    """
    base = float(base)
    # Store the original shape so we can flatten and reshape later
    shape = val.shape
    val = val.reshape(-1)
    sign = np.where(val >= 0, 1, -1)
    abs_val = np.abs(val)
    # Determine how many integer digits are needed to represent max_val
    if max_val <= 0:
        raise ValueError("max_val must be positive")
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)))
    # Compute digits before the decimal point
    before_decimals: List[np.ndarray] = []
    temp = abs_val.copy()
    for i in range(max_bit_pos):
        power = max_bit_pos - i - 1
        factor = base**power
        digit = (temp / factor).astype(int)
        before_decimals.append(digit)
        temp = temp - digit * factor
    before_decimals = np.stack(before_decimals, axis=-1)
    # Compute digits after the decimal point
    if prec > 0:
        after_decimals: List[np.ndarray] = []
        for i in range(prec):
            factor = base**(-i - 1)
            digit = (temp / factor).astype(int)
            after_decimals.append(digit)
            temp = temp - digit * factor
        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    # Reshape sign to match original array size
    sign = sign.reshape(shape)
    digits = digits.reshape(shape + (digits.shape[-1],))
    return sign, digits


def vec_repr2num(sign: np.ndarray, digits: np.ndarray, base: int, prec: int,
                 half_bin_correction: bool = True) -> np.ndarray:
    """Convert digit representations back to floating point numbers.

    Parameters
    ----------
    sign : np.ndarray
        Array of ±1 signs for each number.
    digits : np.ndarray
        2‑D array of digits with shape ``(..., D)``.  The last dimension
        corresponds to the digit positions.
    base : int
        Base of the representation.
    prec : int
        Number of digits after the decimal point.
    half_bin_correction : bool, optional
        If ``True``, add 0.5 / base**prec to account for half‑bin rounding.

    Returns
    -------
    np.ndarray
        The reconstructed numbers.
    """
    base = float(base)
    # Flatten all but last dimension for simplicity
    original_shape = digits.shape[:-1]
    D = digits.shape[-1]
    digits_flipped = np.flip(digits.reshape(-1, D), axis=-1)
    powers = -np.arange(-prec, -prec + D)
    # Summation over the digit positions
    val = np.sum(digits_flipped / (base ** powers), axis=-1)
    if half_bin_correction:
        val = val + 0.5 / (base ** prec)
    val = sign.reshape(-1) * val
    return val.reshape(original_shape)


@dataclass
class SerializerSettings:
    """Configuration for serializing time series to text.

    Attributes
    ----------
    base : int, optional
        Base of the digit representation.  A base of 10 produces decimal
        strings.  A smaller base increases the vocabulary efficiency but
        reduces resolution.
    prec : int, optional
        Number of digits to include after the decimal point.  A value of
        ``0`` produces integer‑like representations; larger values encode
        finer grained information.
    signed : bool, optional
        Whether to include sign information.  If ``False`` all values are
        assumed to be non‑negative.
    fixed_length : bool, optional
        If ``True``, pad numbers to a fixed length determined by ``max_val``
        and ``prec``.  Otherwise, leading zeros are removed to shorten the
        representation.
    max_val : float, optional
        Maximum absolute value allowed in the input array.  This controls the
        number of digits allocated for the integer part.
    time_sep : str, optional
        Separator inserted between time steps.  The default uses a comma
        followed by a space.
    bit_sep : str, optional
        Separator inserted between individual digit symbols.  Setting this to
        ``''`` results in contiguous digit strings, while a space can
        simplify tokenization.
    plus_sign : str, optional
        String prefix for positive numbers when ``signed`` is ``True``.
    minus_sign : str, optional
        String prefix for negative numbers when ``signed`` is ``True``.
    half_bin_correction : bool, optional
        Whether to adjust the deserialized values by half of the smallest
        representable increment.
    decimal_point : str, optional
        String representation of the decimal point inserted when ``bit_sep``
        is non‑empty and ``prec > 0``.  When empty, no explicit decimal
        delimiter is inserted and the last ``prec`` digits are interpreted as
        fractional.
    missing_str : str, optional
        Token representing missing values (``NaN``).  This is passed
        through verbatim during serialization and deserialization.
    """

    base: int = 10
    prec: int = 2
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ", "
    bit_sep: str = ""
    plus_sign: str = ""
    minus_sign: str = "-"
    half_bin_correction: bool = True
    decimal_point: str = ""
    missing_str: str = "NaN"


def serialize_arr(arr: np.ndarray, settings: SerializerSettings) -> str:
    """Serialize a one‑dimensional array into a string.

    Each element of the input array is converted into a string of digit
    characters using the ``vec_num2repr`` routine and joined together with
    the separators defined in ``settings``.  Leading zeros may be stripped
    when ``settings.fixed_length`` is ``False``.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape ``(n,)`` containing real values.  Any ``NaN``
        values will be represented using ``settings.missing_str``.
    settings : SerializerSettings
        Serialization parameters controlling base, precision and separators.

    Returns
    -------
    str
        A string representation of the time series suitable for feeding into
        a text language model.
    """
    arr = np.asarray(arr).flatten()
    # Ensure values are within the allowed range
    if np.any(np.abs(arr[~np.isnan(arr)]) > settings.max_val):
        raise ValueError(f"abs(arr) must be <= settings.max_val={settings.max_val}")
    # Determine sign prefixes
    if not settings.signed:
        if np.any(arr[~np.isnan(arr)] < 0):
            raise ValueError("unsigned serialization requires non‑negative values")
        plus_sign = minus_sign = ""
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign
    # Convert numbers to digit vectors
    vnum2repr = partial(vec_num2repr, base=settings.base,
                        prec=settings.prec, max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    is_missing = np.isnan(arr)
    # Helper to join digits into a string with bit separators
    def tokenize(bits: List) -> str:
        return "".join([settings.bit_sep + str(b) for b in bits])
    bit_strings: List[str] = []
    for sign, digits, missing in zip(sign_arr, digits_arr, is_missing):
        # When variable length representations are allowed, strip leading zeros
        d = digits
        if not settings.fixed_length:
            nonzero = np.where(np.array(d) != 0)[0]
            if len(nonzero) == 0:
                d = np.array([0])
            else:
                d = d[nonzero[0]:]
            # Optionally insert a decimal point token
            if settings.decimal_point:
                prec = settings.prec
                d = np.concatenate([d[:-prec], np.array([settings.decimal_point]), d[-prec:]])
        digits_str = tokenize(d)
        prefix = plus_sign if sign == 1 else minus_sign
        if missing:
            bit_strings.append(settings.missing_str)
        else:
            bit_strings.append(prefix + digits_str)
    # Join the representations with the time separator and append one final separator
    result = settings.time_sep.join(bit_strings)
    result = result + settings.time_sep
    return result


def deserialize_str(bit_str: str, settings: SerializerSettings,
                    ignore_last: bool = False, steps: Optional[int] = None
                    ) -> Optional[np.ndarray]:
    """Deserialize a textual representation back into a NumPy array.

    Parameters
    ----------
    bit_str : str
        A string created by :func:`serialize_arr`.
    settings : SerializerSettings
        Settings that match those used during serialization.
    ignore_last : bool, optional
        If ``True``, discard the final time step.  This can be useful
        when generated sequences may be truncated by the language model.
    steps : int, optional
        If provided, stop after parsing this many time steps.

    Returns
    -------
    np.ndarray or None
        The deserialized array.  If parsing fails at the very first time
        step, ``None`` is returned; otherwise any successfully parsed prefix
        will be returned even if an error is encountered later.
    """
    # Split the input into time steps
    time_chunks = [chunk for chunk in bit_str.split(settings.time_sep) if chunk]
    if ignore_last:
        time_chunks = time_chunks[:-1]
    if steps is not None:
        time_chunks = time_chunks[:steps]
    # Prepare lists to collect sign and digit information
    sign_list: List[int] = []
    digits_list: List[List[int]] = []
    max_bit_pos = int(np.ceil(np.log(settings.max_val) / np.log(settings.base)))
    # Attempt to parse each chunk
    try:
        for chunk in time_chunks:
            # Determine the sign
            if settings.signed:
                if chunk.startswith(settings.minus_sign):
                    sign = -1
                    remainder = chunk[len(settings.minus_sign):]
                elif chunk.startswith(settings.plus_sign):
                    sign = 1
                    remainder = chunk[len(settings.plus_sign):]
                else:
                    raise ValueError(
                        f"Signed representation requires a prefix of '{settings.minus_sign}' or '{settings.plus_sign}'")
            else:
                sign = 1
                remainder = chunk
            # Split remainder into individual digit symbols
            if settings.bit_sep:
                # Remove any leading whitespace before splitting
                bits = [b[:1] for b in remainder.lstrip().split(settings.bit_sep)]
            else:
                bits = list(remainder.lstrip())
            if settings.fixed_length and len(bits) != max_bit_pos + settings.prec:
                raise ValueError(
                    f"Fixed length representation requires {max_bit_pos + settings.prec} digits, got {len(bits)}")
            # Collect numeric digits, skipping the decimal point symbol
            digits: List[int] = []
            for b in bits:
                if b == settings.decimal_point:
                    continue
                if b.isdigit():
                    digits.append(int(b))
                else:
                    # Abort on unparseable characters
                    raise ValueError(f"Unexpected symbol '{b}' in digit string")
            sign_list.append(sign)
            digits_list.append(digits)
    except Exception:
        # If an error occurs, return whatever has been parsed so far (unless none)
        if not digits_list:
            return None
    # Pad digit arrays to equal length by pre‑pending zeros
    if digits_list:
        max_len = max(len(d) for d in digits_list)
        padded = [([0] * (max_len - len(d)) + d) for d in digits_list]
        sign_arr = np.array(sign_list)
        digits_arr = np.array(padded)
        return vec_repr2num(sign_arr, digits_arr, settings.base,
                             settings.prec, settings.half_bin_correction)
    return None