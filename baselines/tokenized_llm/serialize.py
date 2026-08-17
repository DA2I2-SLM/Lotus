# SPDX-License-Identifier: MIT
"""Turn a Phase2DownstreamDataset sample into a plain-text prompt.

This is the "tokenized LLM" baseline: instead of projecting patch embeddings into
the backbone's embedding space (what OpenTSLMSPFixPos2D does), the raw numbers are
written into the prompt as text and the model reads them as tokens.

The reason this needs its own module rather than a one-liner is the token budget.
Measured with the Qwen2.5 tokenizer at 2 decimal places:

    task      channels x length   values   ~tokens
    timemqa        1 x   32           32       191
    har            3 x  128          384     2,303
    pamap2         3 x  160          480     2,879
    sleep          1 x 1504        1,504     9,023
    ecg           12 x 1024       12,288    73,724      <-- 2.3x over a 32k context

So ECG cannot be represented losslessly as text at any usable context length. We
subsample uniformly to fit a budget and RECORD the decimation factor on every
sample, so the eval can report honestly which columns were degraded and by how
much. Silently truncating would make the ECG column look like a model result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _as_list(series: Any) -> List[float]:
    if hasattr(series, "tolist"):
        series = series.tolist()
    out: List[float] = []
    for v in series:
        if isinstance(v, (list, tuple)):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return out


def decimate(values: Sequence[float], keep: int) -> Tuple[List[float], int]:
    """Uniformly subsample `values` down to at most `keep` points.

    Returns (subsampled, stride). stride == 1 means nothing was dropped.
    Uniform striding rather than truncation: keeping the first N samples of a
    12-lead ECG would discard whole beats, whereas striding preserves the
    envelope at reduced resolution.
    """
    vals = list(values)
    n = len(vals)
    if keep <= 0 or n <= keep:
        return vals, 1
    idx = np.linspace(0, n - 1, num=keep).round().astype(int)
    return [vals[i] for i in idx], max(1, int(round(n / keep)))


def format_series(values: Sequence[float], precision: int = 2) -> str:
    return ", ".join(f"{v:.{precision}f}" for v in values)


def serialize_sample(
    sample: Dict[str, Any],
    *,
    max_values_total: int = 1200,
    precision: int = 2,
    serializer: str = "ours",
) -> Dict[str, Any]:
    """Render one dataset sample as (prompt, target) plus provenance metadata.

    `max_values_total` is the budget across ALL channels of the sample. At 2 dp a
    value costs ~6 tokens, so 1200 values ~ 7.2k tokens of series, which leaves
    room for the captions, question and generated rationale inside an 8k window.
    """
    captions: List[str] = list(sample.get("time_series_text") or [])
    series_list = list(sample.get("time_series") or [])

    n_series = max(1, len(series_list))
    per_series = max(1, max_values_total // n_series)

    blocks: List[str] = []
    strides: List[int] = []
    n_raw = 0
    for i, raw in enumerate(series_list):
        vals = _as_list(raw)
        n_raw += len(vals)
        kept, stride = decimate(vals, per_series)
        strides.append(stride)
        caption = captions[i] if i < len(captions) else f"Series {i}:"
        note = "" if stride == 1 else f" (subsampled every {stride}th point)"
        if serializer == "llmtime":
            # Gruver et al. (2023) encoding: digit-level spacing, no decimal
            # point, per-series rescale. ~17% more tokens per value than "%.2f"
            # (7.02 vs 5.99 on gemma's tokenizer), which the caller must fund
            # out of the same budget -- so llmtime runs decimate harder.
            from serialize_llmtime import format_series_llmtime
            rendered = format_series_llmtime(kept, precision=3)
        else:
            rendered = format_series(kept, precision)
        blocks.append(f"{caption}{note}\n[{rendered}]")

    prompt = "\n".join(
        [str(sample.get("pre_prompt", "")).strip()]
        + blocks
        + [str(sample.get("post_prompt", "")).strip()]
    ).strip()

    return {
        "prompt": prompt,
        "target": str(sample.get("answer", "")).replace("<eos>", "").strip(),
        "type": sample.get("type"),
        "label": sample.get("label"),
        # provenance for honest reporting
        "n_values_raw": n_raw,
        "n_values_kept": sum(min(len(_as_list(s)), per_series) for s in series_list),
        "max_stride": max(strides) if strides else 1,
        "decimated": any(s > 1 for s in strides),
    }
