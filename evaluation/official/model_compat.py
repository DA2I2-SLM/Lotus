"""Runtime shim (no repo edits) to control how the base LLM is loaded.

OpenTSLMSP / OpenTSLMFlamingo hardcode
    AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16,
                                         attn_implementation="sdpa")
gemma-3 is documented to need eager attention for numerically correct
generation; sdpa(+bf16) can yield degenerate output. This module monkey-patches
from_pretrained so we can override attn_implementation / dtype before the model
is built, without touching repo source.

Call `apply(attn_implementation=..., torch_dtype=...)` BEFORE constructing the
OpenTSLM model.
"""
import transformers
from transformers import AutoModelForCausalLM as _AMC

_STATE = {"attn": None, "dtype": None, "patched": False}
_ORIG = _AMC.from_pretrained


def _patched(cls, *args, **kwargs):
    if _STATE["attn"]:
        kwargs["attn_implementation"] = _STATE["attn"]
    if _STATE["dtype"] is not None:
        kwargs["torch_dtype"] = _STATE["dtype"]
    return _ORIG.__func__(cls, *args, **kwargs)


def apply(attn_implementation=None, torch_dtype=None):
    _STATE["attn"] = attn_implementation
    _STATE["dtype"] = torch_dtype
    if not _STATE["patched"]:
        _AMC.from_pretrained = classmethod(_patched)
        _STATE["patched"] = True
    print(f"[model_compat] LLM load override: attn_implementation="
          f"{attn_implementation!r}, torch_dtype={torch_dtype}", flush=True)
